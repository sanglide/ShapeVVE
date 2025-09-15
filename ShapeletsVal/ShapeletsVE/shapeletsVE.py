"""
input : shapelets [N_s,D,T_s]
2. ShapeletsVE： 目前是mlp，未来看看有无其他处理（如加入CNN/RNN支持可变长度），输入是所有的shapelets，输出是价值向量。目前不支持可变长度
output : shapelets weight [N_s,1]
"""

from typing import Optional
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ShapeletsVal.DataLoader.util import CatDataset
from ShapeletsVal.Evaluate.util import plot_selected_dimensions, plot_selected_shapelets
from ShapeletsVal.ShapeletsVE.api import DataEvaluator, ModelMixin



class DVRLChannel(DataEvaluator, ModelMixin):

    def __init__(
            self,
            hidden_dim: int = 100,
            layer_number: int = 5,
            comb_dim: int = 10,
            rl_epochs: int = 1000,
            rl_batch_size: int = 32,
            lr: float = 0.01,
            threshold: float = 0.9,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            random_state: Optional[RandomState] = None,
            log_path: str = 'runs/experiment1',
            l=0.001
    ):
        super().__init__()
        torch.cuda.set_device(device)
        self.num_ones=0
        self.hidden_dim = hidden_dim
        self.layer_number = layer_number
        self.comb_dim = comb_dim
        self.device = device
        self.rl_epochs = rl_epochs
        self.rl_batch_size = rl_batch_size
        self.lr = lr
        self.threshold = threshold
        self.random_state = check_random_state(random_state)
        self.log_path=log_path
        self.writer = SummaryWriter(log_dir=log_path)  # 日志保存目录
        self.l=l

    def input_data(
            self,
            x_train: torch.Tensor,  # (N, D, T)
            y_train: torch.Tensor,  # (N, 1)
            x_valid: torch.Tensor,
            y_valid: torch.Tensor,
            x_test: torch.Tensor,
            y_test: torch.Tensor,
    ):

        x_train = self.replace_invalid_values(x_train)
        y_train = self.replace_invalid_values(y_train)
        x_valid = self.replace_invalid_values(x_valid)
        y_valid = self.replace_invalid_values(y_valid)
        x_test = self.replace_invalid_values(x_test)
        y_test = self.replace_invalid_values(y_test)

        self.x_train = x_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.x_valid = x_valid.to(self.device)
        self.y_valid = y_valid.to(self.device)
        self.x_test = x_test.to(self.device)
        self.y_test = y_test.to(self.device)

        N, D, T = x_train.shape
        N, y_dim = y_train.shape
        self.value_estimator = DataValueEstimatorRLChannel(
            x_dim=D * T,
            y_dim=y_dim,
            hidden_dim=self.hidden_dim,
            layer_number=self.layer_number,
            comb_dim=self.comb_dim,
            random_state=self.random_state,
            dimension=D,
        ).to(self.device)
        return self

    def _evaluate_baseline_models(self, *args, **kwargs):
        """Train baseline models and compute prediction differences.

        Trains:
        - ori_model: 在训练集上训练的基准模型。
        - val_model: 在验证集上训练的模型（用于计算预测差异）。
        """
        # todo: 这里预测模型的输入需要展平，这个操作是否合适？目前看来只要mask操作是在这一步之前做的就没关系，需要验证一下
        x_train = self.x_train.to(self.device)
        # x_train = x_train.flatten(start_dim=1)

        x_valid = self.x_valid.to(self.device)
        # x_valid = x_valid.flatten(start_dim=1)

        y_train=self.y_train.squeeze().to(self.device)
        y_valid=self.y_valid.squeeze().to(self.device)

        # print(f'line 85 in shapeletsVE.py, evaluate_baseline_models()')

        # Final model
        self.final_model = self.pred_model.clone()

        # 基准模型（全训练集）
        self.ori_model = self.pred_model.clone()
        self.ori_model.fit(x_train, y_train, *args, **kwargs)

        # 验证模型（全验证集）
        self.val_model = self.pred_model.clone()
        self.val_model.fit(x_valid, y_valid, *args, **kwargs)

        # 计算基准性能
        y_valid_pred = self.ori_model.predict(x_valid)
        self.valid_perf = self.evaluate(y_valid, y_valid_pred)
        print(f'valid_pref : {self.valid_perf}')

        # 计算预测差异 |y_train - val_model(x_train)|
        y_train_pred = self.val_model.predict(x_train)
        self.y_pred_diff = torch.abs(y_train - y_train_pred).unsqueeze(-1)

    def set_shapelets_num_sum(self,shapelets_num_sum):
        self.shapelets_num_sum = shapelets_num_sum

    import torch

    def representative_sampling(self,dim_weights, k):

        # 判断是否存在NaN或inf
        # has_nan_or_inf = torch.isnan(dim_weights).any() or torch.isinf(dim_weights).any()
        # print("是否存在NaN或inf:", has_nan_or_inf)

        N, D = dim_weights.shape

        # Step 1: Compute frequency of each dimension (mean over samples)
        freq = dim_weights.float().mean(dim=0)  # Shape [D]

        # Step 2: Compute score for each sample (sum of selected dimensions' frequencies)
        scores = (dim_weights.float() * freq).sum(dim=1)  # Shape [N]

        # Step 3: Convert scores to probabilities
        probs = scores / scores.sum()  # Shape [N], sums to 1

        # Step 4: Sample k indices according to probabilities (with replacement)
        sampled_indices = torch.multinomial(probs, num_samples=k, replacement=True)

        # Step 5: Gather the selected samples
        sampled_k = dim_weights[sampled_indices]  # Shape [k, D]

        return sampled_k

    def train_data_values(self, *args, num_workers: int = 0, **kwargs):
        # Initialize baseline models
        self._evaluate_baseline_models(*args, **kwargs)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.value_estimator.parameters(), lr=self.lr)
        criterion = DveLossChannel(threshold=self.threshold)
        # Data loader
        data = CatDataset(self.x_train, self.y_train, self.y_pred_diff)
        dataloader = DataLoader(
            data,
            batch_size=self.rl_batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        # print(f'bathch number: {self.rl_batch_size} {dataloader.__len__()}')

        count,batch_count=0,0
        rewards,prefs,losses_value=[],[],[]

        # Training loop
        for epoch in range(self.rl_epochs):  # 外层循环控制epoch
            self.value_estimator.train()  # 训练模式
            epoch_loss = 0.0
            count = count + 1
            for x_batch, y_batch, y_hat_batch in dataloader:
                batch_count+=1
                # print(f'train batch : {x_batch.size} {y_batch.size} {y_hat_batch.size}')
                print(f'the {count} -th epochs {batch_count}-th batch')
                x_batch_ve = x_batch.to(self.device).to(torch.float32)
                y_batch_ve = y_batch.to(self.device).to(torch.float32)
                y_hat_batch_ve = y_hat_batch.to(self.device).unsqueeze(1).to(torch.float32)

                optimizer.zero_grad()


                dim_weights = self.value_estimator(x_batch_ve, y_batch_ve, y_hat_batch_ve) # (N,D)
                dim_weights_sum=self.representative_sampling(dim_weights,self.shapelets_num_sum)

                select_mask = torch.bernoulli(dim_weights_sum)  # (K, D)

                # print(f'------ select_mask ------')
                torch.set_printoptions(precision=3, sci_mode=False)
                # print(dim_weights)

                x_valid = self.x_valid
                y_batch_new=y_batch.squeeze()

                new_model = self.pred_model.clone()
                new_model.set_selector(select_mask)
                losses=new_model.fit(x_batch, y_batch_new, *args, sample_weight=select_mask.detach().to(self.device), **kwargs)


                # Compute reward
                sum_1_proper=select_mask.sum().item()/(select_mask.size(0) * select_mask.size(1))
                y_valid_pred = new_model.predict(x_valid)
                dvrl_perf = self.evaluate(self.y_valid, y_valid_pred)
                # reward = dvrl_perf - self.valid_perf - self.l *sum_1_proper
                reward = 1 - self.valid_perf - self.l *sum_1_proper

                # Update DVE
                print(f'dim_weights:{dim_weights.shape},selected_mask:{select_mask.shape},reward:{reward}')
                loss = criterion(dim_weights_sum, select_mask, reward)
                loss.backward()
                optimizer.step()

                self.log_weights(count, dim_weights, select_mask, reward,dvrl_perf,sum_1_proper, loss)
                rewards.append(reward)
                prefs.append(dvrl_perf)
                losses_value.append(loss.detach().to(self.device).item())
                self.writer.add_scalar('reward/train', reward, count)
                self.writer.add_scalar('loss/train', loss, count)

                epoch_loss += loss.item() * x_batch.size(0)  # 按batch加权

            epoch_loss /= len(x_batch)  # 平均损失
            print(f"Epoch {epoch + 1}/{self.rl_epochs}, Loss: {epoch_loss:.4f}")

        # Final model training with learned weights
        dim_weights = self.value_estimator(self.x_train.to(self.device).to(torch.float32),
                                           self.y_train.to(self.device).to(torch.float32),
                                           self.y_pred_diff)

        dim_weights_sum=self.representative_sampling(dim_weights,self.shapelets_num_sum)
        select_mask = torch.bernoulli(dim_weights_sum)  # (N, D)

        num_ones = torch.sum(select_mask).float()/select_mask.numel()
        self.num_ones = num_ones.detach().to(self.device).item()
        print(f'num ones: {num_ones}')

        self.final_model.set_selector(select_mask)
        losses=self.final_model.fit(self.x_train, self.y_train.squeeze(), *args, **kwargs)

        self.final_model.log_losses(losses,f'{self.log_path}/shapelets_loss.png')
        self.log_rewards_losses(rewards,prefs,losses_value)
        self.log_shapelets(dim_weights_sum,select_mask,self.final_model.get_shapelets())

        return self
    def log_rewards_losses(self,rewards,prefs,losses):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(rewards)
        plt.title('rewards')
        plt.xlabel('Iteration')

        plt.subplot(1, 3, 2)
        plt.plot(prefs)
        plt.title('pref acc')
        plt.xlabel('Iteration')

        plt.subplot(1, 3, 3)
        plt.plot(losses)
        plt.title('Loss')
        plt.xlabel('Iteration')

        plt.tight_layout()
        plt.savefig(f'{self.log_path}/rewards_losses.png')

    def log_shapelets(self,dim_weights,select_mask,shapelets):
        with open(f"{self.log_path}/shapelets.txt", "a") as f:
            f.write(f"dim_weights: \n{dim_weights}\nselect_mask: \n{select_mask}\n")  # 保存为可读格式
            count=0
            for shapelet in shapelets:
                f.write(f"shapelets {count}: \n{shapelet}\n")
                # plot_selected_shapelets(shapelet.detach().to(self.device),select_mask[count],f'{self.log_path}/shapelets/{count}_shapelets.png')
                count += 1

    def log_weights(self,count,dim_weights,select_mask,reward,dvrl_perf,sum_1_proper,losses):
        with open(f"{self.log_path}/weights.txt", "a") as f:
            f.write(f"Epoch {count}: reward= {reward} ,pref_acc: {dvrl_perf},sum_1: {sum_1_proper}, losses= {losses}\n")
            f.write(f"dim_weights: \n{dim_weights}\nselect_mask: \n{select_mask}\n")  # 保存为可读格式

    def evaluate_data_values(self) -> torch.Tensor:
        """返回每个训练样本的维度权重 (N, D)。"""
        # 用验证模型计算预测差异
        # todo：展平

        x_train = self.x_train
        # x_train = x_train.flatten(start_dim=1)

        y_train_pred = self.val_model.predict(x_train)
        y_hat = torch.abs(self.y_train - y_train_pred)

        # 初始化输出
        weights = []

        # 分批次计算（避免内存溢出）
        with torch.no_grad():
            for x_batch, y_batch, y_hat_batch in DataLoader(
                    CatDataset(self.x_train, self.y_train, y_hat),
                    batch_size=self.rl_batch_size,
                    shuffle=False
            ):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_hat_batch = y_hat_batch.to(self.device)

                # 获取维度权重 (N, D)
                batch_weights = self.value_estimator(x_batch, y_batch, y_hat_batch)
                weights.append(batch_weights.cpu())

        return torch.cat(weights)  # (N, D)

    def replace_invalid_values(self,tensor, replace_value=0.0):
        # 检测 NaN 和 inf
        mask = torch.isnan(tensor) | torch.isinf(tensor)
        # 替换非法值为 0
        tensor = torch.where(mask, torch.full_like(tensor, replace_value), tensor)
        return tensor

    def get_valid_acc(self):
        return self.valid_perf

    def evaluate_shapeletVE_model(self):
        # print(self.x_test)
        x_test,y_test=self.x_test.to(self.device).to(torch.float32),self.y_test.to(self.device).unsqueeze(1).to(torch.float32)
        y_pred=self.final_model.predict(x_test)

        correct = (y_pred == y_test.squeeze()).sum().item()
        accuracy = correct / len(y_test)
        metrics = {
            'accuracy': accuracy,
            'sum_dimension': self.num_ones,
        }
        return metrics

class DataValueEstimatorRLChannel(nn.Module):
    def __init__(
            self,
            x_dim: int,
            y_dim: int,
            hidden_dim: int,
            layer_number: int,
            comb_dim: int,
            dimension: int,
            random_state: Optional[RandomState] = None,
    ):
        super().__init__()

        if random_state is not None:  # Can't pass generators to nn.Module layers
            torch.manual_seed(check_random_state(random_state).tomaxint())

        # 动态计算 D 和 T（需在forward中通过输入形状推断）
        self.x_dim = x_dim  # 保存总维度 D*T
        self.y_dim = y_dim

        # MLP 结构
        self.mlp = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(layer_number - 3)],
            nn.Linear(hidden_dim, comb_dim),
            nn.ReLU()
        )

        self.dim_weight_head = nn.Sequential(
            nn.Linear(comb_dim + y_dim, comb_dim),
            nn.ReLU(),
            nn.Linear(comb_dim, dimension),
            nn.Sigmoid()
        )

    def forward(
            self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        # Flatten input dimension in case it is more than 3D
        x = x.flatten(start_dim=2)
        y = y.flatten(start_dim=1)
        y_hat = y_hat.flatten(start_dim=1)
        N, D, T = x.shape

        # 展平时间维度：(N, D, T) -> (N, D*T)
        x_flat = x.flatten(start_dim=1)

        # 拼接标签：(N, D*T + 1)
        out = torch.cat([x_flat, y], dim=1)

        # 特征提取
        out = self.mlp(out)  # (N, comb_dim)


        # 拼接预测差异
        out = torch.cat([out, y_hat], dim=1)  # (N, comb_dim + 1)
        # 生成权重（先输出1维，再扩展为D维）
        w = self.dim_weight_head(out)  # (N, 1)
        # w = w.expand(-1, D)  # (N, D)
        return w

class DveLossChannel(nn.Module):
    """Loss function for dimension-wise data valuation (N*D output)."""

    def __init__(self, threshold: float = 0.9, exploration_weight: float = 1e3):
        super().__init__()
        self.threshold = threshold
        self.exploration_weight = exploration_weight

    def forward(
            self,
            pred_dataval: torch.Tensor,  # (N, D)
            selector_input: torch.Tensor,  # (N, D)
            reward_input: float,
    ) -> torch.Tensor:
        # Binary cross-entropy loss (per dimension)
        loss = F.binary_cross_entropy(pred_dataval, selector_input, reduction="sum")

        # Reward-weighted loss
        reward_loss = reward_input * loss
        #reward-input=ACC-a* sum_1

        # Exploration penalty (encourage mean weights to stay near threshold)
        mean_weight = torch.mean(pred_dataval)
        search_loss = (F.relu(mean_weight - self.threshold) +
                       F.relu((1 - self.threshold) - mean_weight))

        reward_loss = reward_loss + (self.exploration_weight * search_loss)

        return reward_loss