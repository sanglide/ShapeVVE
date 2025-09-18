from collections import OrderedDict

import torch
from matplotlib import pyplot as plt
from torch import nn, tensor
from torch.optim import Adam

from ShapeVVE.PredictorModel.LearningShapeletsOrigin import LearningShapelets, MinEuclideanDistBlock, \
    ShapeletsDistBlocks, LearningShapeletsModel, ShapeletsDistanceLoss, ShapeletsSimilarityLoss


# Min...Block中最好只维护一个shapelets，对应一个维度选择
class MinEuclideanDistBlockDimensionSelection(MinEuclideanDistBlock):
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = x.unfold(2, self.shapelets_size, 1).contiguous()
        x = self.compute_shapelet_distances(x, self.shapelets)
        return x

    def compute_shapelet_distances(self, input_tensor, shapelets):
        # input_tensor:[D，T] shapelets [D,L] , self.selection [D]
        n_samples,dimension,window,window_size=input_tensor.shape
        num_shapelets=len(shapelets[0])
        input_tensor_flatten = (input_tensor.permute(0, 2, 1, 3).flatten(0, 1))
        input_tensor_flatten = input_tensor_flatten.permute(1, 0, 2)
        input_tensor_flatten = input_tensor_flatten.to(torch.float32)
        shapelets = shapelets.to(torch.float32)
        distances = torch.cdist(
            input_tensor_flatten,
            shapelets,
            p=2
        )

        # distance [dimension,n_sample*window_num, num_shapelets]
        if hasattr(self, 'selection'):

            # print(f'n_samples,dimension, window,window_size: {n_samples,dimension,window,window_size}')
            # print(f'selection : {self.selection.shape}')

            distances=(distances * self.selection.T.unsqueeze(1)).mean(dim=0)
            # [n_sample*window_num, num_shapelets]
            mask_shapelets = self.selection.T.unsqueeze(-1)
            # [num_shapelets, dimension, 1]
            self.shapelets.data = (self.shapelets * mask_shapelets)  # 广播机制自动对齐维度
            # [dimension,num_shapelets,shapelets_length]
            distances=distances.view(n_samples, window, num_shapelets)
            # [n_sample,window_num,num_shapelets]
            min_distance, _ = torch.min(distances, dim=1)

        else:
            mean_distance = distances.mean(dim=0)
            reshaped_distance = mean_distance.view(input_tensor.size(0), input_tensor.size(2),
                                                   -1)
            min_distance, _ = torch.min(reshaped_distance, dim=1)
        return min_distance

    def set_selection(self,selection):
        self.selection=selection

class ShapeletsDistBlocksDimensionSelection(ShapeletsDistBlocks):
    def __init__(self, shapelets_size_and_len, in_channels=1, dist_measure='euclidean', to_cuda=True):
        super(ShapeletsDistBlocks, self).__init__()
        self.to_cuda = to_cuda
        self.shapelets_size_and_len = OrderedDict(sorted(shapelets_size_and_len.items(), key=lambda x: x[0]))
        self.in_channels = in_channels
        self.dist_measure = dist_measure
        if dist_measure == 'euclidean':
            # 生成一系列的 module
            # 关键看怎么更新参数的
            self.blocks = nn.ModuleList(
                [MinEuclideanDistBlockDimensionSelection(
                    shapelets_size=shapelets_size,
                    num_shapelets=num_shapelets,  # 使用实际的shapelet数量
                    in_channels=in_channels,
                    to_cuda=self.to_cuda
                )
                    for shapelets_size, num_shapelets in self.shapelets_size_and_len.items()
                ])
            # print(f"number of blocks {len(self.blocks)}")
        else:
            raise ValueError("dist_measure must be either of 'euclidean', 'cross-correlation', 'cosine'")

    def set_selector(self,selector):
        #做映射，现在已有的接口是shapelets_size, num_shapelets
        index=0
        for module in self.blocks:
            k=module.num_shapelets
            # selection=self.downsample_binary_matrix(selector,k)
            module.set_selection(selector[index:index+k])
            index+=k


class LearningShapeletsModelDimensionSelection(LearningShapeletsModel):
    def __init__(self, shapelets_size_and_len, in_channels=1, num_classes=2, dist_measure='euclidean',
                 to_cuda=True):
        super(LearningShapeletsModel, self).__init__()

        self.to_cuda = to_cuda
        self.shapelets_size_and_len = shapelets_size_and_len
        self.num_shapelets = sum(shapelets_size_and_len.values())
        self.shapelets_blocks = ShapeletsDistBlocksDimensionSelection(in_channels=in_channels,
                                                    shapelets_size_and_len=shapelets_size_and_len,
                                                    dist_measure=dist_measure, to_cuda=to_cuda)
        self.linear = nn.Linear(self.num_shapelets, num_classes)
        if self.to_cuda:
            self.cuda()
    def set_selector(self,selector):
        self.shapelets_blocks.set_selector(selector)

class LearningShapeletsDimensionSelection(LearningShapelets):
    def __init__(self, shapelets_size_and_len, loss_func, in_channels=1, num_classes=2,
                 dist_measure='euclidean', verbose=0, to_cuda=True,device='cuda', k=0, l1=0.0, l2=0.0,lr=0.01, weight_decay=1e-4):
        super(LearningShapelets, self).__init__()

        torch.cuda.set_device(device)
        # 核心就是包括一系列 blocks，对应产生shapelet的modules
        self.model = LearningShapeletsModelDimensionSelection(shapelets_size_and_len=shapelets_size_and_len,
                                            in_channels=in_channels, num_classes=num_classes, dist_measure=dist_measure,
                                            to_cuda=to_cuda)
        self.to_cuda = to_cuda
        if self.to_cuda:
            self.model.cuda()

        self.shapelets_size_and_len = shapelets_size_and_len
        self.loss_func = loss_func
        self.verbose = verbose
        self.optimizer = None

        if not all([k == 0, l1 == 0.0, l2 == 0.0]) and not all([k > 0, l1 > 0.0]):
            raise ValueError("For using the regularizer, the parameters 'k' and 'l1' must be greater than zero."
                             " Otherwise 'k', 'l1', and 'l2' must all be set to zero.")
        self.k = k
        self.l1 = l1
        self.l2 = l2
        self.loss_dist = ShapeletsDistanceLoss(dist_measure=dist_measure, k=k)
        self.loss_sim_block = ShapeletsSimilarityLoss()
        # add a variable to indicate if regularization shall be used, just used to make code more readable
        self.use_regularizer = True if k > 0 and l1 > 0.0 else False
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.set_optimizer(optimizer)


    def set_selector(self,selector):
        # selector [N,D,1]
        self.selector = selector
        self.model.set_selector(selector)

    def update(self, x, y):
        # self.model.set_selector(self.selector)
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def log_losses(self,losses,path):
        if isinstance(losses, tuple):
            # 如果有多个损失（使用正则化时）
            losses_ce, losses_dist, losses_sim = losses
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(losses_ce)
            plt.title('Cross-Entropy Loss')
            plt.xlabel('Iteration')

            plt.subplot(1, 3, 2)
            plt.plot(losses_dist)
            plt.title('Distance Loss')
            plt.xlabel('Iteration')

            plt.subplot(1, 3, 3)
            plt.plot(losses_sim)
            plt.title('Similarity Loss')
            plt.xlabel('Iteration')

        else:
            # 只有交叉熵损失
            plt.figure(figsize=(6, 4))
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Cross-Entropy Loss')

        plt.tight_layout()
        plt.savefig(path)
