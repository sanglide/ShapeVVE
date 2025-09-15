import torch


class ShapeletsCNN(torch.nn.Module):
    def __init__(self, shapelet_num, shapelet_dimension, shapelet_length, hidden_size, class_num):
        super(ShapeletsCNN, self).__init__()
        self.shapelet_num = shapelet_num
        self.shapelet_dimension = shapelet_dimension
        self.shapelet_length = shapelet_length
        self.hidden_size = hidden_size
        self.class_num = class_num

        # Define the CNN layers for shapelets extraction
        self.conv = [
            torch.nn.Parameter(torch.randn(1, 1, shapelet_dimension, shapelet_length, dtype=torch.float32))
            for _ in range(shapelet_num)
        ]

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.shapelet_num, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.class_num),
        )

    def forward(self, x):
        # x (ts_num, ts_dimension, ts_length)

        # 1. 计算B的平方和
        shapelet_square = [torch.sum(conv ** 2) for conv in self.conv]
        # shapelet_num

        # 2. 计算每个窗口的平方和 (使用卷积)
        x = x.unsqueeze(dim=1)
        # (ts_num, 1, ts_dimension, ts_length)

        # 创建全1卷积核 (输出通道, 输入通道, 高度, 宽度)
        ones_kernel = torch.ones(1, 1, self.shapelet_dimension, self.shapelet_length)
        # (1, 1, self.shapelet_dimension, self.shapelet_length)

        # 计算每个(X,Y)窗口的平方和 (N, 1, H-X+1, W-Y+1)
        window_square = torch.nn.functional.conv2d(x ** 2, ones_kernel, stride=1, padding=0)
        # (ts_num, 1, ts_dimension - shapelet_dimension + 1, ts_length - shapelet_length + 1)

        # 3. 计算A和B的互相关 (使用卷积)
        # 计算互相关
        cross_corr = [torch.nn.functional.conv2d(x, conv, stride=1, padding=0) for conv in self.conv]
        # shapelet_num * (ts_num, 1, ts_dimension - shapelet_dimension + 1, ts_length - shapelet_length + 1)

        # 4. 计算欧氏距离平方: ||A-B||² = A² + B² - 2A·B
        # 广播B_sq到相同形状
        shapelet_square = [shapelet_square_item.expand(window_square.shape) for shapelet_square_item in shapelet_square]
        # 计算欧式距离平方
        distance_matrix = [(window_square + shapelet_square[index] - 2 * cross_corr[index]) for index in range(self.shapelet_num)]
        # shapelet_num * (ts_num, 1, ts_dimension - shapelet_dimension + 1, ts_length - shapelet_length + 1)

        # 直接使用卷积学习（不具备可解释性）
        # x = x.unsqueeze(1)
        # # (ts_num, 1, ts_dimension, ts_length)
        #
        # distance_matrix = [conv(x) for conv in self.conv]
        # # shapelet_num * (ts_num, 1, ts_dimension - shapelet_dimension + 1, ts_length - shapelet_length + 1)
        #

        # 取最小值作为距离
        distance = torch.stack([torch.min(matrix.view(matrix.shape[0], -1), dim=1)[0] for matrix in distance_matrix], dim=1)
        # (ts_num, shapelet_num)

        # 分类
        classification = self.classifier(distance)
        # (ts_num, class_num)

        return distance, classification

    def loss(self, distance, y_hat, y):
        # acc
        accuracy_loss = torch.nn.functional.cross_entropy(y_hat, y)
        # other

        return accuracy_loss


if __name__ == "__main__":
    shapelet_num = 2
    shapelet_dimension = 2
    shapelet_length = 20
    hidden_size = 100
    class_num = 2

    ts = torch.randn(5, 3, 30)
    shapelet_network = ShapeletsCNN(shapelet_num, shapelet_dimension, shapelet_length, hidden_size, class_num)
    result = shapelet_network(ts)
    print(result)