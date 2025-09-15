import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from ShapeletsVal.PredictorModel.api import TorchClassMixinChannel, TorchPredictMixin
from ShapeletsVal.PredictorModel.grad import TorchGradMixin


class LogisticRegressionChannel(TorchClassMixinChannel, TorchPredictMixin, TorchGradMixin):
    """Initialize LogisticRegression

    Parameters
    ----------
    input_dim : int
        Size of the input dimension of the LogisticRegression
    num_classes : int
        Size of the output dimension of the LR, outputs selection probabilities
    """

    def __init__(self, input_dim: tuple, num_classes: int):
        super().__init__()

        self.input_dim = np.prod(input_dim)
        self.num_classes = num_classes

        self.linear = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Logistic Regression.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output Tensor of logistic regression
        """

        x = self.linear(x)
        if self.num_classes <= 2:
            x = F.sigmoid(x)
        else:
            # Equivalent to sigmoid for classes of size 2.
            x = F.softmax(x, dim=1)
        return x

    def score(self, X: torch.Tensor, y: torch.tensor) -> float:
        """Compute accuracy score.

        Parameters
        ----------
        X : torch.Tensor
            Input features
        y : torch.Tensor
            True labels

        Returns
        -------
        float
            Accuracy score
        """
        with torch.no_grad():
            predictions = self.forward(X)
            if self.num_classes <= 2:
                predicted_labels = (predictions > 0.5).long()
            else:
                predicted_labels = predictions.argmax(dim=1)
            return accuracy_score(y.cpu().numpy(), predicted_labels.cpu().numpy())