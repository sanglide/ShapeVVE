import copy
import warnings
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, TypeVar, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier, DummyRegressor
from torch.utils.data import DataLoader, Dataset, default_collate

from ShapeletsVal.DataLoader.util import CatDataset

Self = TypeVar("Self", bound="Model")


class Model(ABC):
    """Abstract class of Models. Provides a template for models."""

    Models: ClassVar[dict[str, Self]] = {}

    def __init_subclass__(cls, *args, **kwargs):
        """Registers Model types, used as part of the CLI."""
        super().__init_subclass__(*args, **kwargs)
        cls.Models[cls.__name__.lower()] = cls

    @abstractmethod
    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        *args,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Self:
        """Fits the model on the training data.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        args : tuple[Any]
            Additional positional args
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, must be passed in as key word arg,
            by default None
        kwargs : dict[str, Any]
            Addition key word args

        Returns
        -------
        self : object
            Returns self for api consistency with sklearn.
        """
        return self

    @abstractmethod
    def predict(self, x: Union[torch.Tensor, Dataset], *args, **kwargs) -> torch.Tensor:
        """Predict the label from the input covariates data.

        Parameters
        ----------
        x : torch.Tensor | Dataset
            Input data covariates


        Returns
        -------
        torch.Tensor
            Output predictions based on the input
        """

    def clone(self) -> Self:
        """Clone Model object.

        Copy and returns object representing current state. We often take a base
        model and train it several times, so we need to have the same initial conditions
        Default clone implementation.

        Returns
        -------
        self : object
            Returns deep copy of model.
        """
        return copy.deepcopy(self)


class TorchModel(Model, nn.Module):
    """Torch Models have a device they belong to and shared behavior"""

    @property
    def device(self):
        return next(self.parameters()).device


class TorchClassMixinChannel(TorchModel):
    """Classifier Mixin for Torch Neural Networks."""

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
    ):
        """Fits the model on the training data.

        Fits a torch classifier Model object using ADAM optimizer and cross
        categorical entropy loss.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        criterion = F.binary_cross_entropy if self.num_classes == 2 else F.cross_entropy
        dataset = CatDataset(x_train, y_train, sample_weight)

        self.train()
        for _ in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(
                dataset, batch_size, shuffle=True, pin_memory=True
            ):
                # Moves data to correct device
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                optimizer.zero_grad()
                outputs = self.__call__(x_batch)

                if sample_weight is not None:
                    '''
                    *todo：
                    重点修改部位：
                    原本的size是 outputs torch.Size([32, 10]) y_batch torch.Size([32, 10])  weights[0] torch.Size([32, 1])
                    意思是 每一个样本的预测值和真实值，以及权重值之间需要 结合起来计算一个loss
                    这里需要根据我们自己的设计写loss公式，但是对于这部分，先暂时将维度加权求和做减法，或者先暂时去掉这一项
                    '''
                    # F.cross_entropy doesn't support sample_weights
                    loss = criterion(outputs, y_batch, reduction="mean")
                    # loss = criterion(outputs, y_batch, reduction="none")
                    # loss = (loss * weights[0].to(device=self.device)).mean()
                else:
                    loss = criterion(outputs, y_batch, reduction="mean")


                # todo:需要看以下这里加个retain_graph=True是否合理呢
                loss.backward(retain_graph=True)  # Compute gradient
                optimizer.step()  # Updates weights

        return self


class TorchClassMixin(TorchModel):
    """Classifier Mixin for Torch Neural Networks."""

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
    ):
        """Fits the model on the training data.

        Fits a torch classifier Model object using ADAM optimizer and cross
        categorical entropy loss.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weights : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        criterion = F.binary_cross_entropy if self.num_classes == 2 else F.cross_entropy
        dataset = CatDataset(x_train, y_train, sample_weight)

        self.train()
        for _ in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(
                dataset, batch_size, shuffle=True, pin_memory=True
            ):
                # Moves data to correct device
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                optimizer.zero_grad()
                outputs = self.__call__(x_batch)

                if sample_weight is not None:

                    print(f'line 221 in model/api.py')
                    print(outputs.size(), y_batch.size(),len(weights),weights[0].size())
                    # F.cross_entropy doesn't support sample_weights
                    loss = criterion(outputs, y_batch, reduction="none")
                    loss = (loss * weights[0].to(device=self.device)).mean()
                else:
                    loss = criterion(outputs, y_batch, reduction="mean")

                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights

        return self


class TorchRegressMixin(TorchModel):
    """Regressor Mixin for Torch Neural Networks."""

    def fit(
        self,
        x_train: Union[torch.Tensor, Dataset],
        y_train: Union[torch.Tensor, Dataset],
        sample_weight: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.01,
    ):
        """Fits the regression model on the training data.

        Fits a torch regression Model object using ADAM optimizer and MSE loss.

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weight : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        criterion = F.mse_loss
        dataset = CatDataset(x_train, y_train, sample_weight)

        self.train()
        for _ in range(int(epochs)):
            # *weights helps check if we passed weights into the Dataloader
            for x_batch, y_batch, *weights in DataLoader(
                dataset,
                batch_size,
                shuffle=True,
                pin_memory=True,
            ):
                # Moves data to correct device
                x_batch = x_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                optimizer.zero_grad()
                y_hat = self.__call__(x_batch)

                if sample_weight is not None:
                    # F.cross_entropy doesn't support sample_weight
                    loss = criterion(y_hat, y_batch, reduction="none")
                    loss = (loss * weights[0].to(device=self.device)).mean()
                else:
                    loss = criterion(y_hat, y_batch, reduction="mean")

                loss.backward()  # Compute gradient
                optimizer.step()  # Updates weights

        return self


class TorchPredictMixin(TorchModel):
    """Torch ``.predict()`` method mixin for Torch Neural Networks."""

    def predict(self, x: Union[torch.Tensor, Dataset]) -> torch.Tensor:
        """Predict output from input tensor/data set.

        Parameters
        ----------
        x : torch.Tensor
            Input covariates

        Returns
        -------
        torch.Tensor
            Predicted tensor output
        """
        if isinstance(x, Dataset):
            x = next(iter(DataLoader(x, batch_size=len(x), pin_memory=True)))
        x = x.to(device=self.device)

        self.eval()
        with torch.no_grad():
            y_hat = self.__call__(x)

        return y_hat


def to_numpy(tensors: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    """Mini function to move tensor to CPU for sk-learn."""
    return tuple(t.numpy(force=True) for t in default_collate(tensors))