from typing import Union,Dict

import numpy as np
import torch
from sklearn.utils import check_random_state
from torch.utils.data import Dataset

from ShapeVVE.DataLoader.fetcher import DataFetcher
from ShapeVVE.DataLoader.util import IndexTransformDataset
from ShapeVVE.ShapeletsVE.util import FuncEnum


def add_no_noise(fetcher: DataFetcher,
    noise_rate: float = 0.2,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> dict[str, Union[Dataset, np.ndarray]]:
    x_train = np.array(fetcher.x_train, dtype=np.float64)  # Shape: (N, D, T)
    x_valid = np.array(fetcher.x_valid, dtype=np.float64)
    x_test = np.array(fetcher.x_test, dtype=np.float64)

    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "x_test": x_test,
        "noisy_train_indices": np.array([]),
    }


def add_gauss_noise_by_dimension(
    fetcher: DataFetcher,
    noise_rate: float = 0.2,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> dict[str, Union[Dataset, np.ndarray]]:
    """Add Gaussian noise to covariates at the dimension level.

    Parameters
    ----------
    fetcher : DataFetcher
        DataFetcher object housing the data to have noise added to.
    noise_rate : float
        Proportion of dimensions to add noise to (per sample).
    mu : float, optional
        Mean of the Gaussian distribution (default: 0.0).
    sigma : float, optional
        Standard deviation of the Gaussian distribution (default: 1.0).

    Returns
    -------
    dict[str, Union[Dataset, np.ndarray]]
        Dictionary containing:
        - "x_train": Training data with noise added to selected dimensions.
        - "x_valid": Validation data with noise added to selected dimensions.
        - "x_test": Original test data (unchanged).
        - "noisy_dim_indices": Boolean mask of shape (N, D) indicating which dimensions were perturbed.
    """
    rs = check_random_state(fetcher.random_state)

    x_train = np.array(fetcher.x_train, dtype=np.float64)  # Shape: (N, D, T)
    x_valid = np.array(fetcher.x_valid, dtype=np.float64)
    x_test = np.array(fetcher.x_test, dtype=np.float64)
    N_train, D, T = x_train.shape
    N_valid, _, _ = x_valid.shape
    N_test,_,_=x_test.shape

    # Generate noisy indices: (N, D) boolean mask
    noisy_dims_train = rs.random((N_train, D)) < noise_rate  # Shape: (N_train, D)
    noisy_dims_test = rs.random((N_test, D)) < noise_rate  # Shape: (N_train, D)
    noisy_dims_valid = rs.random((N_valid, D)) < noise_rate  # Shape: (N_valid, D)

    # Generate Gaussian noise for all (N, D, T), but only apply where noisy_dims=True
    noise_train = rs.normal(mu, sigma, size=(N_train, D, T)) * noisy_dims_train[..., None]
    noise_test = rs.normal(mu, sigma, size=(N_test, D, T)) * noisy_dims_test[..., None]
    noise_valid = rs.normal(mu, sigma, size=(N_valid, D, T)) * noisy_dims_valid[..., None]

    if isinstance(fetcher.x_train, Dataset):
        # Handle PyTorch Dataset (dynamic noise addition)
        noise_add_train = torch.tensor(noise_train, dtype=torch.float32)
        noise_add_test = torch.tensor(noise_test, dtype=torch.float32)
        noise_add_valid = torch.tensor(noise_valid, dtype=torch.float32)

        x_train = IndexTransformDataset(
            x_train,
            lambda data, idx: data + noise_add_train[idx],  # Add noise to the entire (D, T) slice
        )
        x_test = IndexTransformDataset(
            x_test,
            lambda data, idx: data + noise_add_test[idx],  # Add noise to the entire (D, T) slice
        )
        x_valid = IndexTransformDataset(
            x_valid,
            lambda data, idx: data + noise_add_valid[idx],
        )
    else:
        # Handle numpy array
        x_train = x_train + noise_train
        x_test = x_test + noise_test
        x_valid = x_valid + noise_valid

    # print("*********************************************")
    # print(noisy_dims_valid)
    # print(noise_valid)
    # print(x_valid.shape)
    # print("*********************************************")

    return {
        "x_train": x_train,
        "x_valid": x_valid,
        "x_test": x_test,
        "noisy_train_indices": noisy_dims_train,
    }

class NoiseFunc(FuncEnum):
    ADD_GAUSS_NOISE_BY_DIMENSION = FuncEnum.wrap(add_gauss_noise_by_dimension)
