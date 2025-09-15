import numpy as np
from sklearn.model_selection import train_test_split
from torch import Tensor
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_from_arff_file
from typing import Optional
import os
import zipfile
import requests
from pathlib import Path

def load_arff(
        dataset_name: str,
        dir: Optional[str] = None,
        split_method: str = "from_test",
        val_ratio: float = 0.2,
        random_state: int = 42,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Parameters
    ----------
    dataset_name : str
        Base name of the dataset (e.g., "GunPoint" for "GunPoint_TRAIN.arff").
    dir : str, optional
        Directory containing the dataset files.
    split_method : str, optional
        How to split validation set:
        - "presplit": Use predefined VALIDATION.arff file (default)
        - "from_train": Split from training set
        - "from_test": Split from test set
    val_ratio : float, optional
        Proportion for validation split when using "from_train" or "from_test".
    random_state : int, optional
        Random seed for reproducible splits.

    Returns
    -------
    tuple
        (x_train, y_train, x_val, y_val, x_test, y_test)
    """

    x_train, y_train = load_from_arff_file(f"{dir}/{dataset_name}/{dataset_name}_TRAIN.arff")
    x_test, y_test = load_from_arff_file(f"{dir}/{dataset_name}/{dataset_name}_TEST.arff")

    if split_method == "presplit":
        try:
            x_val, y_val = load_from_arff_file(f"{dir}/{dataset_name}/{dataset_name}_VALID.arff")
        except FileNotFoundError:
            print(f"Warning: No predefined VALIDATION.arff found, falling back to splitting from training set")
            split_method = "from_train"

    if split_method != "presplit":
        if split_method == "from_train":
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train,
                test_size=val_ratio,
                random_state=random_state,
                stratify=y_train
            )
        elif split_method == "from_test":
            x_test, x_val, y_test, y_val = train_test_split(
                x_test, y_test,
                test_size=val_ratio,
                random_state=random_state,
                stratify=y_test
            )
        else:
            raise ValueError(f"Invalid split_method: {split_method}")

    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train, y_val, y_test])
    label_encoder.fit(all_labels)

    y_train = label_encoder.transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    print(f"Shapes - Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    return x_train, y_train, x_val, y_val, x_test, y_test


def ensure_dataset_available(dataset_path: str):
    download_url = "http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_arff.zip"
    dataset_dir = Path(dataset_path)
    zip_path = dataset_dir / "Multivariate2018_arff.zip"

    # Create directory if it doesn't exist
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        print(f"Dataset not found at {dataset_path}, downloading...")

        # Download the zip file
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)

            print("Download and extraction completed successfully.")

            # Remove the zip file after extraction
            os.remove(zip_path)

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise

    return dataset_dir


def get_UEA_dataset(dataset_path: str, dataset_name: str):
    if not os.path.exists(dataset_path):
        dataset_path = ensure_dataset_available(dataset_path)
    return load_arff(dataset_name, dataset_path)
