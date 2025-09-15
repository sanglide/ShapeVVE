import numpy as np
from typing import Dict, List, Union

from torch import Tensor


def extract_subsequences(data: np.ndarray, L: List[int], D_S: Dict[int, List[int]]) -> dict[str, Tensor]:
    """
    Extracts all possible subsequences of specified lengths and dimensions from a multi-dimensional time series.

    Parameters:
    -----------
    data : np.ndarray
        Input time series data with shape [N, Dimension, Time], where:
        - N: Number of samples
        - Dimension: Number of dimensions/features
        - Time: Time steps
    L : List[int]
        List of subsequence lengths to extract
    D_S : Dict[int, List[int]]
        Dictionary specifying which dimensions to select for each sample.
        Keys are sample indices (0-based), values are lists of dimension indices (0-based).

    Returns:
    --------
    Dict[str, List[Tensor]]
        A dictionary where keys are in format 'sample_{n}_dim_{d}_len_{l}' and values are lists of subsequences.
        Each subsequence has shape [n',d,l] where n' is the number of selected shapelets in this parameter(N-l+1), l is the subsequence length and d is the number of selected dimensions.
    """

    N, num_dims, T = data.shape
    result = {}

    # Validate input parameters
    if not all(0 <= sample_idx < N for sample_idx in D_S.keys()):
        raise ValueError("Sample indices in D_S must be within [0, N-1]")

    if not all(0 <= dim_idx < num_dims for dim_list in D_S.values() for dim_idx in dim_list):
        raise ValueError("Dimension indices in D_S must be within [0, num_dims-1]")

    max_L = max(L) if L else 0
    if max_L > T:
        raise ValueError(f"Maximum subsequence length {max_L} exceeds time dimension {T}")

    # Process each sample specified in D_S
    for sample_idx in range(N):
        for idx, dim_list in D_S.items():
            sample_data = data[sample_idx]  # Shape: [Dimension, Time]
            selected_dims = sample_data[dim_list, :]  # Shape: [len(dim_list), Time]

            # For each specified subsequence length
            for l in L:
                if l > T:
                    continue  # Skip lengths longer than time dimension

                num_subsequences = T - l + 1
                subsequences = []

                # Extract all possible subsequences of length l
                for i in range(num_subsequences):
                    subsequence = selected_dims[:, i:i + l]  # Shape: [len(dim_list), l]
                    subsequences.append(subsequence)
                    # print(f'{i} : {i + l}')
                    # print(subsequence)
                    # print("----------------")

                # Store results with descriptive key
                key = f"sample_{sample_idx}_dim_{dim_list}_len_{l}"
                result[key] = Tensor(subsequences)

    return result


# # Example usage:
# if __name__ == "__main__":
#     # Create dummy data: 2 samples, 3 dimensions, 10 time steps
#     dummy_data = np.random.rand(2, 3, 5)
#     print(dummy_data)
#
#     # Define parameters
#     L = [3, 4, 5]  # Subsequence lengths
#     D_S = {0: [1, 2], 1: [0, 2]}  # Dimensions to select for each sample
#
#     # Extract subsequences
#     subsequences = extract_subsequences(dummy_data, L, D_S)
#
#     # Print results
#     for key, subseq_list in subsequences.items():
#         print(f"{key}: {len(subseq_list)} subsequences")
#         for subseq in subseq_list:
#             print(f'{subseq}\n')