"""Experiments to test :py:class:`~opendataval.dataval.api.DataEvaluator`.

Experiments pass into :py:meth:`~opendataval.experiment.api.ExperimentMediator.evaluate`
and :py:meth:`~opendataval.experiment.api.ExperimentMediator.plot` evaluate performance
of one :py:class:`~opendataval.dataval.api.DataEvaluator` at a time.
"""

from pathlib import Path
import pandas as pd
import torch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch.utils.data import Subset
from typing import Dict, List, Optional, Any
from matplotlib.axes import Axes
from ShapeVVE.DataLoader.fetcher import DataFetcher
from ShapeVVE.ShapeletsVE.api import DataEvaluator
from ShapeVVE.Evaluate.util import f1_score, oned_twonn_clustering
from ShapeVVE.ShapeletsVE.metrics import Metrics
from ShapeVVE.PredictorModel.api import Model
from ShapeVVE.ShapeletsVE.util import get_name

from typing import Dict, Optional, Sequence
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def noisy_dimension_detection(
        evaluator: DataEvaluator,
        fetcher: Optional[DataFetcher] = None,
        dim_indices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate ability to identify noisy DIMENSIONS using F1 score.

    Parameters
    ----------
    evaluator : DataEvaluator
        Evaluator providing dimension-level data values (shape: N x D)
    fetcher : DataFetcher, optional
        DataFetcher containing noisy dimension mask (shape: N x D)
    dim_indices : np.ndarray, optional
        Boolean array indicating noisy dimensions (shape: N x D)

    Returns
    -------
    dict[str, float]
        - "kmeans_f1": F1 score of identifying noisy dimensions
        - "precision": Precision score
        - "recall": Recall score
    """
    # 获取维度级数据值 (N x D)
    data_values = evaluator.data_values

    # 获取噪声维度标记 (N x D 布尔数组)
    noisy_dim_mask = (
        fetcher.noisy_train_indices if isinstance(fetcher, DataFetcher) else dim_indices
    )

    # 验证形状匹配
    assert data_values.shape == noisy_dim_mask.shape, "Shapes must match!"

    # 展平数据用于聚类 (N*D,)
    flat_values = data_values.ravel()
    flat_noise = noisy_dim_mask.ravel().astype(int)

    # 1D 两聚类（低值簇和高值簇）
    low_value_cluster, _ = oned_twonn_clustering(flat_values)

    # 创建预测标签 (1=低价值/可能是噪声，0=高价值)
    predicted = np.zeros(len(flat_values), dtype=int)
    predicted[list(low_value_cluster)] = 1

    # 计算各项指标
    f1 = f1_score(flat_noise, predicted)
    precision = precision_score(flat_noise, predicted)
    recall = recall_score(flat_noise, predicted)

    return {
        "kmeans_f1": f1,
        "precision": precision,
        "recall": recall
    }


def remove_high_low_dimensions(
        evaluator: DataEvaluator,
        fetcher: Optional[DataFetcher] = None,
        model: Optional[Model] = None,
        data: Optional[Dict[str, Any]] = None,
        percentile: float = 0.05,
        plot: Optional[Axes] = None,
        metric: Metrics = Metrics.ACCURACY,
        train_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[float]]:
    """Evaluate performance after masking dimensions per-sample based on data values."""

    # 1. 数据加载和验证
    if isinstance(fetcher, DataFetcher):
        x_train, y_train,*_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    assert x_train.ndim == 3, "Input must be (N, D, T)"
    data_values = evaluator.data_values  # Shape: (N, D)
    model = model or evaluator.pred_model
    train_kwargs = train_kwargs or {}

    print(f'line 151 in exper_methods.py')
    print(f'data values : {data_values}')

    # 2. 保留样本级维度重要性 (N, D)
    per_sample_dim_importance = data_values
    print(f"per_sample_dim_importance 1 : {per_sample_dim_importance}")

    # 3. 对每个样本的维度单独排序 (N, D)
    per_sample_sorted_dims = np.argsort(per_sample_dim_importance, axis=1)
    print(f"per_sample_sorted_dims 2 : {per_sample_sorted_dims}")

    # 4. 确定每次mask的维度数
    N, D, T = x_train.shape
    dims_per_step = max(1, int(D * percentile))
    steps = np.arange(0, D + dims_per_step, dims_per_step) / D

    # 5. 修正后的评估函数（接收step参数）
    def evaluate_with_masked_dims(mask_strategy: str, current_step: int):
        """根据策略和当前step mask维度并评估"""
        model_copy = model.clone()

        # 计算当前step需要mask的维度数
        mask_count = current_step * dims_per_step

        # 对每个样本应用不同的mask
        masks = []
        for i in range(N):
            if mask_strategy == "low":
                # mask低价值维度
                dims_to_mask = per_sample_sorted_dims[i, :mask_count]
            else:  # "high"
                # mask高价值维度
                dims_to_mask = per_sample_sorted_dims[i, -mask_count:] if mask_count > 0 else []
            # 创建样本级mask (D,)
            mask = np.ones(D, dtype=bool)
            mask[dims_to_mask] = False
            masks.append(mask)

        # print(f'len masks: {len(masks)}')

        # 转换为tensor (N, D, 1)
        mask_tensor = torch.tensor(np.array(masks)[:, :, None],
                                   dtype=torch.float32,
                                   device=x_train.device)
        # print(f'mask_tensor.shape: {mask_tensor.shape}')
        # print(f'x_train.shape: {x_train.shape}')
        # print(f'x_test.shape: {x_test.shape}')

        # 应用mask
        x_train_masked = x_train * mask_tensor

        # todo:flatten
        x_train_masked = x_train_masked.flatten(start_dim=1)
        x_test_flatten = x_test.flatten(start_dim=1)

        model_copy.fit(x_train_masked, y_train, **train_kwargs)
        y_pred = model_copy.predict(x_test_flatten)
        return metric(y_test, y_pred)

    # 6. 逐步mask维度并评估
    low_masked_scores, high_masked_scores = [], []

    for step in range(len(steps)):
        # 计算当前step的mask数量
        current_mask_count = step * dims_per_step

        # 评估两种策略
        low_score = evaluate_with_masked_dims("low", step)
        high_score = evaluate_with_masked_dims("high", step)

        low_masked_scores.append(low_score)
        high_masked_scores.append(high_score)

    # 7. 返回结果
    metric_name = get_name(metric)
    results = {
        "axis": steps.tolist(),
        f"mask_low_value_dims_{metric_name}": low_masked_scores,
        f"mask_high_value_dims_{metric_name}": high_masked_scores,
    }

    # 8. 可视化（保持不变）
    if plot is not None:
        plot.plot(steps, low_masked_scores, "o-", label="Masking low-value dims")
        plot.plot(steps, high_masked_scores, "x-", label="Masking high-value dims")
        plot.axhline(low_masked_scores[0], linestyle="--", color="gray", label="Baseline")

        plot.set_xlabel("Proportion of dimensions masked")
        plot.set_ylabel(f"Test {metric_name}")
        plot.legend()
        plot.set_title(f"Dimension Masking Impact\n{evaluator}")

    return results

def discover_corrupted_dimensions(
        evaluator: DataEvaluator,
        fetcher: Optional[DataFetcher] = None,
        data: Optional[Dict[str, Any]] = None,
        percentile: float = 0.05,
        plot: Optional[Axes] = None,
) -> Dict[str, List[float]]:
    """Evaluate discovery of noisy DIMENSIONS in low data value points.

    Parameters
    ----------
    evaluator : DataEvaluator
        Evaluator providing data values per dimension.
    fetcher : DataFetcher, optional
        DataFetcher containing noisy dimension indices (shape: (N, D)).
    data : dict, optional
        Alternative input with keys:
        - "x_train": Training data (shape: N, D, T)
        - "noisy_dim_indices": Boolean mask of noisy dimensions (shape: N, D)
    percentile : float, optional
        Percentile of data points to search per iteration (default: 0.05).
    plot : Axes, optional
        Matplotlib Axes for plotting results.

    Returns
    -------
    Dict[str, List[float]]
        Results containing:
        - "axis": Proportion of data inspected.
        - "corrupt_found": Proportion of noisy dimensions found.
        - "optimal": Optimal discovery rate (upper bound).
        - "random": Random discovery rate (baseline).
    """
    # 1. Load data and noisy dimension indices
    if isinstance(fetcher, DataFetcher):
        x_train = fetcher.x_train  # Shape: (N, D, T)
        noisy_dim_indices = fetcher.noisy_train_indices  # Shape: (N, D)
    else:
        x_train = data["x_train"]
        noisy_dim_indices = data["noisy_dim_indices"]

    data_values = evaluator.data_values  # Shape: (N, D) (假设评估器已支持维度级数据值)

    # 2. Validate shapes
    N, D = noisy_dim_indices.shape
    assert data_values.shape == (N, D), "Data values must be per-dimension!"

    # 3. Flatten and sort dimensions by data value (ascending: low value = likely noisy)
    flat_values = data_values.ravel()  # Shape: (N*D,)
    flat_noise_mask = noisy_dim_indices.ravel()  # Shape: (N*D,)
    sorted_indices = np.argsort(flat_values, kind="stable")  # 升序排序

    # 4. Calculate discovery rates
    num_total_noisy = flat_noise_mask.sum()
    num_points = N * D
    num_per_bin = max(round(num_points * percentile), 1)
    bins = range(0, num_points + num_per_bin, num_per_bin)

    corrupt_found = []
    for bin_end in bins:
        inspected_indices = sorted_indices[:bin_end]
        num_found = flat_noise_mask[inspected_indices].sum()
        corrupt_found.append(num_found / num_total_noisy)

    # 5. Generate output
    x_axis = [i / num_points for i in bins]
    results = {
        "axis": x_axis,
        "corrupt_found": corrupt_found,
        "optimal": [min(x / (num_total_noisy / num_points), 1.0) for x in bins],
        "random": x_axis,
    }

    # 6. Plot if requested
    if plot is not None:
        plot.plot(x_axis, corrupt_found, "o-", label="Evaluator")
        plot.plot(x_axis, results["optimal"], "--", label="Optimal")
        plot.plot(x_axis, results["random"], ":", label="Random")
        plot.set_xlabel("Proportion of dimensions inspected")
        plot.set_ylabel("Proportion of noisy dimensions found")
        plot.legend()
        plot.set_title(f"{evaluator}\nNoise Detection (Dimension-Level)")

    return results

def save_dataval(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    indices: Optional[list[int]] = None,
    output_path: Optional[Path] = None,
):
    """Save the indices and the respective data values of the DataEvaluator."""
    train_indices = (
        fetcher.train_indices if isinstance(fetcher, DataFetcher) else indices
    )
    data_values = evaluator.data_values

    data = {"indices": train_indices, "data_values": data_values}

    if output_path:
        df_data = {str(evaluator): data}
        print(f'line 579 in exper_method.py')
        print(df_data)
        df = pd.DataFrame.from_dict(df_data, "index")
        df.explode(list(df.columns)).to_csv(output_path)

    return data


def increasing_bin_removal(
    evaluator: DataEvaluator,
    fetcher: Optional[DataFetcher] = None,
    model: Optional[Model] = None,
    data: Optional[dict[str, Any]] = None,
    bin_size: int = 1,
    plot: Optional[Axes] = None,
    metric: Metrics = Metrics.ACCURACY,
    train_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, list[float]]:
    """Evaluate accuracy after removing data points with data values above threshold.

    For each subplot, displays the proportion of the data set with data values less
    than the specified data value (x-axis) and the performance of the model when all
    data values greater than the specified data value is removed. This implementation
    was inspired by V. Feldman and C. Zhang in their paper [1] where the same principle
    was applied to memorization functions.

    References
    ----------
    .. [1] V. Feldman and C. Zhang,
        What Neural Networks Memorize and Why: Discovering the Long Tail via
        Influence Estimation,
        arXiv.org, 2020. Available: https://arxiv.org/abs/2008.03703.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing training and valid data points, by default None
    model : Model, optional
        Model which performance will be evaluated, if not defined,
        uses evaluator's model to evaluate performance if evaluator uses a model
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training and
        test data with the following keys:

        - **"x_train"** Training covariates
        - **"y_train"** Training labels
        - **"x_test"** Testing covariates
        - **"y_test"** Testing labels
    bin_size : float, optional
        We look at bins of equal size and find the data values cutoffs for the x-axis,
        by default 1
    plot : Axes, optional
        Matplotlib Axes to plot data output, by default None
    metric : Metrics | Callable[[Tensor, Tensor], float], optional
        Name of DataEvaluator defined performance metric which is one of the defined
        metrics or a Callable[[Tensor, Tensor], float], by default accuracy
    train_kwargs : dict[str, Any], optional
        Training key word arguments for training the pred_model, by default None

    Returns
    -------
    Dict[str, list[float]]
        dict containing the thresholds of data values examined, proportion of training
        data points removed, and performance after those data points were removed.

        - **"axis"** -- Thresholds of data values examined. For a given threshold,
            considers the subset of data points with data values below.
        - **"frac_datapoints_explored"** -- Proportion of data points with data values
            below the specified threshold
        - **f"{metric}_at_datavalues"** -- Performance metric when data values
            above the specified threshold are removed
    """
    data_values = evaluator.data_values
    model = model if model is not None else evaluator.pred_model
    curr_model = model.clone()
    if isinstance(fetcher, DataFetcher):
        x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    else:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    num_points = len(data_values)

    # Starts with 10 data points
    bins_indices = [*range(5, num_points - 1, bin_size), num_points - 1]
    frac_datapoints_explored = [(i + 1) / num_points for i in bins_indices]

    sorted_indices = np.argsort(data_values)
    x_axis = data_values[sorted_indices[bins_indices]] / np.max(data_values)

    perf = []
    train_kwargs = train_kwargs if train_kwargs is not None else {}

    for bin_end in bins_indices:
        coalition = sorted_indices[:bin_end]

        new_model = curr_model.clone()
        new_model.fit(
            Subset(x_train, coalition),
            Subset(y_train, coalition),
            **train_kwargs,
        )
        y_hat = new_model.predict(x_test)
        perf.append(metric(y_hat, y_test))

    eval_results = {
        "frac_datapoints_explored": frac_datapoints_explored,
        f"{get_name(metric)}_at_datavalues": perf,
        "axis": x_axis,
    }

    if plot is not None:  # Removing everything above this threshold
        plot.plot(x_axis, perf)

        plot.set_xticks([])
        plot.set_ylabel(get_name(metric))
        plot.set_title(str(evaluator))

        divider = make_axes_locatable(plot)
        frac_inspected_plot = divider.append_axes("bottom", size="40%", pad="5%")

        frac_inspected_plot.fill_between(x_axis, frac_datapoints_explored)
        frac_inspected_plot.set_xlabel("Data Values Threshold")
        frac_inspected_plot.set_ylabel("Trainset Fraction")

    return eval_results
