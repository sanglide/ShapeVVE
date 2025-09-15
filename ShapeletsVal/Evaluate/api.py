import pathlib
import time
import warnings
from datetime import timedelta
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import check_random_state

from ShapeletsVal.DataLoader.fetcher import DataFetcher
from ShapeletsVal.DataLoader.noisify import add_no_noise
from ShapeletsVal.Evaluate.util import filter_kwargs
from ShapeletsVal.PredictorModel.api import Model
from ShapeletsVal.ShapeletsVE import DataEvaluator
from ShapeletsVal.ShapeletsVE.metrics import Metrics


class ExperimentMediatorChannel:
    def __init__(
            self,
            fetcher: DataFetcher,
            pred_model: Model,
            train_kwargs: Optional[dict[str, Any]] = None,
            metric_name: Optional[Union[str, Metrics, Callable]] = None,
            output_dir: Optional[Union[str, pathlib.Path]] = None,
            raises_error: bool = False,
    ):
        self.fetcher = fetcher
        self.pred_model = pred_model
        self.train_kwargs = {} if train_kwargs is None else train_kwargs

        if callable(metric_name):
            print("metric_name is callable")
            self.metric = metric_name
        elif metric_name is not None:
            print("metric_name is Metrics(metric_name)")
            self.metric = Metrics(metric_name)
        else:
            print("metric_name is Metrics.ACCURACY")
            self.metric = Metrics.ACCURACY if self.fetcher.one_hot else Metrics.NEG_MSE
        self.data_evaluators = []

        if output_dir is not None:
            self.set_output_directory(output_dir)
        self.timings = {}
        self.raise_error = raises_error

    @classmethod
    def setup(
            cls,
            dataset_name: str,
            cache_dir: Optional[Union[str, pathlib.Path]] = None,
            force_download: bool = False,
            train_count: Union[int, float] = 0,
            valid_count: Union[int, float] = 0,
            test_count: Union[int, float] = 0,
            add_noise: Union[Callable[[DataFetcher], dict[str, Any]], str] = add_no_noise,
            noise_kwargs: Optional[dict[str, Any]] = None,
            random_state: Optional[RandomState] = None,
            pred_model: Optional[Model] = None,
            train_kwargs: Optional[dict[str, Any]] = None,
            metric_name: Optional[Union[str, Metrics, Callable]] = None,
            output_dir: Optional[Union[str, pathlib.Path]] = None,
            raises_error: bool = False,
    ):
        """Create a DataFetcher from args and passes it into the init."""
        random_state = check_random_state(random_state)
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            force_download=force_download,
            random_state=random_state,
            train_count=train_count,
            valid_count=valid_count,
            test_count=test_count,
            add_noise=add_noise,
            noise_kwargs=noise_kwargs,
        )

        return cls(
            fetcher=fetcher,
            pred_model=pred_model,
            train_kwargs=train_kwargs,
            metric_name=metric_name,
            output_dir=output_dir,
            raises_error=raises_error,
        )

    def get_valid_acc(self,data_evaluators:list[DataEvaluator], *args, **kwargs):
        result=[]
        for data_val in data_evaluators:
            result.append(data_val.get_valid_acc())
        return result

    def compute_data_values(
            self,shapelets_num_sum, data_evaluators: list[DataEvaluator], *args, **kwargs
    ):
        kwargs = {**kwargs, **self.train_kwargs}
        for data_val in data_evaluators:
            try:
                start_time = time.perf_counter()

                data_val.set_shapelets_num_sum(shapelets_num_sum)

                result = data_val.train(
                    self.fetcher, self.pred_model, self.metric, *args, **kwargs
                )
                print(result)

                self.data_evaluators.append(
                    result
                )
                end_time = time.perf_counter()
                delta = timedelta(seconds=end_time - start_time)

                self.timings[data_val] = delta

                print(f"Elapsed time {data_val!s}: {delta}")
            except Exception as ex:
                if self.raise_error:
                    raise ex

                warnings.warn(
                    f"""
                           An error occured during training, however training all evaluators
                           takes a long time, so we will be ignoring the evaluator:
                           {data_val!s} and proceeding.

                           The error is as follows: {ex!s}
                           """,
                    stacklevel=10,
                )

        self.num_data_eval = len(self.data_evaluators)

        return self

    def evaluate(
            self,
            exper_func: Callable[[DataEvaluator, DataFetcher, ...], dict[str, Any]],
            save_output: bool = False,
            **exper_kwargs,):
        data_eval_perf = {}
        filtered_kwargs = filter_kwargs(
            exper_func,
            train_kwargs=self.train_kwargs,
            metric=self.metric,
            model=self.pred_model,
            **exper_kwargs,
        )
        # self.evaluator里面是dve的list，self.pred_model是训练的模型，但是这样的模型是训练过的吗？
        # input: evaluater,fetcher,dim_indices
        # output: dict[str,float]
        # evaluator.data_values是什么

        print(f'line in 218 in evaluate()')
        print(self.data_evaluators)
        for data_val in self.data_evaluators:
            # 调用noisy detection函数
            print(f'line in 221 in evaluate()')
            eval_resp = exper_func(data_val, self.fetcher, **filtered_kwargs)
            print(f'line in 223 in evaluate()')
            data_eval_perf[str(data_val)] = eval_resp
            print(f'line in 225 in evaluate()')

        print("[DEBUG] api.evaluate() Input shapes:", {k: v.shape for k, v in locals().items() if hasattr(v, 'shape')})

        # index=[DataEvaluator.DataEvaluator]
        df_resp = pd.DataFrame.from_dict(data_eval_perf, "index")
        print(f"[DEBUG] api.evaluate() df_resp : {df_resp.size}")
        df_resp = df_resp.explode(list(df_resp.columns))
        print(f"[DEBUG] api.evaluate() df_resp : {df_resp.size}")

        if save_output:
            self.save_output(f"{exper_func.__name__}.csv", df_resp)
        return df_resp
    def evaluate_shapeletVE_model(self, data_evaluators: list[DataEvaluator],*args, **kwargs):
        result=[]
        for data_val in data_evaluators:
            result_temp = data_val.evaluate_shapeletVE_model()
            result.append(result_temp)
        return result


