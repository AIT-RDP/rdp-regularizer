"""
Forecasting of grid points whose deadline (grid_time + lag_time) has passed without
any measured sample arriving: the channel must emit anyway to stay strictly regular,
so the value is extrapolated from history only.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

import numpy as np

from ..config import ChannelConfig
from ..logger import LOGGER
from ..sample import Sample
from .algo import daily_naive_reconstruct, knn_reconstruct, soft_threshold_svd_reconstruct
from .chronos import ChronosForecasterBase
from .util import array_to_samples, as_numpy, matrix_fill, nan_ffill, samples_to_array


class Forecaster(Protocol):
    """
    Fills remaining holes in a sample sequence from history only (no right bound).
    """

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        """
        Return a sequence with the same timestamps, length, and order. Known values
        are left untouched. Remaining ``None`` holes are filled with quality
        ``'forecast'``.
        """
        ...


class ConstFillForecaster:
    """
    Naive forecast: ffill last observation, NaN when none.
    """
    def __init__(self, **kwargs: Any):
        LOGGER.warning(f'ConstFillForecaster unknown kwargs: {kwargs}')

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        values = nan_ffill(samples_to_array(samples))
        return array_to_samples(samples, values, 'forecast')


class DailyNaiveForecaster:
    """
    Forecast using daily naive approach.
    """
    def __init__(self, **kwargs: Any):
        LOGGER.warning(f'DailyNaiveForecaster unknown kwargs: {kwargs}')

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'forecast',
            daily_naive_reconstruct
        )


class KNNForecaster:
    """
    Forecast using KNN.
    """
    def __init__(self, k: int = 5, **kwargs: Any):
        LOGGER.warning(f'KNNForecaster unknown kwargs: {kwargs}')
        self.k = k

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'forecast',
            lambda M: knn_reconstruct(M, k=self.k)
        )


class SoftThresholdSVDForecaster:
    """
    Forecast using soft-thresholded SVD.
    """
    def __init__(self, max_rank: int = 3, shrinkage: Optional[float] = None, **kwargs: Any):
        LOGGER.warning(f'SoftThresholdSVDForecaster unknown kwargs: {kwargs}')
        self.max_rank = max_rank
        self.shrinkage = shrinkage

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'forecast',
            lambda M: soft_threshold_svd_reconstruct(
                M, max_rank=self.max_rank, shrinkage=self.shrinkage
            )
        )


class ChronosBoltForecaster(ChronosForecasterBase):
    """
    Zero-shot Chronos-Bolt forecast (on CPU). Univariate context only.
    """

    _default_model_path = 'amazon/chronos-bolt-tiny'

    def _pipeline_class(self):
        """Chronos-Bolt pipeline from the ``chronos`` extra."""
        chronos = self._require_extra('chronos')
        return chronos.BaseChronosPipeline

    def _pipeline_inputs(self, context: np.ndarray) -> np.ndarray:
        """Univariate 1-D context for Chronos-Bolt."""
        return np.asarray(context, dtype=np.float32).reshape(-1)

    def _quantiles_array(self, raw: Any) -> np.ndarray:
        """Bolt tensor ``(batch, n_quantiles, horizon)`` -> ``(n_quantiles, horizon)``."""
        return as_numpy(raw)[0]


class Chronos2Forecaster(ChronosForecasterBase):
    """
    Zero-shot Chronos-2 forecast (on CPU). Univariate context only; no covariates.
    """

    _default_model_path = 'amazon/chronos-2'

    def _pipeline_class(self):
        """Chronos-2 pipeline from the ``chronos`` extra."""
        chronos = self._require_extra('chronos')
        return chronos.Chronos2Pipeline

    def _pipeline_inputs(self, context: np.ndarray) -> np.ndarray:
        """Chronos-2 tensors must be (n_series, n_variates, history_length)."""
        return np.asarray(context, dtype=np.float32).reshape(1, 1, -1)

    def _quantiles_array(self, raw: Any) -> np.ndarray:
        """
        Chronos-2 returns a list of ``(n_variates, n_quantiles, horizon)`` tensors.
        One univariate series: peel the list, then drop the variate axis.
        """
        return as_numpy(raw)[0]


FORECASTERS: Dict[str, Callable[..., Forecaster]] = {
    'default': ConstFillForecaster,
    'const_fill': ConstFillForecaster,
    'daily_naive': DailyNaiveForecaster,
    'knn': KNNForecaster,
    'soft_threshold_svd': SoftThresholdSVDForecaster,
    'chronos_bolt': ChronosBoltForecaster,
    'chronos_2': Chronos2Forecaster,
}
