"""
Forecasting of grid points whose deadline (grid_time + lag_time) has passed without
any measured sample arriving: the channel must emit anyway to stay strictly regular,
so the value is extrapolated from history only.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample
from .algo import daily_naive_reconstruct, knn_reconstruct, soft_threshold_svd_reconstruct
from .util import array_to_samples, matrix_fill, nan_ffill, samples_to_array


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

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        values = nan_ffill(samples_to_array(samples))
        return array_to_samples(samples, values, 'forecast')


class DailyNaiveForecaster:
    """
    Forecast using daily naive approach.
    """
    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'forecast',
            daily_naive_reconstruct
        )


class KNNForecaster:
    """
    Forecast using KNN.
    """
    def __init__(self, k: int = 5):
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

    def __init__(self, max_rank: int = 3, shrinkage: Optional[float] = None):
        self.max_rank = max_rank
        self.shrinkage = shrinkage

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'forecast',
            lambda M: soft_threshold_svd_reconstruct(
                M, max_rank=self.max_rank, shrinkage=self.shrinkage
            )
        )


FORECASTERS: Dict[str, Callable[[], Forecaster]] = {
    'default': ConstFillForecaster,
    'const_fill': ConstFillForecaster,
    'daily_naive': DailyNaiveForecaster,
    'knn': KNNForecaster,
    'soft_threshold_svd': SoftThresholdSVDForecaster,
}
