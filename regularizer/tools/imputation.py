"""
Imputation of interior gaps: grid points that are missing although later measured
data has already arrived, i.e. the gap is bounded by real data on both sides.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample
from ..util import parse_duration
from .algo import daily_naive_reconstruct, knn_reconstruct, linear_interpolation, soft_threshold_svd_reconstruct
from .util import (
    array_to_samples, last_valid_index, matrix_fill, nan_bfill, nan_ffill,
    samples_to_array, warn_unknown_kwargs,
)


class Imputer(Protocol):
    """
    Fills interior holes in a sample sequence (holes bounded by a later known value).
    """

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        """
        Return a sequence with the same timestamps, length, and order. Known values
        are left untouched. Interior ``None`` holes are filled with quality
        ``'imputed'``. Trailing holes (no later non-``None``) are left as ``None``.
        """
        ...


class ConstFillImputer:
    """
    Default imputation: ffill, then bfill, through the last known value.
    Trailing holes are left unfilled.
    """

    def __init__(self, **kwargs: Any):
        warn_unknown_kwargs('ConstFillImputer', kwargs)

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        values = samples_to_array(samples)
        last = last_valid_index(values)
        if last is not None:
            values[: last + 1] = nan_bfill(nan_ffill(values[: last + 1]))
        return array_to_samples(
            samples, values, 'imputed',
            keep_nan=True
        )


class LinearInterpolationImputer:
    """
    Imputation using linear interpolation.
    """

    def __init__(self, **kwargs: Any):
        warn_unknown_kwargs('LinearInterpolationImputer', kwargs)

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        values = samples_to_array(samples)
        return array_to_samples(
            samples, linear_interpolation(values),
            'imputed', keep_nan=True
        )


class DailyNaiveImputer:
    """
    Imputation using the naive period profile. ``period`` defaults to one day.
    """

    def __init__(self, period: Any = timedelta(days=1), **kwargs: Any):
        warn_unknown_kwargs('DailyNaiveImputer', kwargs)
        self.period = parse_duration(period)

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'imputed',
            daily_naive_reconstruct, interior_only=True, period=self.period
        )


class KNNImputer:
    """
    Imputation using KNN. ``period`` defaults to one day.
    """
    def __init__(self, k: int = 5, period: Any = timedelta(days=1), **kwargs: Any):
        warn_unknown_kwargs('KNNImputer', kwargs)
        self.k = k
        self.period = parse_duration(period)

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'imputed',
            lambda M: knn_reconstruct(M, k=self.k),
            interior_only=True, period=self.period
        )


class SoftThresholdSVDImputer:
    """
    Imputation using soft-thresholded SVD. ``period`` defaults to one day.
    """

    def __init__(
            self, max_rank: int = 3, shrinkage: Optional[float] = None,
            period: Any = timedelta(days=1), **kwargs: Any
        ):
        warn_unknown_kwargs('SoftThresholdSVDImputer', kwargs)
        self.max_rank = max_rank
        self.shrinkage = shrinkage
        self.period = parse_duration(period)

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'imputed',
            lambda M: soft_threshold_svd_reconstruct(
                M, max_rank=self.max_rank, shrinkage=self.shrinkage
            ), interior_only=True, period=self.period
        )


IMPUTERS: Dict[str, Callable[..., Imputer]] = {
    'default': ConstFillImputer,
    'const_fill': ConstFillImputer,
    'linear': LinearInterpolationImputer,
    'daily_naive': DailyNaiveImputer,
    'knn': KNNImputer,
    'soft_threshold_svd': SoftThresholdSVDImputer,
}
