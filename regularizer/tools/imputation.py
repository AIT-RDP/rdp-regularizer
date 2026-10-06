"""
Imputation of interior gaps: grid points that are missing although later measured
data has already arrived, i.e. the gap is bounded by real data on both sides.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample
from .algo import daily_naive_reconstruct, knn_reconstruct, linear_interpolation, soft_threshold_svd_reconstruct
from .util import array_to_samples, last_valid_index, matrix_fill, nan_bfill, nan_ffill, samples_to_array


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

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        values = samples_to_array(samples)
        return array_to_samples(
            samples, linear_interpolation(values),
            'imputed', keep_nan=True
        )


class DailyNaiveImputer:
    """
    Imputation using daily naive approach.
    """
    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'imputed',
            daily_naive_reconstruct, interior_only=True
        )


class KNNImputer:
    """
    Imputation using KNN.
    """
    def __init__(self, k: int = 5):
        self.k = k

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'imputed',
            lambda M: knn_reconstruct(M, k=self.k), interior_only=True
        )


class SoftThresholdSVDImputer:
    """
    Imputation using soft-thresholded SVD.
    """

    def __init__(self, max_rank: int = 3, shrinkage: Optional[float] = None):
        self.max_rank = max_rank
        self.shrinkage = shrinkage

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        return matrix_fill(
            samples, config.update_interval, 'imputed',
            lambda M: soft_threshold_svd_reconstruct(
                M, max_rank=self.max_rank, shrinkage=self.shrinkage
            ), interior_only=True
        )


IMPUTERS: Dict[str, Callable[..., Imputer]] = {
    'default': ConstFillImputer,
    'const_fill': ConstFillImputer,
    'linear': LinearInterpolationImputer,
    'daily_naive': DailyNaiveImputer,
    'knn': KNNImputer,
    'soft_threshold_svd': SoftThresholdSVDImputer,
}
