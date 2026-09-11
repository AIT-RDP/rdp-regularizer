"""
Imputation of interior gaps: grid points that are missing although later measured
data has already arrived, i.e. the gap is bounded by real data on both sides.
"""
from __future__ import annotations

from typing import Callable, Dict, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample
from .util import samples_to_series, series_to_samples


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


class DefaultImputer:
    """
    Default imputation: ffill, then bfill, through the last known value.
    Trailing holes are left unfilled.
    """

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        series = samples_to_series(samples)
        last = series.last_valid_index()
        if last is not None:
            bounded = series.iloc[: last + 1]
            series.iloc[: last + 1] = bounded.ffill().bfill()
        return series_to_samples(samples, series, 'imputed', keep_nan=True)


IMPUTERS: Dict[str, Callable[[], Imputer]] = {
    'default': DefaultImputer,
}
