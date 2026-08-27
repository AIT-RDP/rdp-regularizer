"""
Imputation of interior gaps: grid points that are missing although later measured
data has already arrived, i.e. the gap is bounded by real data on both sides.

Real imputation methods are available separately; DummyImputer stands in for them.
"""
from __future__ import annotations

from math import nan
from typing import Callable, Dict, Optional, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample


class Imputer(Protocol):
    """
    Fills a single missing grid point that is bounded by measured data.
    """

    def impute(self, missing_ts, history: Sequence[Sample],
               next_sample: Sample, config: ChannelConfig) -> float:
        """
        Returns the value for the missing grid point. `next_sample` is the measured
        sample after the gap; earlier values come from the pre-fetched `history`.
        """
        ...


class DefaultImputer:
    """
    Default imputation: last observation carried forward from the fetched
    history, falling back to the bounding next sample, then NaN.
    """

    def impute(self, missing_ts, history: Sequence[Sample],
               next_sample: Optional[Sample], config: ChannelConfig) -> float:
        start = missing_ts - config.window
        end = missing_ts
        window = [s for s in history if start <= s.timestamp < end]
        if window:
            return window[-1].value
        if next_sample is not None:
            return next_sample.value
        return nan


IMPUTERS: Dict[str, Callable[[], Imputer]] = {
    'default': DefaultImputer,
}
