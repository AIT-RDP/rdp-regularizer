"""
Forecasting of grid points whose deadline (grid_time + lag_time) has passed without
any measured sample arriving: the channel must emit anyway to stay strictly regular,
so the value is extrapolated from history only.
"""
from __future__ import annotations

from typing import Callable, Dict, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample
from .util import samples_to_series, series_to_samples


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


class DefaultForecaster:
    """
    Naive forecast: ffill last observation, NaN when none.
    """

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        series = samples_to_series(samples).ffill()
        return series_to_samples(samples, series, 'forecast')


FORECASTERS: Dict[str, Callable[[], Forecaster]] = {
    'default': DefaultForecaster,
}
