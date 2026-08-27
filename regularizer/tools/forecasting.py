"""
Forecasting of grid points whose deadline (grid_time + lag_time) has passed without
any measured sample arriving: the channel must emit anyway to stay strictly regular,
so the value is extrapolated from history only.

Real forecasting methods are available separately; DummyForecaster stands in for them.
"""
from __future__ import annotations

from math import nan
from typing import Callable, Dict, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample


class Forecaster(Protocol):
    """
    Extrapolates the value of a grid point from history only.
    """

    def forecast(self, target_ts, history: Sequence[Sample],
                 config: ChannelConfig) -> float:
        """
        Forecast the value of a grid point from history only.
        """
        ...


class DefaultForecaster:
    """
    Naive forecast: repeat the last value from the fetched history, NaN when none.
    """

    def forecast(self, target_ts, history: Sequence[Sample],
                 config: ChannelConfig) -> float:
        start = target_ts - config.window
        end = target_ts
        window = [s for s in history if start <= s.timestamp < end]

        if window:
            return window[-1].value
        return nan


FORECASTERS: Dict[str, Callable[[], Forecaster]] = {
    'default': DefaultForecaster,
}
