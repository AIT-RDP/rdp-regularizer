"""
Forecasting of grid points whose deadline (grid_time + lag_time) has passed without
any measured sample arriving: the channel must emit anyway to stay strictly regular,
so the value is extrapolated from history only.

Real forecasting methods are available separately; DummyForecaster stands in for them.
"""
from __future__ import annotations

from math import nan
from typing import Callable, Dict, List, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample


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
    Naive forecast: last observation carried forward, NaN when none.
    """

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        last: float | None = None
        result: List[Sample] = []
        for sample in samples:
            if sample.value is not None:
                last = sample.value
                result.append(sample)
                continue
            fill = last if last is not None else nan
            result.append(Sample(timestamp=sample.timestamp, value=fill, quality='forecast'))
        return result


FORECASTERS: Dict[str, Callable[[], Forecaster]] = {
    'default': DefaultForecaster,
}
