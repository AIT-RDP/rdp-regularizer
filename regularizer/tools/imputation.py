"""
Imputation of interior gaps: grid points that are missing although later measured
data has already arrived, i.e. the gap is bounded by real data on both sides.

Real imputation methods are available separately; DummyImputer stands in for them.
"""
from __future__ import annotations

from math import nan
from typing import Callable, Dict, List, Protocol, Sequence

from ..config import ChannelConfig
from ..sample import Sample


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
    Default imputation: last observation carried forward, falling back to the next
    known value, then NaN. Trailing holes are left unfilled.
    """

    def impute(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        n = len(samples)
        has_right_bound = [False] * n
        next_value: List[float | None] = [None] * n
        seen_value = False
        nv: float | None = None
        for i in range(n - 1, -1, -1):
            has_right_bound[i] = seen_value
            next_value[i] = nv
            if samples[i].value is not None:
                seen_value = True
                nv = samples[i].value

        last: float | None = None
        result: List[Sample] = []
        for i, sample in enumerate(samples):
            if sample.value is not None:
                last = sample.value
                result.append(sample)
                continue
            if not has_right_bound[i]:
                result.append(sample)
                continue
            if last is not None:
                fill = last
            elif next_value[i] is not None:
                fill = next_value[i]
            else:
                fill = nan
            result.append(Sample(timestamp=sample.timestamp, value=fill, quality='imputed'))
        return result


IMPUTERS: Dict[str, Callable[[], Imputer]] = {
    'default': DefaultImputer,
}
