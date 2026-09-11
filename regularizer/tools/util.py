from __future__ import annotations

from math import nan
from typing import List, Sequence

import pandas as pd

from ..sample import Quality, Sample


def samples_to_series(samples: Sequence[Sample]) -> pd.Series:
    """
    Convert a sample sequence to a float series. ``None`` values become NaN.
    """
    return pd.Series(
        [nan if sample.value is None else sample.value for sample in samples],
        dtype='float64',
    )


def series_to_samples(
        samples: Sequence[Sample],
        filled: pd.Series,
        quality: Quality,
        *,
        keep_nan: bool = False,
    ) -> List[Sample]:
    """
    Write filled series values back onto the original sample sequence.

    Known values are left untouched. Holes that received a finite fill become
    new samples with ``quality``. Remaining NaNs stay as the original hole when
    ``keep_nan`` is true, otherwise they become ``math.nan`` with ``quality``.
    """
    result: List[Sample] = []
    for sample, fill in zip(samples, filled, strict=True):
        if sample.value is not None:
            result.append(sample)
            continue
        if pd.isna(fill):
            if keep_nan:
                result.append(sample)
            else:
                result.append(Sample(timestamp=sample.timestamp, value=nan, quality=quality))
            continue
        result.append(Sample(timestamp=sample.timestamp, value=float(fill), quality=quality))
    return result
