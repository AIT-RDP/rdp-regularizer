from __future__ import annotations

from datetime import timedelta
from math import ceil, nan
from typing import Callable, List, Sequence

import numpy as np

from ..sample import Quality, Sample


def samples_to_array(samples: Sequence[Sample]) -> np.ndarray:
    """
    Convert a sample sequence to a float array. ``None`` values become NaN.
    """
    return np.array(
        [nan if sample.value is None else sample.value for sample in samples],
        dtype=np.float64,
    )


def array_to_samples(
        samples: Sequence[Sample],
        filled: np.ndarray,
        quality: Quality,
        *,
        keep_nan: bool = False,
    ) -> List[Sample]:
    """
    Write filled array values back onto the original sample sequence.

    Known values are left untouched. Holes that received a finite fill become
    new samples with ``quality``. Remaining NaNs stay as the original hole when
    ``keep_nan`` is true, otherwise they become ``math.nan`` with ``quality``.
    """
    values = np.asarray(filled, dtype=np.float64).reshape(-1)
    result: List[Sample] = []
    for sample, fill in zip(samples, values, strict=True):
        if sample.value is not None:
            result.append(sample)
            continue
        if np.isnan(fill):
            if keep_nan:
                result.append(sample)
            else:
                result.append(Sample(timestamp=sample.timestamp, value=nan, quality=quality))
            continue
        result.append(Sample(timestamp=sample.timestamp, value=float(fill), quality=quality))
    return result


def last_valid_index(values: np.ndarray) -> int | None:
    """Index of the last finite value, or ``None`` if the array is all-NaN."""
    valid = np.flatnonzero(~np.isnan(values))
    return int(valid[-1]) if valid.size else None


def nan_ffill(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Forward-fill NaNs along ``axis``. Leading NaNs are left as NaN."""
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.size == 0:
        return arr
    if axis != 0:
        arr = np.moveaxis(arr, axis, 0)
    n = arr.shape[0]
    flat = arr.reshape(n, -1)
    idx = np.where(~np.isnan(flat), np.arange(n)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = flat[idx, np.arange(flat.shape[1])]
    out = out.reshape(arr.shape)
    if axis != 0:
        out = np.moveaxis(out, 0, axis)
    return out


def nan_bfill(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Backward-fill NaNs along ``axis``. Trailing NaNs are left as NaN."""
    arr = np.array(values, dtype=np.float64, copy=True)
    if arr.size == 0:
        return arr
    return np.flip(nan_ffill(np.flip(arr, axis=axis), axis=axis), axis=axis)


def slots_per_day(interval: timedelta) -> int:
    """
    Number of grid slots in a 24-hour day for ``interval``.

    Raises ``ValueError`` if ``interval`` is not positive or does not divide a
    day evenly.
    """
    day = timedelta(days=1)
    if interval <= timedelta(0) or day % interval:
        raise ValueError(f'update interval {interval} does not divide a day evenly')
    return int(day / interval)


def matrix_fill(
        samples: Sequence[Sample],
        interval: timedelta,
        quality: Quality,
        reconstruct: Callable[[np.ndarray], np.ndarray],
        interior_only: bool = False,
    ) -> List[Sample]:
    """
    Shared pipeline for matrix filling: normalize -> reconstruct(matrix) -> invert + clip

    With ``interior_only`` the pipeline stops at the last known value and trailing
    holes are returned untouched, so that a forecaster can still claim them.
    """
    trailing: List[Sample] = []
    if interior_only:
        last = last_valid_index(samples_to_array(samples))
        if last is None:
            return list(samples)
        samples, trailing = samples[: last + 1], list(samples[last + 1:])

    n_slots = slots_per_day(interval)
    n = len(samples)
    if n == 0:
        return []

    # Pad the samples with NaNs to the nearest multiple of n_slots
    n_days = ceil(n / n_slots)
    padded = np.full(n_days * n_slots, np.nan, dtype=np.float64)
    padded[:n] = samples_to_array(samples)

    # Create a mask of the NaNs in the padded array
    mask = np.isnan(padded)

    # Check if the data is positive-only
    lo = np.nanmin(padded)
    positive = bool(lo >= 0)

    # Positive-only -> impute in log space (avoids negative-spike overshoot)
    src = np.log1p(padded) if positive else padded

    # Reshape the source array into a matrix of shape (n_days, n_slots)
    M = src.reshape(n_days, n_slots)

    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    # Guard constant columns
    sd = np.where(sd == 0, 1.0, sd)
    # Normalize each time-of-day column
    Mn = (M - mu) / sd

    # Apply the reconstruction function to the normalized matrix
    M_filled = reconstruct(Mn) * sd + mu

    # Flatten the filled matrix.
    filled = np.asarray(M_filled, dtype=np.float64).reshape(-1)

    # Invert log1p if the data is positive-only
    if positive:
        filled = np.expm1(filled)

    # Replace the NaNs in the padded array with the filled values and clip the spikes
    out = padded.copy()
    out[mask] = filled[mask]
    out = np.clip(out, 0.0 if positive else lo, None)

    return array_to_samples(
        samples, out[:n], quality
    ) + trailing
