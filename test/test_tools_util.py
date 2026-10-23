"""Tests for sample array conversion helpers."""
import warnings
from datetime import datetime, timedelta, timezone
from math import isnan

import numpy as np
import pytest

from regularizer.sample import Sample
from regularizer.tools.algo import daily_naive_reconstruct
from regularizer.tools.util import matrix_fill, nan_bfill, nan_ffill, slots_per_period


def test_slots_per_period_from_interval():
    assert slots_per_period(timedelta(hours=1), timedelta(days=1)) == 24
    assert slots_per_period(timedelta(minutes=1), timedelta(days=1)) == 1440
    assert slots_per_period(timedelta(minutes=15), timedelta(hours=12)) == 48


@pytest.mark.parametrize(
    'interval,period',
    [
        (timedelta(minutes=7), timedelta(days=1)),
        (timedelta(0), timedelta(days=1)),
        (timedelta(hours=1), timedelta(0)),
    ],
)
def test_slots_per_period_rejects_invalid_period(interval: timedelta, period: timedelta):
    with pytest.raises(ValueError, match='does not divide period'):
        slots_per_period(interval, period)


def test_matrix_fill_uses_period():
    """A block shorter than a day stays NaN until ``period`` matches that block."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    interval = timedelta(hours=1)
    # A negative value keeps the fill in linear space, so row copies stay exact.
    values = [1.0, 2.0, None, 4.0, None, -1.0]
    samples = [
        Sample(timestamp=start + index * interval, value=value)
        for index, value in enumerate(values)
    ]

    # Padding out to a day leaves columns with no observations.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        day_filled = matrix_fill(samples, interval, 'imputed', daily_naive_reconstruct)
    assert [sample.value for sample in day_filled[:2]] == [1.0, 2.0]
    assert isnan(day_filled[2].value)
    assert day_filled[3].value == 4.0
    assert isnan(day_filled[4].value)
    assert day_filled[5].value == -1.0

    filled = matrix_fill(
        samples, interval, 'imputed', daily_naive_reconstruct,
        period=timedelta(hours=3),
    )
    assert [sample.value for sample in filled] == [1.0, 2.0, -1.0, 4.0, 2.0, -1.0]
    assert filled[0].quality == 'measured'
    assert filled[2].quality == 'imputed'
    assert filled[4].quality == 'imputed'


def test_nan_ffill_then_bfill_1d():
    values = np.array([np.nan, 2.0, np.nan, 4.0, np.nan])
    filled = nan_bfill(nan_ffill(values))
    np.testing.assert_array_equal(filled, [2.0, 2.0, 2.0, 4.0, 4.0])


def test_nan_ffill_bfill_2d_axis0():
    values = np.array([
        [1.0, np.nan],
        [np.nan, 2.0],
        [np.nan, np.nan],
    ])
    filled = nan_bfill(nan_ffill(values, axis=0), axis=0)
    np.testing.assert_array_equal(filled, [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
