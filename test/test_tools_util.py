"""Tests for sample array conversion helpers."""
from datetime import timedelta

import numpy as np
import pytest

from regularizer.tools.util import nan_bfill, nan_ffill, slots_per_day


def test_slots_per_day_from_interval():
    assert slots_per_day(timedelta(hours=1)) == 24
    assert slots_per_day(timedelta(minutes=1)) == 1440


def test_slots_per_day_rejects_interval_that_does_not_divide_a_day():
    with pytest.raises(ValueError, match='does not divide a day evenly'):
        slots_per_day(timedelta(minutes=7))


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
