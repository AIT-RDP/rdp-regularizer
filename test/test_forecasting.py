"""Tests for DefaultForecaster."""
from datetime import datetime, timedelta
from math import isnan
from typing import List

import pytest

from regularizer.config import ChannelConfig
from regularizer.sample import Sample
from regularizer.tools import FORECASTERS, ConstFillForecaster
from regularizer.tools.forecasting import (
    DailyNaiveForecaster,
    KNNForecaster,
    SoftThresholdSVDForecaster,
)

SLOTS_PER_DAY = 96


def daily_profile(index: int) -> float:
    """Value of a profile that repeats identically every day."""
    return 1.0 + 0.5 * ((index % SLOTS_PER_DAY) / SLOTS_PER_DAY)


def profile_samples(
    start: datetime, interval: timedelta, days: int = 3, holes: int = 4
) -> List[Sample]:
    """``days`` of the repeating profile followed by ``holes`` trailing gaps."""
    known = SLOTS_PER_DAY * days
    return [
        Sample(timestamp=start + index * interval, value=daily_profile(index))
        for index in range(known)
    ] + [
        Sample(timestamp=start + (known + offset) * interval, value=None)
        for offset in range(holes)
    ]


def test_forecaster_uses_last_value(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time - timedelta(minutes=2), value=1.0),
        Sample(timestamp=start_time - timedelta(minutes=1), value=8.0),
        Sample(timestamp=start_time, value=None),
    ]

    filled = ConstFillForecaster().forecast(samples, channel_config)

    assert [s.timestamp for s in filled] == [s.timestamp for s in samples]
    assert filled[0].value == 1.0
    assert filled[1].value == 8.0
    assert filled[2].value == 8.0
    assert filled[2].quality == 'forecast'


def test_forecaster_returns_nan_without_history(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [Sample(timestamp=start_time, value=None)]

    filled = ConstFillForecaster().forecast(samples, channel_config)

    assert filled[0].quality == 'forecast'
    assert isnan(filled[0].value)


def test_forecaster_leaves_known_values_untouched(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time, value=4.0, quality='measured'),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
    ]

    filled = ConstFillForecaster().forecast(samples, channel_config)

    assert filled[0].value == 4.0
    assert filled[0].quality == 'measured'
    assert filled[1].value == 4.0
    assert filled[1].quality == 'forecast'


def test_forecaster_same_locf_across_run(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time, value=11.0),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
        Sample(timestamp=start_time + timedelta(minutes=2), value=None),
    ]

    filled = ConstFillForecaster().forecast(samples, channel_config)

    assert filled[1].value == 11.0
    assert filled[2].value == 11.0
    assert filled[1].quality == 'forecast'
    assert filled[2].quality == 'forecast'


def test_forecasters_registry_default():
    assert isinstance(FORECASTERS['default'](), ConstFillForecaster)


@pytest.mark.parametrize(
    'forecaster',
    [DailyNaiveForecaster(), KNNForecaster(), SoftThresholdSVDForecaster()],
)
def test_matrix_forecaster_leaves_known_values_untouched(
    forecaster, matrix_channel_config: ChannelConfig, matrix_interval: timedelta,
    start_time: datetime
):
    samples = profile_samples(start_time, matrix_interval)
    known = len(samples) - 4

    filled = forecaster.forecast(samples, matrix_channel_config)

    for original, sample in zip(samples[:known], filled[:known]):
        assert sample.value == original.value
        assert sample.quality == 'measured'


@pytest.mark.parametrize(
    'name,expected',
    [
        ('daily_naive', DailyNaiveForecaster),
        ('knn', KNNForecaster),
        ('soft_threshold_svd', SoftThresholdSVDForecaster),
    ],
)
def test_forecasters_registry_matrix_entries(name: str, expected: type):
    assert isinstance(FORECASTERS[name](), expected)


@pytest.mark.parametrize(
    'method,forecaster',
    [
        ('soft', SoftThresholdSVDForecaster()),
        ('knn', KNNForecaster()),
        ('seasonal', DailyNaiveForecaster()),
    ],
)
def test_matrix_forecaster_matches_reference(
    evaluate_data: dict, evaluate_samples, matrix_channel_config: ChannelConfig,
    method: str, forecaster
):
    # Blank the trailing horizon of the reference series
    n_forecast = evaluate_data['n_forecast']
    total = len(evaluate_data['full_data'])
    samples = evaluate_samples(range(total - n_forecast, total))

    # Forecast the blanked horizon
    filled = forecaster.forecast(samples, matrix_channel_config)

    # Assert the horizon matches the reference values (stored with 3 decimals)
    expected = evaluate_data[method]['forecast']
    assert len(expected) == n_forecast
    for sample in filled[-n_forecast:]:
        assert sample.quality == 'forecast'
        assert round(sample.value, 3) == expected[sample.timestamp.isoformat()]
