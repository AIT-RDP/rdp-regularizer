"""Tests for DefaultForecaster."""
from datetime import datetime, timedelta
from math import isnan

from regularizer.config import ChannelConfig
from regularizer.sample import Sample
from regularizer.tools import FORECASTERS, DefaultForecaster


def test_forecaster_uses_last_value(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time - timedelta(minutes=2), value=1.0),
        Sample(timestamp=start_time - timedelta(minutes=1), value=8.0),
        Sample(timestamp=start_time, value=None),
    ]

    filled = DefaultForecaster().forecast(samples, channel_config)

    assert [s.timestamp for s in filled] == [s.timestamp for s in samples]
    assert filled[0].value == 1.0
    assert filled[1].value == 8.0
    assert filled[2].value == 8.0
    assert filled[2].quality == 'forecast'


def test_forecaster_returns_nan_without_history(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [Sample(timestamp=start_time, value=None)]

    filled = DefaultForecaster().forecast(samples, channel_config)

    assert filled[0].quality == 'forecast'
    assert isnan(filled[0].value)


def test_forecaster_leaves_known_values_untouched(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time, value=4.0, quality='measured'),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
    ]

    filled = DefaultForecaster().forecast(samples, channel_config)

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

    filled = DefaultForecaster().forecast(samples, channel_config)

    assert filled[1].value == 11.0
    assert filled[2].value == 11.0
    assert filled[1].quality == 'forecast'
    assert filled[2].quality == 'forecast'


def test_forecasters_registry_default():
    assert isinstance(FORECASTERS['default'](), DefaultForecaster)
