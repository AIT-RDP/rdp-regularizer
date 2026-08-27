"""Tests for DefaultForecaster."""
from datetime import datetime, timedelta
from math import isnan

from regularizer.config import ChannelConfig
from regularizer.sample import Sample
from regularizer.tools import FORECASTERS, DefaultForecaster


def test_forecaster_uses_last_value_in_window(
    channel_config: ChannelConfig, start_time: datetime
):
    target_ts = start_time
    history = [
        Sample(timestamp=target_ts - timedelta(minutes=2), value=1.0),
        Sample(timestamp=target_ts - timedelta(minutes=1), value=8.0),
        Sample(timestamp=target_ts, value=99.0),
    ]

    value = DefaultForecaster().forecast(target_ts, history, channel_config)

    assert value == 8.0


def test_forecaster_returns_nan_without_history(
    channel_config: ChannelConfig, start_time: datetime
):
    value = DefaultForecaster().forecast(start_time, [], channel_config)
    assert isnan(value)


def test_forecaster_ignores_history_outside_window(
    channel_config: ChannelConfig, start_time: datetime
):
    too_old = Sample(
        timestamp=start_time - channel_config.window - timedelta(minutes=1),
        value=1.0,
    )

    value = DefaultForecaster().forecast(start_time, [too_old], channel_config)

    assert isnan(value)


def test_forecasters_registry_default():
    assert isinstance(FORECASTERS['default'](), DefaultForecaster)
