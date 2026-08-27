"""Tests for DefaultImputer."""
from datetime import datetime, timedelta
from math import isnan

from regularizer.config import ChannelConfig
from regularizer.sample import Sample
from regularizer.tools import IMPUTERS, DefaultImputer


def test_imputer_uses_last_value_in_window(channel_config: ChannelConfig, start_time: datetime):
    missing_ts = start_time
    history = [
        Sample(timestamp=missing_ts - timedelta(minutes=2), value=1.0),
        Sample(timestamp=missing_ts - timedelta(minutes=1), value=2.0),
        Sample(timestamp=missing_ts, value=99.0),
    ]
    next_sample = Sample(timestamp=missing_ts + timedelta(minutes=1), value=50.0)

    value = DefaultImputer().impute(missing_ts, history, next_sample, channel_config)

    assert value == 2.0


def test_imputer_falls_back_to_next_sample(channel_config: ChannelConfig, start_time: datetime):
    next_sample = Sample(timestamp=start_time + timedelta(minutes=1), value=7.5)

    value = DefaultImputer().impute(start_time, [], next_sample, channel_config)

    assert value == 7.5


def test_imputer_returns_nan_without_history_or_next(
    channel_config: ChannelConfig, start_time: datetime
):
    value = DefaultImputer().impute(start_time, [], None, channel_config)
    assert isnan(value)


def test_imputer_ignores_history_outside_window(
    channel_config: ChannelConfig, start_time: datetime
):
    too_old = Sample(
        timestamp=start_time - channel_config.window - timedelta(minutes=1),
        value=1.0,
    )
    next_sample = Sample(timestamp=start_time + timedelta(minutes=1), value=4.0)

    value = DefaultImputer().impute(start_time, [too_old], next_sample, channel_config)

    assert value == 4.0


def test_imputers_registry_default():
    assert isinstance(IMPUTERS['default'](), DefaultImputer)
