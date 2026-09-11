"""Tests for DefaultImputer."""
from datetime import datetime, timedelta

from regularizer.config import ChannelConfig
from regularizer.sample import Sample
from regularizer.tools import IMPUTERS, DefaultImputer


def test_imputer_uses_last_value(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time - timedelta(minutes=2), value=1.0),
        Sample(timestamp=start_time - timedelta(minutes=1), value=2.0),
        Sample(timestamp=start_time, value=None),
        Sample(timestamp=start_time + timedelta(minutes=1), value=50.0),
    ]

    filled = DefaultImputer().impute(samples, channel_config)

    assert [s.timestamp for s in filled] == [s.timestamp for s in samples]
    assert filled[0].value == 1.0
    assert filled[1].value == 2.0
    assert filled[2].value == 2.0
    assert filled[2].quality == 'imputed'
    assert filled[3].value == 50.0
    assert filled[3].quality == 'measured'


def test_imputer_falls_back_to_next_sample(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=None),
        Sample(timestamp=start_time + timedelta(minutes=1), value=7.5),
    ]

    filled = DefaultImputer().impute(samples, channel_config)

    assert filled[0].value == 7.5
    assert filled[0].quality == 'imputed'
    assert filled[1].value == 7.5


def test_imputer_leaves_trailing_holes(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=None),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
    ]

    filled = DefaultImputer().impute(samples, channel_config)

    assert filled[0].value is None
    assert filled[1].value is None


def test_imputer_fills_only_interior_holes(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=3.0),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
        Sample(timestamp=start_time + timedelta(minutes=2), value=9.0),
        Sample(timestamp=start_time + timedelta(minutes=3), value=None),
    ]

    filled = DefaultImputer().impute(samples, channel_config)

    assert filled[1].value == 3.0
    assert filled[1].quality == 'imputed'
    assert filled[2].value == 9.0
    assert filled[2].quality == 'measured'
    assert filled[3].value is None


def test_imputer_same_locf_across_run(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=5.0),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
        Sample(timestamp=start_time + timedelta(minutes=2), value=None),
        Sample(timestamp=start_time + timedelta(minutes=3), value=20.0),
    ]

    filled = DefaultImputer().impute(samples, channel_config)

    assert filled[1].value == 5.0
    assert filled[2].value == 5.0
    assert filled[1].quality == 'imputed'
    assert filled[2].quality == 'imputed'


def test_imputers_registry_default():
    assert isinstance(IMPUTERS['default'](), DefaultImputer)
