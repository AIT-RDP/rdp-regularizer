"""Tests for ChannelConfig YAML loading."""
from datetime import timedelta

import pytest

from regularizer.config import ChannelConfig, HistoryProviderConfig


def _channel_entry(**overrides):
    entry = {
        'input_stream': 'in',
        'output_stream': 'out',
        'polling_interval': '1m',
        'update_interval': '1m',
        'window': '1h',
    }
    entry.update(overrides)
    return entry


def test_missing_history_provider_skips_bootstrap():
    configs = ChannelConfig.load_channel_configs({'ch': _channel_entry()})
    assert len(configs) == 1
    assert configs[0].history_provider is None


def test_data_provider_defaults_to_rdp_regularizer():
    configs = ChannelConfig.load_channel_configs({'ch': _channel_entry()})
    assert configs[0].data_provider_name == 'rdp-regularizer'


def test_data_provider_is_parsed():
    configs = ChannelConfig.load_channel_configs({
        'ch': _channel_entry(data_provider_name='custom-provider'),
    })
    assert configs[0].data_provider_name == 'custom-provider'


def test_required_fields_and_default_jitter_tolerance():
    config = ChannelConfig.load_channel_configs({'ch': _channel_entry()})[0]
    assert config.name == 'ch'
    assert config.input_stream == 'in'
    assert config.output_stream == 'out'
    assert config.polling_interval == timedelta(minutes=1)
    assert config.update_interval == timedelta(minutes=1)
    assert config.window == timedelta(hours=1)
    assert config.jitter_tolerance == timedelta(seconds=30)
    assert config.offset == timedelta(0)
    assert config.lag_time == timedelta(0)
    assert config.output_maxlen == 200
    assert config.imputer == 'default'
    assert config.forecaster == 'default'


def test_explicit_optional_fields_are_parsed():
    config = ChannelConfig.load_channel_configs({
        'ch': _channel_entry(
            jitter_tolerance='10s',
            offset='5s',
            lag_time='2s',
            output_maxlen=50,
            imputer='default',
            forecaster='default',
        ),
    })[0]
    assert config.jitter_tolerance == timedelta(seconds=10)
    assert config.offset == timedelta(seconds=5)
    assert config.lag_time == timedelta(seconds=2)
    assert config.output_maxlen == 50
    assert config.imputer == 'default'
    assert config.forecaster == 'default'


@pytest.mark.parametrize('channels', [{}, None])
def test_empty_channels_raises(channels):
    with pytest.raises(RuntimeError, match='channels config is empty'):
        ChannelConfig.load_channel_configs(channels)


def test_missing_required_field_raises():
    entry = _channel_entry()
    del entry['window']
    with pytest.raises(RuntimeError, match='Missing config field'):
        ChannelConfig.load_channel_configs({'ch': entry})


def test_unknown_field_raises():
    with pytest.raises(RuntimeError, match='Invalid channel config'):
        ChannelConfig.load_channel_configs({
            'ch': _channel_entry(unknown_field='nope'),
        })


def test_null_history_provider_skips_bootstrap():
    configs = ChannelConfig.load_channel_configs(
        {'ch': _channel_entry(history_provider=None)}
    )
    assert configs[0].history_provider is None


def test_history_provider_is_parsed():
    configs = ChannelConfig.load_channel_configs({
        'ch': _channel_entry(history_provider={
            'dp_name': 'temp',
            'dp_location_code': 'loc',
            'bootstrap_delay': '5s',
        }),
    })
    hp = configs[0].history_provider
    assert isinstance(hp, HistoryProviderConfig)
    assert hp.dp_name == 'temp'
    assert hp.dp_location_code == 'loc'
    assert hp.bootstrap_delay == timedelta(seconds=5)


def test_invalid_history_provider_raises():
    with pytest.raises(RuntimeError, match='Invalid history provider config'):
        ChannelConfig.load_channel_configs({
            'ch': _channel_entry(history_provider={'dp_location_code': 'loc'}),
        })
