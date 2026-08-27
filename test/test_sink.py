"""Tests for RedisStreamSink with a mocked Redis client."""
import json
from datetime import datetime
from unittest.mock import MagicMock

from regularizer.config import ChannelConfig, HistoryProviderConfig
from regularizer.io.sink import RedisStreamSink
from regularizer.sample import Sample


def test_emit_empty_does_not_xadd(channel_config: ChannelConfig, logger):
    client = MagicMock()
    sink = RedisStreamSink(client=client, config=channel_config, logger=logger)
    sink.emit([])
    client.xadd.assert_not_called()


def test_emit_without_history_provider(
    channel_config: ChannelConfig, logger, start_time: datetime
):
    client = MagicMock()
    sink = RedisStreamSink(client=client, config=channel_config, logger=logger)
    samples = [
        Sample(timestamp=start_time, value=1.0, quality='measured'),
        Sample(timestamp=start_time, value=2.0, quality='forecast'),
    ]
    sink.emit(samples)

    stream, entry = client.xadd.call_args.args
    kwargs = client.xadd.call_args.kwargs
    assert stream == 'out'
    assert json.loads(entry['name']) == 'test'
    assert json.loads(entry['data_provider']) == 'rdp-regularizer'
    assert json.loads(entry['valid_time']) == [start_time.isoformat(), start_time.isoformat()]
    assert json.loads(entry['value']) == [1.0, 2.0]
    assert json.loads(entry['quality']) == ['measured', 'forecast']
    assert kwargs['maxlen'] == 200
    assert kwargs['approximate'] is True


def test_emit_uses_overridden_data_provider_name(
    interval, logger, start_time: datetime
):
    config = ChannelConfig(
        name='test',
        input_stream='in',
        output_stream='out-custom',
        polling_interval=interval,
        update_interval=interval,
        jitter_tolerance=interval,
        window=interval,
        output_maxlen=50,
        data_provider_name='custom-provider',
    )
    client = MagicMock()
    sink = RedisStreamSink(client=client, config=config, logger=logger)
    sink.emit([Sample(timestamp=start_time, value=1.0, quality='measured')])

    stream, entry = client.xadd.call_args.args
    assert stream == 'out-custom'
    assert json.loads(entry['data_provider']) == 'custom-provider'
    assert client.xadd.call_args.kwargs['maxlen'] == 50


def test_emit_with_history_provider_sets_optional_fields(
    interval, logger, start_time: datetime
):
    hp = HistoryProviderConfig(
        dp_name='temp',
        dp_location_code='loc',
        dp_unit='C',
        dp_device_id='dev-1',
    )
    config = ChannelConfig(
        name='ignored',
        input_stream='in',
        output_stream='out',
        polling_interval=interval,
        update_interval=interval,
        jitter_tolerance=interval,
        window=interval,
        history_provider=hp,
    )
    client = MagicMock()
    sink = RedisStreamSink(client=client, config=config, logger=logger)
    sink.emit([Sample(timestamp=start_time, value=1.0, quality='measured')])

    _, entry = client.xadd.call_args.args
    assert json.loads(entry['name']) == 'temp'
    assert json.loads(entry['location_code']) == 'loc'
    assert json.loads(entry['unit']) == 'C'
    assert json.loads(entry['device_id']) == 'dev-1'
    assert json.loads(entry['data_provider']) == 'rdp-regularizer'


def test_emit_with_history_provider_omits_unset_fields(
    interval, logger, start_time: datetime
):
    hp = HistoryProviderConfig(dp_name='temp')
    config = ChannelConfig(
        name='ignored',
        input_stream='in',
        output_stream='out',
        polling_interval=interval,
        update_interval=interval,
        jitter_tolerance=interval,
        window=interval,
        history_provider=hp,
    )
    client = MagicMock()
    sink = RedisStreamSink(client=client, config=config, logger=logger)
    sink.emit([Sample(timestamp=start_time, value=1.0, quality='measured')])

    _, entry = client.xadd.call_args.args
    assert json.loads(entry['name']) == 'temp'
    assert 'location_code' not in entry
    assert 'unit' not in entry
    assert 'device_id' not in entry
