"""Tests for RedisStreamSource with a mocked Redis client."""
from __future__ import annotations
import json
import logging
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import redis

from regularizer.io.source import RedisStreamSource
from regularizer.sample import Sample


def _logger() -> logging.Logger:
    return logging.getLogger('rdp-regularizer.test')


def _source(client: MagicMock | None = None) -> RedisStreamSource:
    if client is None:
        client = MagicMock()
        client.xrevrange.return_value = []
    return RedisStreamSource(client=client, stream='in', logger=_logger())


def test_empty_stream_tip_and_read():
    client = MagicMock()
    client.xrevrange.return_value = []
    client.xread.return_value = []
    source = RedisStreamSource(client=client, stream='in', logger=_logger())
    assert source._last_id == '0-0'
    assert source.read() == []


def test_stream_tip_id_from_newest_entry():
    client = MagicMock()
    client.xrevrange.return_value = [('42-0', {'_time': '[]'})]
    source = RedisStreamSource(client=client, stream='in', logger=_logger())
    assert source._last_id == '42-0'


def test_stream_tip_id_on_response_error():
    client = MagicMock()
    client.xrevrange.side_effect = redis.ResponseError('no stream')
    source = RedisStreamSource(client=client, stream='in', logger=_logger())
    assert source._last_id == '0-0'


def test_parse_entry_list_and_z_suffix():
    source = _source()
    samples, metadata = source._parse_entry({
        '_time': json.dumps([
            '2026-01-01T12:00:00+00:00',
            '2026-01-01T12:01:00Z',
        ]),
        '_value': json.dumps([1.0, 2.0]),
        '_metadata': json.dumps({'src': 'crawler'}),
    })
    assert metadata == {'src': 'crawler'}
    assert len(samples) == 2
    assert samples[0] == Sample(
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        value=1.0,
        quality='measured',
    )
    assert samples[1].timestamp == datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)
    assert samples[1].value == 2.0


def test_parse_entry_scalar_naive_timestamp():
    source = _source()
    samples, _ = source._parse_entry({
        '_time': json.dumps('2026-01-01T12:00:00'),
        '_value': json.dumps(3.5),
    })
    assert len(samples) == 1
    assert samples[0].timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert samples[0].value == 3.5


def test_parse_entry_length_mismatch_raises():
    source = _source()
    with pytest.raises(ValueError, match='length mismatch'):
        source._parse_entry({
            '_time': json.dumps(['2026-01-01T12:00:00+00:00']),
            '_value': json.dumps([1.0, 2.0]),
        })


def test_malformed_entry_skipped_and_cursor_advances():
    client = MagicMock()
    client.xrevrange.return_value = []
    client.xread.side_effect = [
        [('in', [('1-0', {'_time': 'not-json', '_value': '[1]'})])],
        [],
    ]
    source = RedisStreamSource(client=client, stream='in', logger=_logger())
    assert source.read() == []
    assert source._last_id == '1-0'


def test_wait_until_available_existing_entry():
    client = MagicMock()
    client.xrevrange.return_value = [('1-0', {})]
    source = RedisStreamSource(client=client, stream='in', logger=_logger())
    assert source.wait_until_available(threading.Event()) is True


def test_wait_until_available_stop_event():
    client = MagicMock()
    client.xrevrange.return_value = []
    client.xread.return_value = []
    source = RedisStreamSource(client=client, stream='in', logger=_logger())
    stop = threading.Event()
    stop.set()
    assert source.wait_until_available(stop) is False


def test_wait_until_available_response_error():
    client = MagicMock()
    client.xrevrange.side_effect = redis.ResponseError('err')
    source = RedisStreamSource(client=client, stream='in', logger=_logger())
    assert source.wait_until_available(threading.Event()) is False
