"""Tests for parse_duration and Redis connection-pool loading."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from regularizer.util import (
    load_redis_connection_pool,
    next_polling_datetime,
    next_polling_timestamp,
    parse_duration,
)


@pytest.mark.parametrize(
    'value, expected',
    [
        ('30s', timedelta(seconds=30)),
        ('1m', timedelta(minutes=1)),
        ('12h', timedelta(hours=12)),
        ('1d', timedelta(days=1)),
        ('1w', timedelta(weeks=1)),
        ('100ms', timedelta(milliseconds=100)),
        (' 30s ', timedelta(seconds=30)),
        (30, timedelta(seconds=30)),
        (30.5, timedelta(seconds=30.5)),
    ],
)
def test_parse_duration_converts_strings_and_numbers(value, expected):
    assert parse_duration(value) == expected


def test_parse_duration_passthrough_timedelta():
    value = timedelta(minutes=15)
    assert parse_duration(value) is value


@pytest.mark.parametrize('value', ['30x', 's', None, {}, '1 y'])
def test_parse_duration_rejects_invalid(value):
    with pytest.raises(ValueError, match='Invalid duration'):
        parse_duration(value)


def test_next_polling_timestamp_aligns_to_epoch_plus_offset():
    assert next_polling_timestamp(1000.0, timedelta(minutes=1), timedelta(seconds=5)) == 1025.0


def test_next_polling_datetime_is_utc_wrapper_of_timestamp():
    now = datetime.fromtimestamp(1000.0, tz=timezone.utc)
    got = next_polling_datetime(now, timedelta(minutes=1), timedelta(seconds=5))
    assert got == datetime.fromtimestamp(1025.0, tz=timezone.utc)


@patch('regularizer.util.redis.Redis')
@patch('regularizer.util.redis.ConnectionPool')
def test_load_redis_connection_pool_without_password(mock_pool_cls, mock_redis_cls):
    pool = MagicMock()
    mock_pool_cls.return_value = pool
    client = MagicMock()
    mock_redis_cls.return_value = client

    result = load_redis_connection_pool({'host': 'localhost', 'port': 6379, 'db': 0})

    mock_pool_cls.assert_called_once_with(
        host='localhost', port=6379, db=0, decode_responses=True
    )
    mock_redis_cls.assert_called_once_with(connection_pool=pool)
    client.ping.assert_called_once()
    assert result is pool


@patch('regularizer.util.redis.Redis')
@patch('regularizer.util.redis.ConnectionPool')
def test_load_redis_connection_pool_with_password(mock_pool_cls, mock_redis_cls):
    pool = MagicMock()
    mock_pool_cls.return_value = pool
    mock_redis_cls.return_value = MagicMock()

    result = load_redis_connection_pool({
        'host': 'localhost',
        'port': 6379,
        'db': 1,
        'password': 'secret',
    })

    mock_pool_cls.assert_called_once_with(
        host='localhost', port=6379, db=1, decode_responses=True, password='secret'
    )
    assert result is pool
