"""Tests for Channel step and scheduling (Redis mocked)."""
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from regularizer.channel import Channel
from regularizer.config import ChannelConfig
from regularizer.sample import Sample
from regularizer.tools import DefaultForecaster, DefaultImputer


@pytest.fixture
def channel(channel_config: ChannelConfig):
    with patch('regularizer.channel.redis.Redis') as redis_cls:
        redis_cls.return_value.xrevrange.return_value = []
        ch = Channel(
            config=channel_config,
            redis_pool=MagicMock(),
            imputer=DefaultImputer(),
            forecaster=DefaultForecaster(),
            history_provider=None,
            stop_event=threading.Event(),
        )
        ch._source = MagicMock()
        ch._regularizer = MagicMock()
        ch._sink = MagicMock()
        yield ch


def test_channel_logger_name(channel_config: ChannelConfig):
    with patch('regularizer.channel.redis.Redis') as redis_cls:
        redis_cls.return_value.xrevrange.return_value = []
        ch = Channel(
            config=channel_config,
            redis_pool=MagicMock(),
            imputer=DefaultImputer(),
            forecaster=DefaultForecaster(),
            history_provider=None,
            stop_event=threading.Event(),
        )
    assert ch._logger.name == 'rdp-regularizer.test'


def test_step_reads_adds_and_emits(channel, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=1.0, quality='measured'),
        Sample(timestamp=start_time, value=2.0, quality='measured'),
    ]
    channel._source.read.return_value = samples
    polled = [Sample(timestamp=start_time, value=1.0, quality='measured')]
    channel._regularizer.poll.return_value = polled

    channel._step()

    assert [c.args[0] for c in channel._regularizer.add.call_args_list] == samples
    channel._regularizer.poll.assert_called_once_with()
    channel._sink.emit.assert_called_once_with(polled)


def test_scheduled_step_returns_when_stopped(channel):
    channel._stop_event.set()
    channel._step = MagicMock()
    channel._scheduled_step()
    channel._step.assert_not_called()


def test_scheduled_step_swallows_error_and_reschedules(channel):
    channel._step = MagicMock(side_effect=RuntimeError('boom'))
    channel._scheduler = MagicMock()
    channel._stop_event.wait = MagicMock(return_value=False)

    with patch.object(channel, '_next_polling_timestamp', return_value=123.0):
        channel._scheduled_step()

    channel._stop_event.wait.assert_called_once_with(5)
    channel._scheduler.enterabs.assert_called_once_with(
        123.0, 1, channel._scheduled_step
    )


def test_scheduled_step_does_not_reschedule_when_stopped_after_error(channel):
    channel._step = MagicMock(side_effect=RuntimeError('boom'))
    channel._scheduler = MagicMock()

    def set_stop(_timeout):
        channel._stop_event.set()
        return True

    channel._stop_event.wait = MagicMock(side_effect=set_stop)
    channel._scheduled_step()
    channel._scheduler.enterabs.assert_not_called()


def test_next_polling_timestamp_aligns_to_epoch_plus_offset(interval):
    config = ChannelConfig(
        name='test',
        input_stream='in',
        output_stream='out',
        polling_interval=interval,
        update_interval=interval,
        jitter_tolerance=interval,
        window=interval,
        offset=timedelta(seconds=5),
    )
    with patch('regularizer.channel.redis.Redis') as redis_cls:
        redis_cls.return_value.xrevrange.return_value = []
        ch = Channel(
            config=config,
            redis_pool=MagicMock(),
            imputer=DefaultImputer(),
            forecaster=DefaultForecaster(),
            history_provider=None,
            stop_event=threading.Event(),
        )
    with patch('regularizer.channel.time.time', return_value=1000.0):
        assert ch._next_polling_timestamp() == 1025.0


def test_run_schedules_bootstrap_first_live_tick(channel):
    first_live = datetime(2030, 1, 1, 12, 2, tzinfo=timezone.utc)
    channel._regularizer.bootstrap_history.return_value = True
    channel._regularizer.first_live_ts = first_live
    channel._scheduler = MagicMock()

    with patch('regularizer.channel.time.time', return_value=first_live.timestamp() - 60):
        channel.run()

    channel._scheduler.enterabs.assert_called_once_with(
        first_live.timestamp(), 1, channel._scheduled_step
    )
    channel._scheduler.run.assert_called_once()


def test_run_reschedules_when_bootstrap_overruns_first_live_tick(channel):
    first_live = datetime(2020, 1, 1, tzinfo=timezone.utc)
    channel._regularizer.bootstrap_history.return_value = True
    channel._regularizer.first_live_ts = first_live
    channel._scheduler = MagicMock()

    with patch.object(channel, '_next_polling_timestamp', return_value=999.0):
        channel.run()

    channel._scheduler.enterabs.assert_called_once_with(
        999.0, 1, channel._scheduled_step
    )
