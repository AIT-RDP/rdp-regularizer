"""Tests for TimeGridRegularizer, including late measured corrections."""
import threading
from datetime import datetime, timedelta
from math import isnan
from unittest.mock import MagicMock, patch

from regularizer.config import ChannelConfig, HistoryProviderConfig
from regularizer.regularizer import TimeGridRegularizer
from regularizer.sample import Sample
from regularizer.tools import DefaultForecaster, DefaultImputer


def _make_regularizer(logger, start_time, interval, provider=None, **config_overrides):
    kwargs = dict(
        name='test',
        input_stream='in',
        output_stream='out',
        polling_interval=interval,
        update_interval=interval,
        jitter_tolerance=timedelta(seconds=30),
        window=timedelta(hours=1),
        lag_time=timedelta(seconds=0),
    )
    kwargs.update(config_overrides)
    config = ChannelConfig(**kwargs)
    reg = TimeGridRegularizer(
        config=config,
        imputer=DefaultImputer(),
        forecaster=DefaultForecaster(),
        history_provider=provider,
        logger=logger,
        start_time=start_time,
    )
    reg._next_grid_ts = start_time
    return reg


def test_measured_on_open_grid(regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta):
    regularizer.add(Sample(timestamp=start_time, value=10.0, quality='measured'))
    emitted = regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)
    assert len(emitted) == 1
    assert emitted[0].timestamp == start_time
    assert emitted[0].value == 10.0
    assert emitted[0].quality == 'measured'


def test_deadline_emits_forecast(regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta):
    emitted = regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)
    assert len(emitted) == 1
    assert emitted[0].timestamp == start_time
    assert emitted[0].quality == 'forecast'
    assert regularizer._next_grid_ts == start_time + interval


def test_interior_gap_imputed(regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta):
    # Seed history so imputer has a prior value, then leave a gap before a later measured.
    regularizer.add(Sample(timestamp=start_time, value=5.0, quality='measured'))
    regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)

    later = start_time + 2 * interval
    regularizer.add(Sample(timestamp=later, value=20.0, quality='measured'))
    # Poll far enough to finalize gap + measured (horizon = next + interval covers one step;
    # use a wider interval so both gap and later point finalize).
    emitted = regularizer.poll(now=later + timedelta(seconds=1), interval=2 * interval)

    by_ts = {s.timestamp: s for s in emitted}
    gap_ts = start_time + interval
    assert gap_ts in by_ts
    assert by_ts[gap_ts].quality == 'imputed'
    assert later in by_ts
    assert by_ts[later].quality == 'measured'
    assert by_ts[later].value == 20.0


def test_jitter_beyond_tolerance_dropped(
    logger, start_time: datetime, interval: timedelta
):
    # With default tolerance (0.5 * interval) every timestamp is on-grid enough;
    # use a tighter tolerance so an off-grid sample is dropped.
    from regularizer.config import ChannelConfig
    from regularizer.tools import DefaultForecaster, DefaultImputer

    config = ChannelConfig(
        name='test-jitter',
        input_stream='in',
        output_stream='out',
        polling_interval=interval,
        update_interval=interval,
        jitter_tolerance=timedelta(seconds=10),
        window=timedelta(hours=1),
        lag_time=timedelta(seconds=0),
    )
    reg = TimeGridRegularizer(
        config=config,
        imputer=DefaultImputer(),
        forecaster=DefaultForecaster(),
        history_provider=None,
        logger=logger,
        start_time=start_time,
    )
    reg._next_grid_ts = start_time
    off = start_time + timedelta(seconds=20)
    reg.add(Sample(timestamp=off, value=1.0, quality='measured'))
    assert reg._pending == {}


def test_closest_wins(regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta):
    farther = start_time + timedelta(seconds=20)
    nearer = start_time + timedelta(seconds=5)
    regularizer.add(Sample(timestamp=farther, value=100.0, quality='measured'))
    regularizer.add(Sample(timestamp=nearer, value=7.0, quality='measured'))
    emitted = regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)
    assert len(emitted) == 1
    assert emitted[0].value == 7.0
    assert emitted[0].quality == 'measured'


def test_late_upgrades_forecast(
    regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta
):
    emitted = regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)
    assert emitted[0].quality == 'forecast'
    frontier = regularizer._next_grid_ts

    regularizer.add(Sample(timestamp=start_time, value=42.0, quality='measured'))
    assert len(regularizer._corrections) == 1
    prior = regularizer._history.get(start_time)
    assert prior is not None
    assert prior.quality == 'measured'
    assert prior.value == 42.0

    emitted2 = regularizer.poll(now=start_time + interval + timedelta(seconds=1), interval=interval)
    assert emitted2[0].timestamp == start_time
    assert emitted2[0].quality == 'measured'
    assert emitted2[0].value == 42.0
    assert regularizer._next_grid_ts == frontier + interval
    assert regularizer._corrections == []


def test_late_upgrades_imputed(
    regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta
):
    regularizer.add(Sample(timestamp=start_time, value=5.0, quality='measured'))
    regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)

    later = start_time + 2 * interval
    gap_ts = start_time + interval
    regularizer.add(Sample(timestamp=later, value=20.0, quality='measured'))
    emitted = regularizer.poll(now=later + timedelta(seconds=1), interval=2 * interval)
    assert any(s.timestamp == gap_ts and s.quality == 'imputed' for s in emitted)

    frontier = regularizer._next_grid_ts
    regularizer.add(Sample(timestamp=gap_ts, value=15.0, quality='measured'))
    assert len(regularizer._corrections) == 1
    assert regularizer._history.get(gap_ts).quality == 'measured'
    assert regularizer._history.get(gap_ts).value == 15.0

    emitted2 = regularizer.poll(now=frontier + timedelta(seconds=1), interval=interval)
    assert emitted2[0].timestamp == gap_ts
    assert emitted2[0].quality == 'measured'
    assert emitted2[0].value == 15.0
    assert regularizer._next_grid_ts >= frontier


def test_late_when_already_measured_dropped(
    regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta
):
    regularizer.add(Sample(timestamp=start_time, value=10.0, quality='measured'))
    regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)

    regularizer.add(Sample(timestamp=start_time, value=99.0, quality='measured'))
    assert regularizer._corrections == []
    assert regularizer._history.get(start_time).value == 10.0


def test_late_unknown_timestamp_dropped(
    regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta
):
    regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)
    unknown = start_time - interval
    regularizer.add(Sample(timestamp=unknown, value=1.0, quality='measured'))
    assert regularizer._corrections == []
    assert regularizer._history.get(unknown) is None


def test_poll_before_deadline_emits_nothing(logger, start_time: datetime, interval: timedelta):
    reg = _make_regularizer(logger, start_time, interval, lag_time=timedelta(seconds=30))
    emitted = reg.poll(now=start_time + timedelta(seconds=1), interval=interval)
    assert emitted == []
    assert reg._next_grid_ts == start_time


def test_lag_time_delays_forecast(logger, start_time: datetime, interval: timedelta):
    lag = timedelta(seconds=30)
    reg = _make_regularizer(logger, start_time, interval, lag_time=lag)
    before = reg.poll(now=start_time + timedelta(seconds=1), interval=interval)
    assert before == []
    after = reg.poll(now=start_time + lag, interval=interval)
    assert len(after) == 1
    assert after[0].quality == 'forecast'
    assert after[0].timestamp == start_time


def test_first_forecast_is_nan(
    regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta
):
    emitted = regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)
    assert emitted[0].quality == 'forecast'
    assert isnan(emitted[0].value)


def test_forecast_and_impute_use_last_history_value(
    regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta
):
    regularizer.add(Sample(timestamp=start_time, value=11.0, quality='measured'))
    regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)

    forecasted = regularizer.poll(
        now=start_time + interval + timedelta(seconds=1), interval=interval
    )
    assert len(forecasted) == 1
    assert forecasted[0].quality == 'forecast'
    assert forecasted[0].value == 11.0

    later = start_time + 3 * interval
    gap_ts = start_time + 2 * interval
    regularizer.add(Sample(timestamp=later, value=20.0, quality='measured'))
    emitted = regularizer.poll(now=later + timedelta(seconds=1), interval=2 * interval)
    by_ts = {s.timestamp: s for s in emitted}
    assert by_ts[gap_ts].quality == 'imputed'
    assert by_ts[gap_ts].value == 11.0


def test_jitter_equal_to_tolerance_is_kept(logger, start_time: datetime, interval: timedelta):
    reg = _make_regularizer(
        logger, start_time, interval, jitter_tolerance=timedelta(seconds=10)
    )
    on_edge = start_time + timedelta(seconds=10)
    reg.add(Sample(timestamp=on_edge, value=1.0, quality='measured'))
    assert start_time in reg._pending
    assert reg._pending[start_time][0].value == 1.0


def test_closer_pending_not_replaced_by_farther(
    regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta
):
    nearer = start_time + timedelta(seconds=5)
    farther = start_time + timedelta(seconds=20)
    regularizer.add(Sample(timestamp=nearer, value=7.0, quality='measured'))
    regularizer.add(Sample(timestamp=farther, value=100.0, quality='measured'))
    emitted = regularizer.poll(now=start_time + timedelta(seconds=1), interval=interval)
    assert len(emitted) == 1
    assert emitted[0].value == 7.0


def test_poll_finalizes_several_grid_points(
    regularizer: TimeGridRegularizer, start_time: datetime, interval: timedelta
):
    regularizer.add(Sample(timestamp=start_time, value=1.0, quality='measured'))
    regularizer.add(Sample(timestamp=start_time + interval, value=2.0, quality='measured'))
    regularizer.add(Sample(timestamp=start_time + 2 * interval, value=3.0, quality='measured'))
    emitted = regularizer.poll(now=start_time + timedelta(seconds=1), interval=3 * interval)
    assert [s.value for s in emitted] == [1.0, 2.0, 3.0]
    assert [s.quality for s in emitted] == ['measured', 'measured', 'measured']


def test_bootstrap_skipped_without_provider(
    regularizer: TimeGridRegularizer,
):
    sink = MagicMock()
    source = MagicMock()
    assert regularizer.bootstrap_history(sink, source, threading.Event()) is True
    sink.emit.assert_not_called()
    source.wait_until_available.assert_not_called()


def test_bootstrap_replays_history_and_emits(
    logger, start_time: datetime, interval: timedelta
):
    provider = MagicMock()
    bootstrap_start = start_time - timedelta(hours=1)
    provider.get_history.return_value = [
        Sample(timestamp=bootstrap_start, value=3.0, quality='measured'),
    ]
    hp = HistoryProviderConfig(dp_name='temp')
    reg = _make_regularizer(
        logger, start_time, interval, provider=provider, history_provider=hp
    )
    sink = MagicMock()
    source = MagicMock()

    assert reg.bootstrap_history(sink, source, threading.Event()) is True
    provider.get_history.assert_called_once()
    assert provider.get_history.call_args.args[0] == 'temp'
    sink.emit.assert_called_once()
    emitted = sink.emit.call_args.args[0]
    assert emitted[0].timestamp == bootstrap_start
    assert emitted[0].value == 3.0
    assert emitted[0].quality == 'measured'


def test_bootstrap_aborts_when_source_unavailable(
    logger, start_time: datetime, interval: timedelta
):
    provider = MagicMock()
    hp = HistoryProviderConfig(dp_name='temp', init_when_source_available=True)
    reg = _make_regularizer(
        logger, start_time, interval, provider=provider, history_provider=hp
    )
    sink = MagicMock()
    source = MagicMock()
    source.wait_until_available.return_value = False

    assert reg.bootstrap_history(sink, source, threading.Event()) is False
    provider.get_history.assert_not_called()
    sink.emit.assert_not_called()


def test_bootstrap_waits_then_continues_when_source_available(
    logger, start_time: datetime, interval: timedelta
):
    provider = MagicMock()
    provider.get_history.return_value = []
    hp = HistoryProviderConfig(
        dp_name='temp',
        init_when_source_available=True,
        bootstrap_delay=timedelta(seconds=5),
    )
    reg = _make_regularizer(
        logger, start_time, interval, provider=provider, history_provider=hp
    )
    sink = MagicMock()
    source = MagicMock()
    source.wait_until_available.return_value = True

    with patch('regularizer.regularizer.time.sleep') as sleep:
        assert reg.bootstrap_history(sink, source, threading.Event()) is True
        sleep.assert_called_once_with(5.0)
    provider.get_history.assert_called_once()
