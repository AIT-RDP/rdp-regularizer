"""Tests for RegularizedHistoryStore."""
from datetime import datetime, timedelta, timezone

from regularizer.history.store import RegularizedHistoryStore
from regularizer.sample import Sample


def test_get_returns_none_when_missing():
    store = RegularizedHistoryStore(timedelta(hours=1))
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert store.get(ts) is None


def test_get_returns_exact_match():
    store = RegularizedHistoryStore(timedelta(hours=1))
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.extend([Sample(timestamp=ts, value=1.5, quality='forecast')])
    got = store.get(ts)
    assert got is not None
    assert got.value == 1.5
    assert got.quality == 'forecast'


def test_extend_latest_wins_including_quality_upgrade():
    store = RegularizedHistoryStore(timedelta(hours=1))
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.extend([Sample(timestamp=ts, value=1.0, quality='forecast')])
    store.extend([Sample(timestamp=ts, value=42.0, quality='measured')])
    got = store.get(ts)
    assert got is not None
    assert got.value == 42.0
    assert got.quality == 'measured'
    assert len(store.samples) == 1


def test_trim_drops_samples_older_than_window():
    window = timedelta(hours=1)
    store = RegularizedHistoryStore(window)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    old = now - window - timedelta(minutes=1)
    recent = now - timedelta(minutes=30)
    store.extend([
        Sample(timestamp=old, value=1.0, quality='measured'),
        Sample(timestamp=recent, value=2.0, quality='measured'),
    ])
    store.trim(now)
    assert store.get(old) is None
    assert store.get(recent) is not None
    assert store.get(recent).value == 2.0


def test_window_is_half_open():
    store = RegularizedHistoryStore(timedelta(hours=1))
    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    mid = start + timedelta(minutes=30)
    end = start + timedelta(hours=1)
    store.extend([
        Sample(timestamp=start, value=1.0, quality='measured'),
        Sample(timestamp=mid, value=2.0, quality='measured'),
        Sample(timestamp=end, value=3.0, quality='measured'),
    ])
    got = store.window(start, end)
    assert [s.value for s in got] == [1.0, 2.0]


def test_extend_empty_is_noop():
    store = RegularizedHistoryStore(timedelta(hours=1))
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.extend([Sample(timestamp=ts, value=1.0, quality='measured')])
    store.extend([])
    assert len(store.samples) == 1
    assert store.get(ts).value == 1.0


def test_trim_keeps_sample_exactly_on_cutoff():
    window = timedelta(hours=1)
    store = RegularizedHistoryStore(window)
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    cutoff = now - window
    store.extend([Sample(timestamp=cutoff, value=1.0, quality='measured')])
    store.trim(now)
    assert store.get(cutoff) is not None
    assert store.get(cutoff).value == 1.0
