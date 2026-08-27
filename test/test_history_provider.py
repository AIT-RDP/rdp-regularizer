"""Tests for HistoryProvider against an in-memory SQLite database."""
from datetime import datetime, timedelta, timezone

import pytest
from peewee import SqliteDatabase

from regularizer.history.provider import HistoryProvider
from regularizer.history.timescaledb import DB_PROXY, DataPoint, UnitemporalDoubleDetails


def _provider(db: SqliteDatabase) -> HistoryProvider:
    provider = object.__new__(HistoryProvider)
    provider.db_ = db
    return provider


@pytest.fixture
def history_db():
    db = SqliteDatabase(':memory:')
    saved_dp_schema = DataPoint._meta.schema
    saved_details_schema = UnitemporalDoubleDetails._meta.schema
    DataPoint._meta.schema = None
    UnitemporalDoubleDetails._meta.schema = None
    DB_PROXY.initialize(db)
    db.connect()
    db.create_tables([DataPoint, UnitemporalDoubleDetails])
    yield db
    db.drop_tables([DataPoint, UnitemporalDoubleDetails])
    db.close()
    DataPoint._meta.schema = saved_dp_schema
    UnitemporalDoubleDetails._meta.schema = saved_details_schema


def _insert_datapoint(**overrides) -> DataPoint:
    fields = dict(
        name='temp',
        location_code='loc-a',
        device_id='dev-a',
        data_provider='crawler',
    )
    fields.update(overrides)
    return DataPoint.create(**fields)


def _insert_sample(dp: DataPoint, valid_time: datetime, value: float) -> None:
    UnitemporalDoubleDetails.create(dp_id=dp.id, valid_time=valid_time, value=value)


def test_get_history_filters_by_datapoint_identity(history_db):
    wanted = _insert_datapoint()
    other = _insert_datapoint(
        name='other', location_code='loc-b', device_id='dev-b', data_provider='other'
    )
    t0 = datetime(2026, 1, 1, 12, 0)
    t1 = datetime(2026, 1, 1, 12, 1)
    _insert_sample(wanted, t0, 1.5)
    _insert_sample(other, t0, 99.0)
    _insert_sample(wanted, t1, 2.5)

    provider = _provider(history_db)
    samples = provider.get_history(
        'temp',
        start=t0,
        end=t1,
        dp_location_code='loc-a',
        dp_device_id='dev-a',
        dp_data_provider='crawler',
    )

    assert [s.value for s in samples] == [1.5, 2.5]


def test_get_history_start_and_end_are_inclusive(history_db):
    dp = _insert_datapoint()
    start = datetime(2026, 1, 1, 12, 0)
    mid = start + timedelta(minutes=1)
    end = start + timedelta(minutes=2)
    before = start - timedelta(minutes=1)
    after = end + timedelta(minutes=1)
    _insert_sample(dp, before, 0.0)
    _insert_sample(dp, start, 1.0)
    _insert_sample(dp, mid, 2.0)
    _insert_sample(dp, end, 3.0)
    _insert_sample(dp, after, 4.0)

    samples = _provider(history_db).get_history('temp', start=start, end=end)
    assert [s.value for s in samples] == [1.0, 2.0, 3.0]


def test_get_history_wraps_errors_in_value_error(history_db, monkeypatch):
    provider = _provider(history_db)

    def boom(*_args, **_kwargs):
        raise RuntimeError('db down')

    monkeypatch.setattr(provider, '_fetch_samples', boom)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match='Failed to retrieve history'):
        provider.get_history('temp', start=start, end=start)
