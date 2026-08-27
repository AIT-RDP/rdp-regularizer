"""Shared fixtures for RDP regularizer unit tests."""
import logging
from datetime import datetime, timedelta, timezone

import pytest

from regularizer.config import ChannelConfig
from regularizer.regularizer import TimeGridRegularizer
from regularizer.tools import DefaultForecaster, DefaultImputer


@pytest.fixture
def interval() -> timedelta:
    return timedelta(minutes=1)


@pytest.fixture
def start_time() -> datetime:
    return datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def channel_config(interval: timedelta) -> ChannelConfig:
    return ChannelConfig(
        name='test',
        input_stream='in',
        output_stream='out',
        polling_interval=interval,
        update_interval=interval,
        jitter_tolerance=timedelta(seconds=30),
        window=timedelta(hours=1),
        lag_time=timedelta(seconds=0),
    )


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger('rdp-regularizer.test')


@pytest.fixture
def regularizer(
    channel_config: ChannelConfig,
    logger: logging.Logger,
    start_time: datetime,
) -> TimeGridRegularizer:
    reg = TimeGridRegularizer(
        config=channel_config,
        imputer=DefaultImputer(),
        forecaster=DefaultForecaster(),
        history_provider=None,
        logger=logger,
        start_time=start_time,
    )
    reg._next_grid_ts = start_time
    return reg
