"""Shared fixtures for RDP regularizer unit tests."""
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, List

import pytest

from regularizer.config import ChannelConfig
from regularizer.regularizer import TimeGridRegularizer
from regularizer.sample import Sample
from regularizer.tools import ConstFillForecaster, ConstFillImputer

EVALUATE_PATH = Path(__file__).parent / 'evaluate.json'


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
def matrix_interval() -> timedelta:
    return timedelta(minutes=15)


@pytest.fixture
def matrix_channel_config(matrix_interval: timedelta) -> ChannelConfig:
    """
    Config for the matrix-based tools: a 15-minute grid gives 96 slots per day,
    so a multi-day window yields enough rows to reconstruct from.
    """
    return ChannelConfig(
        name='test',
        input_stream='in',
        output_stream='out',
        polling_interval=matrix_interval,
        update_interval=matrix_interval,
        jitter_tolerance=timedelta(seconds=30),
        window=timedelta(days=3),
        lag_time=timedelta(seconds=0),
    )


@pytest.fixture(scope='session')
def evaluate_data() -> dict:
    """Reference series, imputation mask, and per-method expectations."""
    with EVALUATE_PATH.open(encoding='utf-8') as handle:
        return json.load(handle)


@pytest.fixture
def evaluate_samples(evaluate_data: dict) -> Callable[[Iterable[int]], List[Sample]]:
    """Build the evaluation series with a hole at every index in ``holes``."""
    def build(holes: Iterable[int]) -> List[Sample]:
        blanks = set(holes)
        return [
            Sample(
                timestamp=datetime.fromisoformat(timestamp),
                value=None if index in blanks else float(value),
            )
            for index, (timestamp, value) in enumerate(evaluate_data['full_data'].items())
        ]
    return build


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
        imputer=ConstFillImputer(),
        forecaster=ConstFillForecaster(),
        history_provider=None,
        logger=logger,
        start_time=start_time,
    )
    reg._next_grid_ts = start_time
    return reg
