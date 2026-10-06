"""Tests for DefaultImputer."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from regularizer.config import ChannelConfig
from regularizer.sample import Sample
from regularizer.tools import IMPUTERS, ConstFillImputer, create_imputer
from regularizer.tools.imputation import DailyNaiveImputer, KNNImputer, LinearInterpolationImputer, SoftThresholdSVDImputer
from regularizer.tools.util import samples_to_array


def test_imputer_uses_last_value(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time - timedelta(minutes=2), value=1.0),
        Sample(timestamp=start_time - timedelta(minutes=1), value=2.0),
        Sample(timestamp=start_time, value=None),
        Sample(timestamp=start_time + timedelta(minutes=1), value=50.0),
    ]

    filled = ConstFillImputer().impute(samples, channel_config)

    assert [s.timestamp for s in filled] == [s.timestamp for s in samples]
    assert filled[0].value == 1.0
    assert filled[1].value == 2.0
    assert filled[2].value == 2.0
    assert filled[2].quality == 'imputed'
    assert filled[3].value == 50.0
    assert filled[3].quality == 'measured'


def test_imputer_falls_back_to_next_sample(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=None),
        Sample(timestamp=start_time + timedelta(minutes=1), value=7.5),
    ]

    filled = ConstFillImputer().impute(samples, channel_config)

    assert filled[0].value == 7.5
    assert filled[0].quality == 'imputed'
    assert filled[1].value == 7.5


def test_imputer_leaves_trailing_holes(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=None),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
    ]

    filled = ConstFillImputer().impute(samples, channel_config)

    assert filled[0].value is None
    assert filled[1].value is None


def test_imputer_fills_only_interior_holes(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=3.0),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
        Sample(timestamp=start_time + timedelta(minutes=2), value=9.0),
        Sample(timestamp=start_time + timedelta(minutes=3), value=None),
    ]

    filled = ConstFillImputer().impute(samples, channel_config)

    assert filled[1].value == 3.0
    assert filled[1].quality == 'imputed'
    assert filled[2].value == 9.0
    assert filled[2].quality == 'measured'
    assert filled[3].value is None


def test_imputer_same_locf_across_run(channel_config: ChannelConfig, start_time: datetime):
    samples = [
        Sample(timestamp=start_time, value=5.0),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
        Sample(timestamp=start_time + timedelta(minutes=2), value=None),
        Sample(timestamp=start_time + timedelta(minutes=3), value=20.0),
    ]

    filled = ConstFillImputer().impute(samples, channel_config)

    assert filled[1].value == 5.0
    assert filled[2].value == 5.0
    assert filled[1].quality == 'imputed'
    assert filled[2].quality == 'imputed'


def test_imputers_registry_default():
    assert isinstance(IMPUTERS['default'](), ConstFillImputer)


def test_create_imputer_passes_kwargs():
    imputer = create_imputer('knn', {'k': 7})
    assert isinstance(imputer, KNNImputer)
    assert imputer.k == 7


def test_create_imputer_unknown_name_raises():
    with pytest.raises(RuntimeError, match='Unknown imputer'):
        create_imputer('nope')


def test_create_imputer_invalid_kwargs_raises():
    with pytest.raises(RuntimeError, match='Invalid imputer config'):
        create_imputer('knn', {'bogus': 1})


@pytest.mark.parametrize(
    'method,imputer',
    [
        ('soft', SoftThresholdSVDImputer()),
        ('knn', KNNImputer()),
        ('seasonal', DailyNaiveImputer()),
        ('linear', LinearInterpolationImputer()),
    ],
)
def test_matrix_imputer_mae_rmse(
    evaluate_data: dict, evaluate_samples, matrix_channel_config: ChannelConfig,
    method: str, imputer
):
    # Retrieve test samples with missing values
    test_samples = evaluate_samples(evaluate_data['mask'])
    # Impute the missing values
    filled = imputer.impute(test_samples, matrix_channel_config)
    # Retrieve the predicted values as an array
    pred = samples_to_array(filled)[evaluate_data['mask']]
    # Retrieve the true values as an array
    true = np.array(list(evaluate_data['full_data'].values()))[evaluate_data['mask']]
    # Calculate the MAE
    mae = np.mean(np.abs(pred - true))
    # Calculate the RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    # Assert the MAE and RMSE are close to the expected values
    assert mae == pytest.approx(evaluate_data[method]['mae'])
    assert rmse == pytest.approx(evaluate_data[method]['rmse'])


@pytest.mark.parametrize(
    'imputer',
    [SoftThresholdSVDImputer(), KNNImputer(), DailyNaiveImputer()],
)
def test_matrix_imputer_leaves_trailing_holes(
    imputer, matrix_channel_config: ChannelConfig, matrix_interval: timedelta,
    start_time: datetime
):
    # Two days of data, one interior hole, then trailing holes for the forecaster.
    known = 2 * 96
    samples = [
        Sample(timestamp=start_time + index * matrix_interval, value=float(index % 96))
        for index in range(known)
    ]
    samples[100] = Sample(timestamp=samples[100].timestamp, value=None)
    samples += [
        Sample(timestamp=start_time + (known + offset) * matrix_interval, value=None)
        for offset in range(3)
    ]

    filled = imputer.impute(samples, matrix_channel_config)

    assert [s.timestamp for s in filled] == [s.timestamp for s in samples]
    assert filled[100].value is not None
    assert filled[100].quality == 'imputed'
    for sample in filled[known:]:
        assert sample.value is None
