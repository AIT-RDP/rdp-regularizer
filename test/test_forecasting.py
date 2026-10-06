"""Tests for DefaultForecaster."""
import threading
import time
from datetime import datetime, timedelta
from math import isnan
from typing import List

import numpy as np
import pytest

from regularizer.config import ChannelConfig
from regularizer.sample import Sample
from regularizer.tools import FORECASTERS, ConstFillForecaster, create_forecaster
from regularizer.tools.forecasting import (
    Chronos2Forecaster,
    ChronosBoltForecaster,
    ChronosForecasterBase,
    DailyNaiveForecaster,
    KNNForecaster,
    SoftThresholdSVDForecaster,
)

SLOTS_PER_DAY = 96


def daily_profile(index: int) -> float:
    """Value of a profile that repeats identically every day."""
    return 1.0 + 0.5 * ((index % SLOTS_PER_DAY) / SLOTS_PER_DAY)


def profile_samples(
    start: datetime, interval: timedelta, days: int = 3, holes: int = 4
) -> List[Sample]:
    """``days`` of the repeating profile followed by ``holes`` trailing gaps."""
    known = SLOTS_PER_DAY * days
    return [
        Sample(timestamp=start + index * interval, value=daily_profile(index))
        for index in range(known)
    ] + [
        Sample(timestamp=start + (known + offset) * interval, value=None)
        for offset in range(holes)
    ]


def test_forecaster_uses_last_value(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time - timedelta(minutes=2), value=1.0),
        Sample(timestamp=start_time - timedelta(minutes=1), value=8.0),
        Sample(timestamp=start_time, value=None),
    ]

    filled = ConstFillForecaster().forecast(samples, channel_config)

    assert [s.timestamp for s in filled] == [s.timestamp for s in samples]
    assert filled[0].value == 1.0
    assert filled[1].value == 8.0
    assert filled[2].value == 8.0
    assert filled[2].quality == 'forecast'


def test_forecaster_returns_nan_without_history(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [Sample(timestamp=start_time, value=None)]

    filled = ConstFillForecaster().forecast(samples, channel_config)

    assert filled[0].quality == 'forecast'
    assert isnan(filled[0].value)


def test_forecaster_leaves_known_values_untouched(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time, value=4.0, quality='measured'),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
    ]

    filled = ConstFillForecaster().forecast(samples, channel_config)

    assert filled[0].value == 4.0
    assert filled[0].quality == 'measured'
    assert filled[1].value == 4.0
    assert filled[1].quality == 'forecast'


def test_forecaster_same_locf_across_run(
    channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time, value=11.0),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
        Sample(timestamp=start_time + timedelta(minutes=2), value=None),
    ]

    filled = ConstFillForecaster().forecast(samples, channel_config)

    assert filled[1].value == 11.0
    assert filled[2].value == 11.0
    assert filled[1].quality == 'forecast'
    assert filled[2].quality == 'forecast'


def test_forecasters_registry_default():
    assert isinstance(FORECASTERS['default'](), ConstFillForecaster)


@pytest.mark.parametrize(
    'forecaster',
    [DailyNaiveForecaster(), KNNForecaster(), SoftThresholdSVDForecaster()],
)
def test_matrix_forecaster_leaves_known_values_untouched(
    forecaster, matrix_channel_config: ChannelConfig, matrix_interval: timedelta,
    start_time: datetime
):
    samples = profile_samples(start_time, matrix_interval)
    known = len(samples) - 4

    filled = forecaster.forecast(samples, matrix_channel_config)

    for original, sample in zip(samples[:known], filled[:known]):
        assert sample.value == original.value
        assert sample.quality == 'measured'


@pytest.mark.parametrize(
    'name,expected',
    [
        ('daily_naive', DailyNaiveForecaster),
        ('knn', KNNForecaster),
        ('soft_threshold_svd', SoftThresholdSVDForecaster),
    ],
)
def test_forecasters_registry_matrix_entries(name: str, expected: type):
    assert isinstance(FORECASTERS[name](), expected)


@pytest.mark.parametrize(
    'method,forecaster',
    [
        ('soft', SoftThresholdSVDForecaster()),
        ('knn', KNNForecaster()),
        ('seasonal', DailyNaiveForecaster()),
    ],
)
def test_matrix_forecaster_matches_reference(
    evaluate_data: dict, evaluate_samples, matrix_channel_config: ChannelConfig,
    method: str, forecaster
):
    # Blank the trailing horizon of the reference series
    n_forecast = evaluate_data['n_forecast']
    total = len(evaluate_data['full_data'])
    samples = evaluate_samples(range(total - n_forecast, total))

    # Forecast the blanked horizon
    filled = forecaster.forecast(samples, matrix_channel_config)

    # Assert the horizon matches the reference values (stored with 3 decimals)
    expected = evaluate_data[method]['forecast']
    assert len(expected) == n_forecast
    for sample in filled[-n_forecast:]:
        assert sample.quality == 'forecast'
        assert round(sample.value, 3) == expected[sample.timestamp.isoformat()]


CHRONOS_CLASSES = [ChronosBoltForecaster, Chronos2Forecaster]


def _mock_predict(forecaster, values):
    def fake_predict(context, horizon):
        assert horizon == len(values)
        return np.asarray(values, dtype=np.float64)
    forecaster._predict = fake_predict


@pytest.mark.parametrize(
    'name,expected',
    [
        ('chronos_bolt', ChronosBoltForecaster),
        ('chronos_2', Chronos2Forecaster),
    ],
)
def test_forecasters_registry_chronos_entries(name: str, expected: type):
    assert isinstance(FORECASTERS[name](), expected)


def test_create_forecaster_passes_kwargs():
    forecaster = create_forecaster(
        'chronos_bolt', {'model_path': 'amazon/chronos-bolt-small', 'cache_dir': '/cache'}
    )
    assert isinstance(forecaster, ChronosBoltForecaster)
    assert forecaster.model_path == 'amazon/chronos-bolt-small'
    assert forecaster.cache_dir == '/cache'
    knn = create_forecaster('knn', {'k': 7})
    assert isinstance(knn, KNNForecaster)
    assert knn.k == 7


def test_create_forecaster_unknown_name_raises():
    with pytest.raises(RuntimeError, match='Unknown forecaster'):
        create_forecaster('nope')


def test_create_forecaster_unknown_kwargs_warns(monkeypatch):
    logged = []
    monkeypatch.setattr('regularizer.tools.chronos.LOGGER.warning', logged.append)
    forecaster = create_forecaster('chronos_bolt', {'bogus': 1})
    assert isinstance(forecaster, ChronosBoltForecaster)
    assert logged == ["ChronosForecasterBase unknown kwargs: {'bogus': 1}"]


@pytest.mark.parametrize('forecaster_cls', CHRONOS_CLASSES)
def test_chronos_forecaster_fills_trailing_holes(
    forecaster_cls, channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time, value=4.0, quality='measured'),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
        Sample(timestamp=start_time + timedelta(minutes=2), value=None),
    ]
    forecaster = forecaster_cls()
    _mock_predict(forecaster, [9.0, 8.0])

    filled = forecaster.forecast(samples, channel_config)

    assert filled[0].value == 4.0
    assert filled[0].quality == 'measured'
    assert filled[1].value == 9.0
    assert filled[1].quality == 'forecast'
    assert filled[2].value == 8.0
    assert filled[2].quality == 'forecast'
    assert [s.timestamp for s in filled] == [s.timestamp for s in samples]


@pytest.mark.parametrize('forecaster_cls', CHRONOS_CLASSES)
def test_chronos_forecaster_returns_nan_without_history(
    forecaster_cls, channel_config: ChannelConfig, start_time: datetime
):
    samples = [Sample(timestamp=start_time, value=None)]
    forecaster = forecaster_cls()
    called = []

    def fake_predict(context, horizon):
        called.append(horizon)
        return np.zeros(horizon)

    forecaster._predict = fake_predict

    filled = forecaster.forecast(samples, channel_config)

    assert called == []
    assert filled[0].quality == 'forecast'
    assert isnan(filled[0].value)


@pytest.mark.parametrize('forecaster_cls', CHRONOS_CLASSES)
def test_chronos_forecaster_fills_interior_from_context(
    forecaster_cls, channel_config: ChannelConfig, start_time: datetime
):
    samples = [
        Sample(timestamp=start_time, value=1.0, quality='measured'),
        Sample(timestamp=start_time + timedelta(minutes=1), value=None),
        Sample(timestamp=start_time + timedelta(minutes=2), value=3.0, quality='measured'),
        Sample(timestamp=start_time + timedelta(minutes=3), value=None),
        Sample(timestamp=start_time + timedelta(minutes=4), value=None),
    ]
    forecaster = forecaster_cls()
    _mock_predict(forecaster, [9.0, 8.0])

    filled = forecaster.forecast(samples, channel_config)

    assert filled[0].value == 1.0
    assert filled[0].quality == 'measured'
    assert filled[1].value == 1.0
    assert filled[1].quality == 'forecast'
    assert filled[2].value == 3.0
    assert filled[2].quality == 'measured'
    assert filled[3].value == 9.0
    assert filled[3].quality == 'forecast'
    assert filled[4].value == 8.0
    assert filled[4].quality == 'forecast'


def test_chronos_bolt_pipeline_inputs_is_1d():
    context = np.arange(8, dtype=np.float64)
    assert ChronosBoltForecaster()._pipeline_inputs(context).shape == (8,)


def test_chronos2_pipeline_inputs_is_3d():
    context = np.arange(8, dtype=np.float64)
    assert Chronos2Forecaster()._pipeline_inputs(context).shape == (1, 1, 8)


def test_median_point_forecast_from_bolt_quantiles():
    raw = np.zeros((1, 3, 4), dtype=np.float64)
    raw[0, 1, :] = [9.0, 8.0, 7.0, 6.0]
    np.testing.assert_array_equal(
        ChronosBoltForecaster()._median_point_forecast(raw, [0.1, 0.5, 0.9], 4),
        [9.0, 8.0, 7.0, 6.0],
    )


def test_median_point_forecast_from_chronos2_quantiles():
    raw = [np.zeros((1, 3, 4), dtype=np.float64)]
    raw[0][0, 1, :] = [9.0, 8.0, 7.0, 6.0]
    np.testing.assert_array_equal(
        Chronos2Forecaster()._median_point_forecast(raw, [0.1, 0.5, 0.9], 4),
        [9.0, 8.0, 7.0, 6.0],
    )


def test_resolve_hub_cache_dir_explicit_wins(monkeypatch):
    monkeypatch.setenv('HF_HUB_CACHE', '/from-env')
    assert ChronosForecasterBase._get_weights_cache_dir('/explicit') == '/explicit'
    assert ChronosBoltForecaster(cache_dir='/explicit').cache_dir == '/explicit'
    assert Chronos2Forecaster(cache_dir='/explicit').cache_dir == '/explicit'


def test_resolve_hub_cache_dir_from_env(monkeypatch):
    monkeypatch.setenv('HF_HUB_CACHE', '/from-env')
    assert ChronosForecasterBase._get_weights_cache_dir() == '/from-env'
    assert ChronosBoltForecaster().cache_dir == '/from-env'
    assert Chronos2Forecaster().cache_dir == '/from-env'


def test_resolve_hub_cache_dir_default(monkeypatch):
    monkeypatch.delenv('HF_HUB_CACHE', raising=False)
    assert ChronosForecasterBase._get_weights_cache_dir() is None
    assert ChronosBoltForecaster().cache_dir is None
    assert Chronos2Forecaster().cache_dir is None


@pytest.mark.parametrize(
    'forecaster_cls,kwargs,expected',
    [
        (
            ChronosBoltForecaster,
            {'model_path': 'amazon/chronos-bolt-small', 'cache_dir': '/cache'},
            'ChronosBoltForecaster loading model=amazon/chronos-bolt-small cache_dir=/cache',
        ),
        (
            Chronos2Forecaster,
            {},
            'Chronos2Forecaster loading model=amazon/chronos-2 cache_dir=Hugging Face default',
        ),
    ],
)
def test_ensure_pipeline_logs_model_and_cache(
    monkeypatch, forecaster_cls, kwargs, expected
):
    class Pipeline:
        quantiles = [0.5]
        predict = object()

        @classmethod
        def from_pretrained(cls, model_path, **from_kwargs):
            return cls()

    logged = []
    monkeypatch.setattr('regularizer.tools.chronos.LOGGER.debug', logged.append)
    monkeypatch.delenv('HF_HUB_CACHE', raising=False)

    class Torch:
        float32 = 'float32'

    def require(target):
        if target == 'torch':
            return Torch
        if callable(target):
            return target()
        raise AssertionError(target)

    forecaster = forecaster_cls(**kwargs)
    monkeypatch.setattr(forecaster, '_pipeline_class', lambda: Pipeline)
    monkeypatch.setattr(forecaster, '_require_extra', require)
    forecaster._ensure_pipeline()
    forecaster._ensure_pipeline()
    assert logged == [expected]


def test_ensure_pipeline_serializes_concurrent_loads(monkeypatch):
    """Two instances may load, but not at the same time, and not twice."""
    state = {'in_flight': 0, 'max_in_flight': 0, 'calls': 0}
    state_lock = threading.Lock()

    class Pipeline:
        quantiles = [0.5]
        predict = object()

        @classmethod
        def from_pretrained(cls, model_path, **from_kwargs):
            with state_lock:
                state['calls'] += 1
                state['in_flight'] += 1
                state['max_in_flight'] = max(state['max_in_flight'], state['in_flight'])
            time.sleep(0.05)
            with state_lock:
                state['in_flight'] -= 1
            return cls()

    class Torch:
        float32 = 'float32'

    def require(target):
        if target == 'torch':
            return Torch
        if callable(target):
            return target()
        raise AssertionError(target)

    monkeypatch.delenv('HF_HUB_CACHE', raising=False)
    first = ChronosBoltForecaster(model_path='amazon/chronos-bolt-tiny', cache_dir='/cache')
    second = ChronosBoltForecaster(model_path='amazon/chronos-bolt-tiny', cache_dir='/cache')
    for forecaster in (first, second):
        monkeypatch.setattr(forecaster, '_pipeline_class', lambda: Pipeline)
        monkeypatch.setattr(forecaster, '_require_extra', require)

    start = threading.Barrier(2)

    def load(forecaster):
        start.wait()
        forecaster._ensure_pipeline()

    threads = [threading.Thread(target=load, args=(forecaster,)) for forecaster in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state['max_in_flight'] == 1
    assert state['calls'] == 2
    second._ensure_pipeline()
    assert state['calls'] == 2


def test_predict_serializes_concurrent_calls(monkeypatch):
    """Shared instance: pipeline.predict must not run on two threads at once."""
    state = {'in_flight': 0, 'max_in_flight': 0}
    state_lock = threading.Lock()

    class Pipeline:
        quantiles = [0.5]

        def predict(self, inputs, prediction_length):
            with state_lock:
                state['in_flight'] += 1
                state['max_in_flight'] = max(state['max_in_flight'], state['in_flight'])
            time.sleep(0.05)
            with state_lock:
                state['in_flight'] -= 1
            return np.zeros((1, 1, prediction_length), dtype=np.float64)

        @classmethod
        def from_pretrained(cls, model_path, **from_kwargs):
            return cls()

    class Torch:
        float32 = 'float32'

        @staticmethod
        def as_tensor(values):
            return values

    def require(target):
        if target == 'torch':
            return Torch
        if callable(target):
            return target()
        raise AssertionError(target)

    monkeypatch.delenv('HF_HUB_CACHE', raising=False)
    forecaster = ChronosBoltForecaster(cache_dir='/cache')
    monkeypatch.setattr(forecaster, '_pipeline_class', lambda: Pipeline)
    monkeypatch.setattr(forecaster, '_require_extra', require)
    context = np.arange(8, dtype=np.float64)
    start = threading.Barrier(2)

    def run():
        start.wait()
        forecaster._predict(context, 4)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state['max_in_flight'] == 1


def test_chronos2_pipeline_inputs_univariate_is_3d():
    payload = Chronos2Forecaster()._pipeline_inputs(np.arange(8, dtype=np.float64))
    assert isinstance(payload, np.ndarray)
    assert payload.shape == (1, 1, 8)
