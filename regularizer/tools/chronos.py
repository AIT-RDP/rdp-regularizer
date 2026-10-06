from __future__ import annotations
from typing import Any, Callable, Optional, Protocol, Sequence

import abc
import importlib
import numpy as np
import os
import threading

from ..sample import Sample
from ..config import ChannelConfig
from ..logger import LOGGER
from .util import array_to_samples, last_valid_index, nan_bfill, nan_ffill, samples_to_array


class _ChronosPredict(Protocol):
    """Bound ``pipeline.predict``: context tensor plus ``prediction_length``."""

    def __call__(self, inputs: Any, prediction_length: int) -> Any: ...


class ChronosForecasterBase(abc.ABC):
    """
    Shared univariate Chronos fill: interpolate interior holes (backfilled/forward-filling
    of NaNs), then forecast the trailing horizon. Intended to run on CPU.
    """

    _default_model_path: str = ''

    # One pipeline at a time: Hub download and predict. Channel threads share
    # Chronos instances; CPU torch / from_pretrained are not thread-safe.
    _pipeline_lock = threading.RLock()

    def __init__(
            self, model_path: Optional[str] = None, cache_dir: Optional[str] = None, **kwargs: Any
        ):
        LOGGER.warning(f'ChronosForecasterBase unknown kwargs: {kwargs}')
        self.model_path = model_path or self._default_model_path
        self.cache_dir = ChronosForecasterBase._get_weights_cache_dir(cache_dir)
        self._pipeline = None  # loaded on first predict; constructor stays extra-free

    def forecast(self, samples: Sequence[Sample], config: ChannelConfig) -> Sequence[Sample]:
        """
        First, fill holes from history (backfilled/forward-filling of NaNs).
        Then, predict the trailing horizon. Known values are left untouched.
        """
        # Convert the samples to a numpy array.
        values = samples_to_array(samples)
        # Find the last valid index.
        last = last_valid_index(values)
        # If there is no valid history, return the samples with forecast quality, so the grid stays regular.
        if last is None:
            return array_to_samples(samples, values, 'forecast')

        # Interpolate interior holes through the last known value, then forecast the rest.
        context = nan_bfill(nan_ffill(values[: last + 1]))
        values[: last + 1] = context
        horizon = len(values) - last - 1
        if horizon:
            values[last + 1:] = self._predict(context, horizon)
        return array_to_samples(samples, values, 'forecast')


    @abc.abstractmethod
    def _pipeline_class(self) -> type:
        """Return the Chronos pipeline class (imports the optional extra)."""
        ...

    @abc.abstractmethod
    def _pipeline_inputs(self, context: np.ndarray) -> np.ndarray:
        """Reshape the univariate context to the pipeline's expected tensor layout."""
        ...

    @abc.abstractmethod
    def _quantiles_array(self, raw: Any) -> np.ndarray:
        """Reduce pipeline output to shape ``(n_quantiles, horizon)``."""
        ...

    def _ensure_pipeline(self) -> tuple[_ChronosPredict, Optional[list[float]]]:
        """Load the CPU pipeline once (torch + Chronos extra, Hub weights)."""
        if self._pipeline is not None:
            return self._pipeline.predict, self._pipeline.quantiles

        with ChronosForecasterBase._pipeline_lock:
            # Another thread may have finished loading while this one waited.
            if self._pipeline is not None:
                return self._pipeline.predict, self._pipeline.quantiles

            # Load the optional torch package.
            torch = self._require_extra('torch')
            # Load the Chronos pipeline class.
            pipeline_cls = self._require_extra(self._pipeline_class)

            # Pin CPU even if a GPU is present: the extra ships CPU torch wheels only (see pyproject.toml).
            kwargs = {
                'device_map': 'cpu',
                'torch_dtype': torch.float32,
            }

            # Add the cache directory if provided.
            if self.cache_dir:
                kwargs['cache_dir'] = self.cache_dir

            cache_dir = self.cache_dir or 'Hugging Face default'
            LOGGER.debug(
                f'{type(self).__name__} loading model={self.model_path} cache_dir={cache_dir}'
            )

            # Load the pipeline from the model path.
            self._pipeline = pipeline_cls.from_pretrained(self.model_path, **kwargs)
            return self._pipeline.predict, self._pipeline.quantiles

    def _predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Zero-shot forecast of ``horizon`` steps from ``context`` (median quantile)."""
        with ChronosForecasterBase._pipeline_lock:
            # Load the optional torch package.
            torch = self._require_extra('torch')
            # Load the pipeline (nested lock: same RLock as this block).
            predict, quantiles = self._ensure_pipeline()
            # Convert the context to a torch tensor.
            context_tensor = torch.as_tensor(self._pipeline_inputs(context))
            # Predict the trailing horizon.
            raw = predict(context_tensor, prediction_length=horizon)
            # Reduce the quantile tensor to a 1-D median forecast of length ``horizon``.
            return self._median_point_forecast(raw, quantiles, horizon)

    def _median_point_forecast(self, raw: Any, quantiles: Optional[list[float]], horizon: int) -> np.ndarray:
        """Pick the median quantile from pipeline output as a 1-D series of length ``horizon``."""
        arr = np.asarray(self._quantiles_array(raw), dtype=np.float64)
        if quantiles:
            try:
                idx = list(quantiles).index(0.5)
            except ValueError:
                idx = len(list(quantiles)) // 2
        else:
            idx = arr.shape[0] // 2
        point = np.asarray(arr[idx], dtype=np.float64).reshape(-1)
        if point.size < horizon:
            raise ValueError(
                f'Chronos forecast length {point.size} is shorter than horizon {horizon}'
            )
        return point[:horizon]

    @staticmethod
    def _get_weights_cache_dir(cache_dir: Optional[str] = None) -> Optional[str]:
        """
        Cache for model weights: explicit ``cache_dir``, else environment variable ``HF_HUB_CACHE``,
        else ``None`` for using the default (Hugging Face's default).
        """
        if cache_dir:
            return cache_dir
        env = os.environ.get('HF_HUB_CACHE')
        if env:
            return env
        return None

    @staticmethod
    def _require_extra(target: str | Callable[[], Any]) -> Any:
        """
        Load an optional extra: a module name, or a callable that imports one.
        Missing packages raise the chronos extra install hint.
        """
        try:
            if callable(target):
                return target()
            return importlib.import_module(target)
        except ImportError as exc:
            hint = 'Chronos forecasters require the chronos extra. Install with: uv sync --extra chronos'
            raise ImportError(hint) from exc
