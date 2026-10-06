from typing import Any, Mapping, Optional

from .forecasting import FORECASTERS, ConstFillForecaster, Forecaster
from .imputation import IMPUTERS, ConstFillImputer, Imputer


def _create(registry, name: str, kwargs: Optional[Mapping[str, Any]], kind: str):
    try:
        factory = registry[name]
    except KeyError as exc:
        raise RuntimeError(f'Unknown {kind} {name!r}') from exc
    try:
        return factory(**dict(kwargs or {}))
    except TypeError as exc:
        raise RuntimeError(f'Invalid {kind} config for {name}: {exc}') from exc


def create_imputer(name: str, kwargs: Optional[Mapping[str, Any]] = None) -> Imputer:
    return _create(IMPUTERS, name, kwargs, 'imputer')


def create_forecaster(name: str, kwargs: Optional[Mapping[str, Any]] = None) -> Forecaster:
    return _create(FORECASTERS, name, kwargs, 'forecaster')


__all__ = [
    'FORECASTERS', 'ConstFillForecaster', 'Forecaster', 'create_forecaster',
    'IMPUTERS', 'ConstFillImputer', 'Imputer', 'create_imputer',
]
