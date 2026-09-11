"""
Sample data model for the RDP regularizer channels.
"""
import dataclasses
import datetime
import typing

Quality = typing.Literal['measured', 'imputed', 'forecast']

@dataclasses.dataclass(frozen=True)
class Sample:
    """A single time series sample."""
    timestamp: datetime.datetime  # timezone-aware, UTC
    value: typing.Optional[float]  # None marks a hole for impute/forecast input
    quality: Quality = 'measured'
