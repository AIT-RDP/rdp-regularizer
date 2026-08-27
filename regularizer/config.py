"""
Channel configuration loaded from the channels section of config.yml.
"""
import dataclasses
import datetime
import typing

from .util import parse_duration

@dataclasses.dataclass(frozen=True)
class HistoryProviderConfig:
    """Identity used to fetch TimescaleDB history for bootstrap."""
    dp_name: str
    dp_location_code: typing.Optional[str] = None
    dp_unit: typing.Optional[str] = None
    dp_data_provider: typing.Optional[str] = None
    dp_device_id: typing.Optional[str] = None
    init_when_source_available: bool = False
    bootstrap_delay: datetime.timedelta = datetime.timedelta(seconds=10)


@dataclasses.dataclass(frozen=True)
class ChannelConfig:
    """
    Static configuration of a single measurement channel.

    Args:
        name: name of the channel (string)
        input_stream: name of the input stream (string)
        output_stream: name of the output stream (string)
        polling_interval: interval at which to poll the input stream (timedelta)
        update_interval: interval at which to update the regularizer (timedelta)
        jitter_tolerance: tolerance for jitter in the input stream (timedelta)
        window: window of the regularizer (timedelta)
        history_provider: optional TimescaleDB history provider (HistoryProviderConfig, default=None)
        offset: offset of the channel (timedelta, default=0)
        lag_time: lag time of the channel (timedelta, default=0)
        output_maxlen: maximum length of the output stream (int, default=200)
        data_provider_name: name of the data provider (string, default='rdp-regularizer')
        imputer: imputer to use (string, default='default')
        forecaster: forecaster to use (string, default='default')
    """
    name: str
    input_stream: str
    output_stream: str
    polling_interval: datetime.timedelta
    update_interval: datetime.timedelta
    jitter_tolerance: datetime.timedelta
    window: datetime.timedelta
    history_provider: typing.Optional[HistoryProviderConfig] = None
    offset: datetime.timedelta = dataclasses.field(default=datetime.timedelta(seconds=0))
    lag_time: datetime.timedelta = dataclasses.field(default=datetime.timedelta(seconds=0))
    output_maxlen: int = dataclasses.field(default=200)
    data_provider_name: str = dataclasses.field(default='rdp-regularizer')
    imputer: str = dataclasses.field(default='default')
    forecaster: str = dataclasses.field(default='default')

    @staticmethod
    def load_channel_configs(channels: dict) -> typing.List['ChannelConfig']:
        """
        Builds one ChannelConfig per entry in the channels dict.
        """
        if not channels:
            raise RuntimeError('channels config is empty')

        configs = []
        for channel_key, entry in channels.items():
            # Convert the entry to a dictionary.
            entry = dict(entry)

            # Optional: omit history_provider to skip Timescale bootstrap.
            hp_entry = entry.pop('history_provider', None)
            if hp_entry is not None:
                try:
                    if 'bootstrap_delay' in hp_entry:
                        hp_entry['bootstrap_delay'] = parse_duration(hp_entry['bootstrap_delay'])
                    entry['history_provider'] = HistoryProviderConfig(**hp_entry)
                except TypeError as exc:
                    raise RuntimeError(
                        f'Invalid history provider config for channel {channel_key}: {exc}'
                    ) from exc

            # Parse channel config, handle special cases.
            try:
                entry['name'] = channel_key
                entry['window'] = parse_duration(entry['window'])
                entry['update_interval'] = parse_duration(entry['update_interval'])
                entry['polling_interval'] = parse_duration(entry['polling_interval'])
                entry['jitter_tolerance'] = (
                    0.5 * entry['update_interval'] if 'jitter_tolerance' not in entry
                    else parse_duration(entry['jitter_tolerance'])
                )
                if 'offset' in entry:
                    entry['offset'] = parse_duration(entry['offset'])
                if 'lag_time' in entry:
                    entry['lag_time'] = parse_duration(entry['lag_time'])

                config = ChannelConfig(**entry)
            except KeyError as exc:
                raise RuntimeError(
                    f'Missing config field for channel {channel_key}: {exc}'
                ) from exc
            except TypeError as exc:
                raise RuntimeError(
                    f'Invalid channel config for channel {channel_key}: {exc}'
                ) from exc

            # Add the config to the list of configs.
            configs.append(config)

        return configs
