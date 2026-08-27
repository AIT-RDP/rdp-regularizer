"""
Redis stream sink emitting the strictly regular output series.
"""
from __future__ import annotations

import json
import logging
import typing

import redis

from ..config import ChannelConfig
from ..sample import Sample


class RedisStreamSink:
    """
    Publishes one entry per batch of finalized grid points, in the same JSON-array
    field shape the data crawler uses (so rdp-redsql style consumers can unpack it),
    plus a `quality` field marking measured/imputed/forecast values.
    """

    def __init__(self, client: redis.Redis, config: ChannelConfig, logger: logging.Logger):
        self._client = client
        self._config = config
        self._logger = logger

    def emit(self, samples: typing.Sequence[Sample]) -> None:
        if not samples:
            return

        # Create the entry from the samples.
        entry = {
            'valid_time': json.dumps([s.timestamp.isoformat() for s in samples]),
            'value': json.dumps([s.value for s in samples]),
            'quality': json.dumps([s.quality for s in samples]),
        }

        # Add the history provider configuration to the entry.
        hp_config = self._config.history_provider
        if hp_config is not None:
            for field in ('name', 'location_code', 'unit', 'device_id'):
                value = getattr(hp_config, f'dp_{field}')
                if value is not None:
                    entry[field] = json.dumps(value)
        else:
            entry['name'] = json.dumps(self._config.name)

        # Add the data provider name to the entry.
        entry['data_provider'] = json.dumps(self._config.data_provider_name)

        # Emit the entry to the Redis stream.
        self._client.xadd(self._config.output_stream, entry,
                          maxlen=self._config.output_maxlen, approximate=True)

        self._logger.debug(f'Emitted {len(samples)} samples '
                           f'({samples[0].timestamp.isoformat()} .. {samples[-1].timestamp.isoformat()}) '
                           f'to {self._config.output_stream}')
