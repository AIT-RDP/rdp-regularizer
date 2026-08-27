"""
Redis stream source for the RDP regularizer streams published by the data crawler.
"""
import datetime
import json
import logging
import redis
import threading
import typing

from ..sample import Sample

RedisStreamMetadata = dict[str, typing.Any]

class RedisStreamSource:
    """
    Reads samples from a single Redis stream.

    The cursor starts at the current stream tip (so historical bootstrap is not
    replayed from Redis). Each `read()` then drains every entry published since
    the last seen ID (backlog between polls) with non-blocking XREAD calls.
    """

    def __init__(self, client: redis.Redis, stream: str, logger: logging.Logger):
        self._client = client
        self._stream = stream
        self._logger = logger
        self._last_id = self._stream_tip_id()

    def wait_until_available(
            self, stop_event: threading.Event, poll_interval_s: float = 30.
        ) -> bool:
        """
        Returns immediately if events are detected on a stream, otherwise blocks until the first new entry arrives or stop_event is set. In case of event detection, waits for `wait_after_event_s` seconds before returning.
        """
        self._logger.info(f'Checking for data on {self._stream} ...')
        try:
            # Prefer the newest existing entry (non-blocking).
            if self._client.xrevrange(self._stream, count=1):
                return True

            # Stream empty: wait for the first new entry.
            block_ms = int(poll_interval_s * 1000)
            while not stop_event.is_set():
                if self._client.xread(streams={self._stream: '$'}, block=block_ms, count=1):
                    return True

                # Check if the stop event has been set.
                if stop_event.wait(poll_interval_s):
                    break

                # Log the wait.
                self._logger.info(f'Waiting for data on {self._stream} ...')

        except redis.ResponseError as exc:
            self._logger.warning(
                f'Error checking for data on {self._stream}: {exc}; exiting ...'
            )

        return False

    def read(self) -> typing.List[Sample]:
        """
        Drains backlog since the last seen ID with non-blocking XREAD.
        Returns samples sorted by timestamp (empty if none).
        """
        self._logger.info(f'Watching for new samples on {self._stream}')
        samples: typing.List[Sample] = []

        # Continue reading from the stream until no more entries are available.
        while True:
            # Consume a batch of entries from the stream.
            batch = self._consume()
            # If no entries are available, break the loop.
            if not batch:
                break
            # Extend the list of samples with the parsed entries.
            samples.extend(batch)

        if not samples:
            self._logger.info(f'No samples found on {self._stream}')
            return []

        # Sort the samples by timestamp.
        samples.sort(key=lambda s: s.timestamp)
        self._logger.info(f'Found {len(samples)} samples on {self._stream}')
        return samples

    def _consume(self) -> typing.List[Sample]:
        """
        One non-blocking XREAD batch after `_last_id`. Advances the cursor for
        every entry, including malformed ones, so a bad message cannot stall
        catch-up.
        """
        # Retrieve the last 100 entries from the stream since the last seen ID.
        # count=100 caps how many stream entries Redis returns per XREAD, so a large backlog is read in small batches.
        # Without it, one call can dump the whole unread backlog into a single huge reply and a long processing spike.
        response = self._client.xread(streams={self._stream: self._last_id}, count=100)
        if not response:
            return []

        # Parse the entries and create Sample objects.
        samples = []
        for _stream_name, entries in response:
            for entry_id, fields in entries:
                self._last_id = entry_id
                try:
                    entry_samples, _ = self._parse_entry(fields)
                    samples.extend(entry_samples)
                except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._logger.warning(
                        f'Skipping malformed entry {entry_id} on {self._stream}: {exc}'
                    )
        return samples

    def _stream_tip_id(self) -> str:
        """
        Returns the ID of the newest entry so the next XREAD starts after it,
        or '0-0' when the stream is empty / missing.
        """
        try:
            # Retrieve the newest entry from the stream.
            entries = self._client.xrevrange(self._stream, count=1)
        except redis.ResponseError:
            return '0-0'
        if not entries:
            return '0-0'
        return entries[0][0]

    # @staticmethod
    def _parse_entry(self, fields: dict) -> typing.Tuple[typing.List[Sample], RedisStreamMetadata]:
        """
        Parse a single stream entry into samples and metadata.
        """
        # Entry may have metadata
        metadata = json.loads(fields.get('_metadata', '{}'))

        # Parse the _time and _value fields from the entry.
        times = json.loads(fields['_time'])
        values = json.loads(fields['_value'])

        # Convert the _time and _value fields to lists if they are not already.
        if not isinstance(times, list):
            times = [times]
        if not isinstance(values, list):
            values = [values]
        if len(times) != len(values):
            raise ValueError(f'_time/_value length mismatch ({len(times)} vs {len(values)})')

        # Create a list of Sample objects.
        samples = []
        for ts_raw, value in zip(times, values):
            # Convert the _time field to a datetime object.
            timestamp = datetime.datetime.fromisoformat(str(ts_raw).replace('Z', '+00:00'))
            # If the timestamp is not timezone-aware, set it to UTC.
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)

            samples.append(Sample(timestamp=timestamp.astimezone(datetime.timezone.utc),
                                  value=float(value), quality='measured'))
        return samples, metadata
