"""
Regularization of a jittery, gappy input series onto a strict time grid.

The live grid advances monotonically: each frontier point is finalized once,
in order, as measured, imputed, or forecast. Late measured arrivals may
re-emit corrections that upgrade prior imputed/forecast points still held
in history (without rewinding the frontier):

- measured data within the jitter tolerance is snapped onto the grid,
- interior gaps (bounded by later measured data) are filled by the imputer,
- grid points with no data by the deadline (grid_time + lag_time) are filled
  by the forecaster.
"""
import logging
import threading
import time

from datetime import datetime, timedelta, timezone
from math import floor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import ChannelConfig
from .history import HistoryProvider, RegularizedHistoryStore
from .io import RedisStreamSink, RedisStreamSource
from .sample import Sample
from .tools import Forecaster, Imputer
from .util import next_polling_datetime


class TimeGridRegularizer:
    """
    Consumes irregular measured samples and produces a strictly regular sequence
    of samples on the grid `epoch + k * interval`.
    """

    # Epoch for snapping to the grid.
    _EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def __init__(self, config: ChannelConfig, imputer: Imputer, forecaster: Forecaster,
            history_provider: Optional[HistoryProvider], logger: logging.Logger,
            start_time: Optional[datetime] = None
        ) -> None:
        """
        Initialize the time grid regularizer.

        Args:
            config: channel configuration object
            imputer: imputer to use object
            forecaster: forecaster to use object
            history_provider: optional TimescaleDB history provider object
            logger: logger to use object
            start_time: optional start time to use (default=None)
        """
        self._config = config
        self._imputer = imputer
        self._forecaster = forecaster
        self._history_provider = history_provider
        self._logger = logger

        # Snapped samples waiting to be emitted, keyed by grid timestamp. Also keeps
        # the jitter distance so that the closest sample wins for duplicate grid points.
        self._pending: Dict[datetime, Tuple[Sample, timedelta]] = {}

        # Late measured upgrades for already-finalized imputed/forecast points,
        # drained by poll() ahead of newly finalized frontier samples.
        self._corrections: List[Sample] = []

        # First grid point of interest: data older than (start - lag_time) was already
        # published before this service started, so the grid begins right after it.
        start_time = start_time or datetime.now(timezone.utc)
        self._next_grid_ts = self._snap_to_grid(start_time, config.polling_interval, floor) - config.lag_time

        self._logger.debug(f'snapping start_time = {start_time.isoformat()} to _next_grid_ts = {self._next_grid_ts.isoformat()}')

        # Initialize the regularized history store.
        self._history = RegularizedHistoryStore(config.window)

        # First live poll tick targeted by the latest bootstrap, if any.
        self.first_live_ts: Optional[datetime] = None

    def bootstrap_history(
            self,
            sink: RedisStreamSink,
            source: RedisStreamSource,
            stop_event: threading.Event,
            first_live_ts: Optional[datetime] = None,
        ) -> bool:
        """
        Optionally waits for the Redis source, then fetches raw DB history,
        replays regularization, seeds in-memory history, and publishes bootstrap
        emissions to Redis.

        Returns False if startup should abort (source wait failed / stopped);
        True otherwise. Catch-up uses the same deadline clock as live poll
        (``now >= grid_ts + lag_time``) through the first scheduled live tick.
        """
        hp_config = self._config.history_provider

        # Skip bootstrap if history provider is not configured.
        if hp_config is None or self._history_provider is None:
            self._logger.info('Skipping history bootstrap')
            return True

        # Wait for source data to be available if configured.
        if hp_config.init_when_source_available:
            if source.wait_until_available(stop_event):
                self._logger.info('Source data is available, waiting a moment before starting the history bootstrap ...')
                time.sleep(hp_config.bootstrap_delay.total_seconds())
            else:
                self._logger.info('Channel stopped before source data was available')
                return False

        # Next aligned live tick after the bootstrap delay. Always strictly after
        # wall-clock now, so it sits about `window` ahead of bootstrap_start.
        if first_live_ts is None:
            first_live_ts = next_polling_datetime(
                datetime.now(timezone.utc),
                self._config.polling_interval,
                self._config.offset,
            )
        self.first_live_ts = first_live_ts

        # Compute the bootstrap window.
        bootstrap_start = self._next_grid_ts - self._config.window
        bootstrap_end = datetime.now(timezone.utc)

        # Reset the grid to the start of the bootstrap window.
        self._next_grid_ts = bootstrap_start
        self._pending.clear()
        self._corrections.clear()

        # Get the raw history from the history provider.
        history_raw = self._history_provider.get_history(
            hp_config.dp_name,
            dp_location_code=hp_config.dp_location_code,
            dp_device_id=hp_config.dp_device_id,
            dp_data_provider=hp_config.dp_data_provider,
            start=bootstrap_start, end=bootstrap_end
        )
        self._logger.info(f'Retrieved history: start={bootstrap_start.isoformat()}, end={bootstrap_end.isoformat()}, samples={len(history_raw)}')

        # Add the raw history to the regularizer.
        for sample in sorted(history_raw, key=lambda s: s.timestamp):
            self.add(sample)

        # History bootstrap horizon length for poll(): first_live_ts - bootstrap_start.
        # Equals window + (first_live_ts - startup_grid), so > window in normal startup.
        catchup_interval = first_live_ts - self._next_grid_ts
        self._logger.info(
            f'Bootstrap catch-up until first live tick {first_live_ts.isoformat()} '
            f'(interval={catchup_interval})'
        )
        # Catch-up interval is non-positive only if a caller passed a first_live_ts at
        # or before bootstrap_start.
        if catchup_interval <= timedelta(0):
            self._logger.warning(f'Invalid catch-up interval: {catchup_interval}')
            return True

        # Poll the regularizer. This will finalize the grid points that can be emitted.
        emitted = self.poll(now=first_live_ts, interval=catchup_interval)
        if not emitted:
            self._logger.warning(f'No samples emitted during bootstrap')
            return True

        # Emit the catch-up batch (in-memory history may already be trimmed to window).
        sink.emit(emitted)
        self._logger.info(f'Bootstrap complete: {len(emitted)} samples emitted')
        return True

    def add(self, sample: Sample) -> None:
        """
        Registers an incoming measured sample, snapping it onto the grid.
        """
        # Snap the sample to the grid.
        grid_ts = self._snap_to_grid(sample.timestamp, self._config.update_interval)
        # Calculate the distance from the sample to the grid point.
        distance = abs(sample.timestamp - grid_ts)
        # Log the sample addition.
        self._logger.debug(
            f'adding sample: timestamp = {sample.timestamp.isoformat()}, value = {sample.value}, quality = {sample.quality} '
            f'-> grid timestamp = {grid_ts.isoformat()}'
        )

        # Drop the sample if it is outside the jitter tolerance.
        if distance > self._config.jitter_tolerance:
            self._logger.warning(
                f'Dropping sample at {sample.timestamp.isoformat()}: '
                f'{distance} away from nearest grid point {grid_ts.isoformat()}')
            return

        # Add the sample to the history if it is within the jitter tolerance.
        if grid_ts < self._next_grid_ts:
            # Get the prior sample at the grid point.
            prior = self._history.get(grid_ts)
            # If the prior sample is an imputed or forecast, queue a correction.
            if prior is not None and prior.quality in ('imputed', 'forecast'):
                # Create a correction sample.
                correction = Sample(timestamp=grid_ts, value=sample.value, quality='measured')
                # Add the correction to the history.
                self._history.extend([correction])
                # Add the correction to the corrections list.
                self._corrections.append(correction)
                self._logger.info(
                    f'Queuing late measured correction at {grid_ts.isoformat()} '
                    f'(upgrading {prior.quality}, current grid point: {self._next_grid_ts.isoformat()})'
                )
            else:
                self._logger.warning(
                    f'Dropping late sample at {grid_ts.isoformat()} '
                    f'(current grid point: {self._next_grid_ts.isoformat()})'
                )
            return

        # Add the sample to the pending list.
        existing = self._pending.get(grid_ts)
        # If the sample is not in the pending list, or if it is closer to the
        # grid point than the existing sample, add it to the pending list.
        if existing is None or distance < existing[1]:
            self._pending[grid_ts] = (sample, distance)

    def poll(self, now: Optional[datetime] = None, interval: Optional[timedelta] = None) -> List[Sample]:
        """
        Returns all grid points that can be finalized, in order. A grid point is
        finalized when measured data for it (or beyond it) has arrived, or when
        `now` has passed its deadline `grid_time + lag_time`.
        """
        self._logger.debug('polling now')
        # Get the current time and interval.
        now = now or datetime.now(timezone.utc)
        interval = interval or self._config.polling_interval
        # Get the horizon.
        horizon = self._next_grid_ts + interval
        # Get the corrections.
        corrections = self._corrections
        self._corrections = []

        # Classify finalizable grid points; stop at the first point that must wait.
        slots: List[Sample] = []
        grid_ts = self._next_grid_ts
        while grid_ts <= horizon:
            pending_entry = self._pending.get(grid_ts)
            if pending_entry is not None:
                self._logger.debug('classifying pending entry (measured)')
                slots.append(Sample(timestamp=grid_ts, value=pending_entry[0].value,
                                    quality='measured'))
            elif any(ts > grid_ts for ts in self._pending):
                self._logger.debug('classifying interior gap (impute)')
                slots.append(Sample(timestamp=grid_ts, value=None, quality='imputed'))
            elif now >= grid_ts + self._config.lag_time:
                self._logger.debug('classifying deadline gap (forecast)')
                slots.append(Sample(timestamp=grid_ts, value=None, quality='forecast'))
            else:
                break
            grid_ts += self._config.update_interval

        finalized = self._fill_slots(slots) if slots else []

        for slot in slots:
            self._pending.pop(slot.timestamp, None)
        self._next_grid_ts = grid_ts

        # Extend the history with the finalized samples.
        self._history.extend(finalized)
        # Trim the history to the current time.
        self._history.trim(now)

        # Get the emitted samples.
        emitted = corrections + finalized

        self._logger.debug(f'after polling: _next_grid_ts = {self._next_grid_ts.isoformat()}')
        try:
            self._logger.debug(f'history: start = {self._history.samples[0].timestamp.isoformat()}, '
                                f'end = {self._history.samples[-1].timestamp.isoformat()}')
        except IndexError:
            self._logger.debug('no history samples available')

        # Get the pending keys.
        try:
            pending_keys = sorted(list(self._pending.keys()))
            self._logger.debug(f'pending: start = {pending_keys[0].isoformat()}, '
                                f'end = {pending_keys[-1].isoformat()}')
        except IndexError:
            self._logger.debug('no pending samples available')

        self._logger.info(f'number of emitted samples: {len(emitted)} '
                          f'(corrections={len(corrections)}, finalized={len(finalized)})')

        return emitted

    def _fill_slots(self, slots: Sequence[Sample]) -> List[Sample]:
        """
        Build history + slots (+ optional right-bound pending), impute interior
        holes and/or forecast deadline holes in one call each, then return the
        filled classified slots.
        """
        first_ts = slots[0].timestamp
        history = self._history.window(first_ts - self._config.window, first_ts)

        anchor: List[Sample] = []
        last = slots[-1]
        if last.value is None and last.quality == 'imputed':
            next_ts = min(ts for ts in self._pending if ts > last.timestamp)
            pending_sample = self._pending[next_ts][0]
            anchor = [Sample(timestamp=next_ts, value=pending_sample.value, quality='measured')]

        samples: Sequence[Sample] = list(history) + list(slots) + anchor
        if any(s.value is None and s.quality == 'imputed' for s in slots):
            self._logger.debug('imputing interior gaps')
            samples = self._imputer.impute(samples, self._config)
        if any(s.value is None and s.quality == 'forecast' for s in slots):
            self._logger.debug('forecasting deadline gaps')
            samples = self._forecaster.forecast(samples, self._config)

        by_ts = {s.timestamp: s for s in samples}
        finalized: List[Sample] = []
        for slot in slots:
            filled = by_ts.get(slot.timestamp)
            if filled is None or filled.value is None:
                raise ValueError(
                    f'tool left grid point {slot.timestamp.isoformat()} unfilled'
                )
            self._logger.debug(
                f'finalized {filled.quality} value {filled.value} at {filled.timestamp.isoformat()}'
            )
            finalized.append(filled)
        return finalized

    def _snap_to_grid(self, ts: datetime, interval: timedelta,
            snap_func: Callable[[float], float] = round
        ) -> datetime:
        """
        Snap the timestamp to the grid.
        """
        # Calculate the number of intervals since the epoch.
        k = snap_func((ts - self._EPOCH) / interval)
        # Return the snapped timestamp.
        return self._EPOCH + k * interval
