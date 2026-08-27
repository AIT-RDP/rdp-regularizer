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
from typing import Callable, Dict, List, Optional, Tuple

from .config import ChannelConfig
from .history import HistoryProvider, RegularizedHistoryStore
from .io import RedisStreamSink, RedisStreamSource
from .sample import Sample
from .tools import Forecaster, Imputer


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

    def bootstrap_history(
            self,
            sink: RedisStreamSink,
            source: RedisStreamSource,
            stop_event: threading.Event,
        ) -> bool:
        """
        Optionally waits for the Redis source, then fetches raw DB history,
        replays regularization, seeds in-memory history, and publishes bootstrap
        emissions to Redis.

        Returns False if startup should abort (source wait failed / stopped);
        True otherwise.
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

        # Poll the regularizer. This will finalize the grid points that can be emitted.
        if not self.poll(now=self._next_grid_ts, interval=self._config.window):
            self._logger.warning(f'No samples emitted during bootstrap')
            return True

        # Emit the history samples to the sink.
        sink.emit(self._history.samples)
        self._logger.info(f'Bootstrap complete: {len(self._history.samples)} samples emitted')
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
        # Initialize the list of finalized samples.
        finalized: List[Sample] = []

        while self._next_grid_ts <= horizon:
            # Get the next grid point.
            grid_ts = self._next_grid_ts
            # Get the pending entry for the grid point.
            pending_entry = self._pending.get(grid_ts)
            # If the pending entry exists, process it.
            if pending_entry is not None:
                # Process the pending entry as a measured sample.
                self._logger.debug('processing pending entry (measured)')
                sample = pending_entry[0]
                finalized.append(Sample(timestamp=grid_ts, value=sample.value, quality='measured'))
            elif any(ts > grid_ts for ts in self._pending):
                # Process the interior gap as an imputed sample.
                self._logger.debug('processing interior gap (imputing)')
                next_sample = self._pending[min(ts for ts in self._pending if ts > grid_ts)][0]
                value = self._imputer.impute(grid_ts, self._history_for(grid_ts),
                                             next_sample, self._config)
                self._logger.debug(f'imputed value {value} for interior gap at {grid_ts.isoformat()}')
                finalized.append(Sample(timestamp=grid_ts, value=value, quality='imputed'))
            elif now >= grid_ts + self._config.lag_time:
                # Process the deadline gap as a forecasted sample.
                self._logger.debug('processing deadline gap (forecasting)')
                value = self._forecaster.forecast(grid_ts, self._history_for(grid_ts),
                                                  self._config)
                self._logger.debug(f'forecast value {value} for grid point {grid_ts.isoformat()} '
                                  f'(no data by deadline)')
                finalized.append(Sample(timestamp=grid_ts, value=value, quality='forecast'))
            else:
                break

            # Remove the grid point from the pending list.
            self._pending.pop(grid_ts, None)
            # Update the next grid point.
            self._next_grid_ts += self._config.update_interval

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

    def _history_for(self, grid_ts: datetime) -> List[Sample]:
        """
        Returns the in-memory regularized history window ending at grid_ts.
        """
        return self._history.window(grid_ts - self._config.window, grid_ts)

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
