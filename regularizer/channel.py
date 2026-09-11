"""
Worker thread tying together source, regularizer, gap filling and sink for one
datapoint (dp_name/dp_location_code) pair.
"""
import logging
import redis
import sched
import threading
import time

from typing import Optional

from .config import ChannelConfig
from .history import HistoryProvider
from .io import RedisStreamSink, RedisStreamSource
from .regularizer import TimeGridRegularizer
from .tools import Forecaster, Imputer
from .util import next_polling_timestamp


class Channel(threading.Thread):
    """
    Runs `read -> regularize -> impute/forecast -> emit` on a wall-clock-aligned
    schedule (`epoch + k * polling_interval`) via sched.scheduler. Each step
    drains the Redis backlog with non-blocking reads, then polls the grid.
    """

    def __init__(self, config: ChannelConfig, redis_pool: redis.ConnectionPool,
                 imputer: Imputer, forecaster: Forecaster,
                 history_provider: Optional[HistoryProvider],
                 stop_event: threading.Event):
        """
        Initialize the channel.

        Args:
            config: The channel configuration object
            redis_pool: The Redis connection pool object
            imputer: The imputer to use object
            forecaster: The forecaster to use object
            history_provider: The optional TimescaleDB history provider object
            stop_event: The event to signal when the channel should stop object
        """
        super().__init__(name=f'channel-{config.name}', daemon=True)
        self._config = config
        self._stop_event = stop_event
        self._logger = logging.getLogger(f'rdp-regularizer.{config.name}')

        client = redis.Redis(connection_pool=redis_pool)
        self._source = RedisStreamSource(client=client, stream=config.input_stream,
                                         logger=self._logger)
        self._regularizer = TimeGridRegularizer(config=config, imputer=imputer,
                                                forecaster=forecaster,
                                                history_provider=history_provider,
                                                logger=self._logger)
        self._sink = RedisStreamSink(client=client, config=config, logger=self._logger)
        self._scheduler = sched.scheduler(time.time, self._stop_event.wait)

    def run(self) -> None:
        """
        Run the channel.

        This method is called when the channel is started. It will:
        - Log the channel start.
        - Bootstrap the history if a history provider is configured.
        - Schedule the first step.
        - Run the scheduler.
        - Log the channel stop.
        """
        self._logger.info(f'Channel started: {self._config.input_stream} -> '
                          f'{self._config.output_stream} '
                          f'(polling {self._config.polling_interval}, lag {self._config.lag_time})')

        # Bootstrap the history if a history provider is configured.
        if not self._regularizer.bootstrap_history(
            self._sink, self._source, self._stop_event
        ):
            return

        # Reuse the tick bootstrap targeted; if catch-up overran it, take the next one.
        first_live_ts = self._regularizer.first_live_ts
        now_epoch = time.time()
        if first_live_ts is None or first_live_ts.timestamp() <= now_epoch:
            first_live_epoch = self._next_polling_timestamp()
        else:
            first_live_epoch = first_live_ts.timestamp()

        # Schedule the first step.
        self._scheduler.enterabs(
            first_live_epoch,
            1, self._scheduled_step
        )

        # Run the scheduler.
        self._scheduler.run()

        self._logger.info('Channel stopped')

    def _step(self) -> None:
        """
        Read samples from the source, add them to the regularizer, and emit the poll.
        """
        # Read samples from the source.
        for sample in self._source.read():
            # Add the sample to the regularizer.
            self._regularizer.add(sample)
        # Emit the poll.
        self._sink.emit(self._regularizer.poll())

    def _scheduled_step(self) -> None:
        """
        Scheduled step to read samples from the source, add them to the regularizer, and emit the poll.
        """
        # Check if the channel should stop.
        if self._stop_event.is_set():
            return
        # Try to read samples from the source, add them to the regularizer, and emit the poll.
        try:
            self._step()
        except Exception as exc:
            self._logger.exception(f'Unexpected error in channel step: {exc}')
            self._logger.error(f'Try to resume in 5 seconds ...')
            self._stop_event.wait(5)

        # Schedule the next step.
        if not self._stop_event.is_set():
            self._scheduler.enterabs(
                self._next_polling_timestamp(),
                1, self._scheduled_step
            )

    def _next_polling_timestamp(self) -> float:
        """
        Next wall-clock tick, aligned to `epoch + k * interval_s + offset_s`.
        """
        return next_polling_timestamp(
            time.time(), self._config.polling_interval, self._config.offset
        )
