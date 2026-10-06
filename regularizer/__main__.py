import click
import pathlib
import pyrdp_commons.cli
import threading

from .channel import Channel
from .config import ChannelConfig
from .history import HistoryProvider
from .logger import LOGGER
from .tools import FORECASTERS, create_forecaster, create_imputer
from .tools.chronos import ChronosForecasterBase
from .util import load_redis_connection_pool


def _forecaster_for_channel(channel_config: ChannelConfig, shared: dict):
    """
    Most forecasters have a small footprint (in terms of memory usage). In these
    cases, one forecaster instance is created per channel. For forecasters with a
    large footprint (Chronos-based forecasters), we share one instance per (name, kwargs).
    """
    # Get forecaster name and kwargs.
    name = channel_config.forecaster
    kwargs = channel_config.forecaster_kwargs
    # Get forecaster factory.
    factory = FORECASTERS.get(name)
    # If factory is None or not a Chronos-based forecaster, create a new instance per channel.
    if factory is None or not issubclass(factory, ChronosForecasterBase):
        return create_forecaster(name, kwargs)

    # Get shared forecaster instance.
    key = (name, tuple(sorted(kwargs.items())))
    forecaster = shared.get(key)
    if forecaster is None:
        # If no shared forecaster instance exists, create a new one.
        forecaster = create_forecaster(name, kwargs)
        shared[key] = forecaster
    return forecaster


@click.command()
@click.option('-c', '--config', default='config.yml', envvar="REGULARIZER_CONFIG", help='config file path')
@click.option("--env", default=None, envvar="REGULARIZER_ENV", help="environment file path")
def main(config, env):
    # Read config file.
    config_file_path = pathlib.Path(config).resolve(strict=True)
    config = pyrdp_commons.cli.setup_app(config_file=str(config_file_path), env_file=env)
    LOGGER.warning(f'Config: {config}')

    # Redis config.
    redis_config = config['redis']
    redis_pool = load_redis_connection_pool(redis_config=redis_config)

    # Load channel configs.
    channel_configs = ChannelConfig.load_channel_configs(config['channels'])

    # Optional TimescaleDB history provider.
    history_provider = None
    if any(cc.history_provider is not None for cc in channel_configs):
        history_provider = HistoryProvider(config['timescale'])

    # Some forecasters are shared across channels. See `_forecaster_for_channel` for more details.
    shared_forecasters: dict = {}

    # Create channels.
    stop_event = threading.Event()
    channels = [
        Channel(config=channel_config, redis_pool=redis_pool,
                imputer=create_imputer(channel_config.imputer, channel_config.imputer_kwargs),
                forecaster=_forecaster_for_channel(channel_config, shared_forecasters),
                history_provider=history_provider if channel_config.history_provider else None,
                stop_event=stop_event)
        for channel_config in channel_configs
    ]
    for forecaster in shared_forecasters.values():
        forecaster._ensure_pipeline()

    # Run channels.
    LOGGER.info(f'Starting {len(channels)} channels ...')
    for channel in channels:
        channel.start()

    # Wait for stop event.
    try:
        while not stop_event.is_set():
            stop_event.wait(1)
    except KeyboardInterrupt:
        LOGGER.info('Stopping RDP regularizer service ...')
    finally:
        # Stop channels.
        stop_event.set()
        for channel in channels:
            channel.join(timeout=10)
        # Log success.
        LOGGER.info('RDP regularizer service stopped')

if __name__ == '__main__':
    main()
