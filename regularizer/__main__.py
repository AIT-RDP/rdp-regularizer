import click
import pathlib
import pyrdp_commons.cli
import threading

from .channel import Channel
from .config import ChannelConfig
from .history import HistoryProvider
from .logger import LOGGER
from .tools import FORECASTERS, IMPUTERS
from .util import load_redis_connection_pool

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

    # Start channels.
    stop_event = threading.Event()
    channels = [
        Channel(config=channel_config, redis_pool=redis_pool,
                imputer=IMPUTERS[channel_config.imputer](),
                forecaster=FORECASTERS[channel_config.forecaster](),
                history_provider=history_provider if channel_config.history_provider else None,
                stop_event=stop_event)
        for channel_config in channel_configs
    ]

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
