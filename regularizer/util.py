import datetime
import math
import re
import redis
import typing

from .logger import LOGGER

# Used to parse duration strings into timedeltas.
__DURATION_UNITS = {
    'ms': datetime.timedelta(milliseconds=1),
    's': datetime.timedelta(seconds=1),
    'm': datetime.timedelta(minutes=1),
    'h': datetime.timedelta(hours=1),
    'd': datetime.timedelta(days=1),
    'w': datetime.timedelta(weeks=1),
}
__DURATION_PATTERN = re.compile(
    r'^\s*(\d+(?:\.\d+)?)\s*('
    + '|'.join(sorted(__DURATION_UNITS, key=len, reverse=True))
    + r')\s*$'
)

def parse_duration(value: typing.Any) -> datetime.timedelta:
    """
    Converts duration strings ('30s', '1m', '15m', '12h') to timedelta.
    Bare numbers are interpreted as seconds.
    Timedeltas are returned unchanged.
    """
    if isinstance(value, str):
        match = __DURATION_PATTERN.match(value)
        if match:
            amount, unit = match.groups()
            return float(amount) * __DURATION_UNITS[unit]
        raise ValueError(f'Invalid duration: {value!r}')
    if isinstance(value, (int, float)):
        return datetime.timedelta(seconds=value)
    if isinstance(value, datetime.timedelta):
        return value

    raise ValueError(f'Invalid duration: {value!r}')


def next_polling_timestamp(
        now: float,
        polling_interval: datetime.timedelta,
        offset: datetime.timedelta = datetime.timedelta(0),
    ) -> float:
    """
    Next wall-clock tick at or after ``now``, aligned to
    ``epoch + k * polling_interval + offset``.
    """
    interval_s = polling_interval.total_seconds()
    offset_s = offset.total_seconds()
    return (math.floor(now / interval_s) + 1) * interval_s + offset_s


def next_polling_datetime(
        now: datetime.datetime,
        polling_interval: datetime.timedelta,
        offset: datetime.timedelta = datetime.timedelta(0),
    ) -> datetime.datetime:
    """
    UTC datetime of the next aligned live tick at or after ``now``.
    """
    epoch = next_polling_timestamp(now.timestamp(), polling_interval, offset)
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc)


def load_redis_connection_pool(redis_config: dict) -> redis.ConnectionPool:
    """
    Parses the configuration and instantiates the Redis connection pool
    """
    host = redis_config['host']
    port = redis_config['port']
    db = redis_config['db']
    pwd = redis_config.get('password')
    LOGGER.info(f'Configure redis connection to {host}:{port} using db {db}')

    if pwd:
        pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True, password=pwd)
    else:
        pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)

    client = redis.Redis(connection_pool=pool)
    client.ping()  # Will raise an exception in case a connection error occurs
    LOGGER.info(f'Redis connection to {host}:{port} using db {db} is alive.')

    return pool
