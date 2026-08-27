import logging
import time

# Render %(asctime)s in UTC to match the UTC timestamps in the data.
logging.Formatter.converter = time.gmtime

LOGGER = logging.getLogger('rdp-regularizer')

# Disable propagation to root to avoid duplicate lines.
LOGGER.propagate = False
