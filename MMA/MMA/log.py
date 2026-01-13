import logging
from logging.config import dictConfig
from pathlib import Path
from sys import stdout
from typing import Optional

from mma.settings import settings

selected_log_level = logging.DEBUG if settings.debug else logging.INFO

def get_logger(name: Optional[str] = None) -> "logging.Logger":
    logger = logging.getLogger("MMA")
    logger.setLevel(logging.INFO)
    return logger
