"""Defines logger used globally. Only exports "logger" symbol."""

import logging


class _ColoredFormatter(logging.Formatter):
  """Defines custom colored logging for logger."""

  _FORMAT = "{} [{}] \x1b[0m %(message)s"

  FORMATS = {
    logging.DEBUG: _FORMAT.format("\x1b[30;47m", "DEBG"),
    logging.INFO: _FORMAT.format("\x1b[30;42m", "INFO"),
    logging.WARNING: _FORMAT.format("\x1b[30;43m", "WARN"),
    logging.ERROR: _FORMAT.format("\x1b[30;41m", "ERRO"),
    logging.CRITICAL: _FORMAT.format("\x1b[31;40;1m", "CRIT"),
  }

  def format(self, record: logging.LogRecord) -> str:
    """Format the incoming record according to our styles."""
    return logging.Formatter(self.FORMATS.get(record.levelno)).format(record)


logger = logging.getLogger("peer")
logger.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(_ColoredFormatter())

logger.addHandler(_console_handler)
