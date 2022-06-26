"""Defines ASCII text suitable for printing in Python."""


from re import compile

from pydantic import ConstrainedStr


class ASCII(ConstrainedStr):
  """Ensures that the text is printable."""

  strip_whitespace = True
  min_length = 1
  regex = compile("^[ -~]+$")
  strict = True

  class Config:
    frozen = True
