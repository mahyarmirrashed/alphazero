"""Defines a board where games are played on."""


from abc import ABC, abstractmethod
from typing import Dict, Tuple

from pydantic import BaseModel, NonNegativeInt, validator
from utils.classes.ascii import ASCII

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WINNING_LENGTH = 4


class Board(BaseModel, ABC):
  """Board object representing playing area."""

  height: NonNegativeInt = DEFAULT_HEIGHT
  width: NonNegativeInt = DEFAULT_WIDTH
  winning_length: NonNegativeInt = DEFAULT_WINNING_LENGTH

  @validator("winning_length")
  def winning_length_fits(cls, value: NonNegativeInt, values: Dict) -> NonNegativeInt:
    assert value <= values["height"]
    assert value <= values["width"]
    return value

  class Config:
    frozen = True

  @abstractmethod
  def render(self) -> ASCII:
    """Return rendered ASCII representation of the board."""
    pass

  @abstractmethod
  def size(self) -> Tuple[NonNegativeInt, NonNegativeInt]:
    """Return the size of the board."""
    pass
