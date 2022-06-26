"""Defines the game base class (i.e. required methods) for training the model."""


from abc import ABC, abstractmethod
from math import prod

from pydantic import BaseModel, NonNegativeInt

from interface.problem.board import Board

DEFAULT_ADDITIONAL_PARAMETERS = 0


class Game(BaseModel, ABC):
  """Game object holding information about the state, valid moves, and updating the board."""

  board: Board

  additional_parameters: NonNegativeInt = DEFAULT_ADDITIONAL_PARAMETERS

  class Config:
    frozen = True

  @abstractmethod
  def total_actions(self) -> NonNegativeInt:
    """Number of maximum number of different actions that can be taken on a turn."""
    return prod(self.board.size()) + self.additional_parameters
