"""Defines players used during model training."""


from enum import IntEnum, auto


class Player(IntEnum):
  NIL = auto()
  ALPHA = auto()
  BETA = auto()
