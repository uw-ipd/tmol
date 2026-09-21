from enum import Enum


class AutoNumber(Enum):
    """Enum base that assigns consecutive integer values in declaration order."""

    def __new__(cls):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj
