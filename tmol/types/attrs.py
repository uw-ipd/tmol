from .converters import get_converter
from .validators import get_validator


class ConvertAttrs:
    def __attrs_post_init__(self):
        for a in self.__attrs_attrs__:
            if not a.converter:
                object.__setattr__(
                    self,
                    a.name,
                    get_converter(a.type)(getattr(self, a.name)),
                )


class ValidateAttrs:
    def __attrs_post_init__(self):
        for a in self.__attrs_attrs__:
            if not a.validator:
                get_validator(a.type)(getattr(self, a.name))
