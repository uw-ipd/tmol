import attr
from functools import singledispatch
from ..extern.typeguard import check_type
import typing


@attr.s(frozen=True, slots=True)
class ValidateConvert:
    """No-op 'converter', just validate type."""
    expected_type = attr.ib()

    def __call__(self, value):
        check_type("value", value, self.expected_type)
        return value


@attr.s(frozen=True, slots=True)
class ConstructorConvert:
    """No-op 'converter', just validate type."""
    expected_type = attr.ib()

    def __call__(self, value):
        try:
            check_type("value", value, self.expected_type)
            return value
        except (ValueError, TypeError):
            return self.expected_type(value)


@singledispatch
def get_converter(type_annotation):
    return ConstructorConvert(type_annotation)


get_converter.register(typing._Union)(ValidateConvert)
get_converter.register(typing.TupleMeta)(ValidateConvert)
