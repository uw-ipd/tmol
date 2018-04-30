from functools import singledispatch
from tmol.extern.typeguard import check_type

import attr


@attr.s(frozen=True, slots=True)
class TypeGuardValidator:
    """No-op 'converter', just validate type."""
    expected_type = attr.ib()

    def __call__(self, value):
        check_type("value", value, self.expected_type)


@singledispatch
def get_validator(type_annotation):
    return TypeGuardValidator(type_annotation)
