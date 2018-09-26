from functools import singledispatch

import attr
from typing import List

from tmol.types.functional import validate_args

from .factory import Factory
from tmol.utility.reactive import reactive_attrs

# Note: I'm not sure this class needs to be decorated with reactive_attrs?
@reactive_attrs(auto_attribs=True)
class ResidueProperties(Factory):
    """State for each residue the set of properties that describe it"""

    @staticmethod
    @singledispatch
    def factory_for(other, **_):
        """``clone``-factory, extract residue properties from other."""

        return dict(residue_properties=other.residue_properties)

    residue_properties: List[List[str]] = attr.ib()
