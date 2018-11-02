from functools import singledispatch

import attr
from typing import List

from .factory import Factory
from tmol.utility.reactive import reactive_attrs


@reactive_attrs(auto_attribs=True)
class ResidueProperties(Factory):
    """State for each residue the set of properties that describe it.

    Each residue may be described by more than one property, and each
    property is a string of unknown length, therefore these properties
    are represented as a list of lists of strings.
    """

    @staticmethod
    @singledispatch
    def factory_for(other, **_):
        """``clone``-factory; extract residue properties from other."""

        return dict(residue_properties=other.residue_properties)

    residue_properties: List[List[str]] = attr.ib()
