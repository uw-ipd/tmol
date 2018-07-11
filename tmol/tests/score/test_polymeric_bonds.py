import pytest

import torch
import numpy

from tmol.utility.reactive import reactive_attrs
from tmol.tests.torch import requires_cuda
from tmol.score import (
    TotalScoreGraph,
    CartesianAtomicCoordinateProvider,
)
from tmol.score.polymeric_bonds import PolymericBonds
from tmol.database.chemical import three_letter_to_aatype


@reactive_attrs(auto_attribs=True)
class TCartBonds(CartesianAtomicCoordinateProvider, PolymericBonds,
                 TotalScoreGraph):
    """Cart total."""
    pass


def test_create_torsion_provider(ubq_system):
    src = TCartBonds.build_for(ubq_system)
    assert (src.upper.numpy()[:75] == (numpy.arange(75) + 1)).all()
    assert src.upper.numpy()[75] == -1
    assert (src.lower.numpy()[1:] == numpy.arange(75)).all()
    assert src.lower[0] == -1
