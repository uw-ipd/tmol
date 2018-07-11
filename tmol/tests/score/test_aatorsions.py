import pytest

import torch
import numpy

from tmol.utility.reactive import reactive_attrs
from tmol.tests.torch import requires_cuda
from tmol.score import (
    TotalScoreGraph,
    CartesianAtomicCoordinateProvider,
)
from tmol.score.torsions import (AlphaAABackboneTorsionProvider)
from tmol.database.chemical import three_letter_to_aatype


@reactive_attrs(auto_attribs=True)
class TCartTorsions(CartesianAtomicCoordinateProvider,
                    AlphaAABackboneTorsionProvider, TotalScoreGraph):
    """Cart total."""
    pass


def test_nab_aas(ubq_system):
    print([
        three_letter_to_aatype[res.residue_type.name3]
        for res in ubq_system.residues
    ])


def test_create_torsion_provider(ubq_system):
    src = TCartTorsions.build_for(ubq_system)
    assert src
    print(src.phi_tor * 180 / numpy.pi)
