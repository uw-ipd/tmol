import torch
import numpy

from tmol.utility.reactive import reactive_attrs
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.residue_properties import ResidueProperties


@reactive_attrs(auto_attribs=True)
class TCartResProps(CartesianAtomicCoordinateProvider, ResidueProperties, TorchDevice):
    """Cart total."""

    pass


def test_create_residue_properties(ubq_system):
    src = TCartResProps.build_for(ubq_system)

    assert len(src.residue_properties) == len(ubq_system.residues)
    assert src.residue_properties[0][0] == "aa.alpha.l.methionine"


def test_clone_torsion_provider(ubq_system):
    src = TCartResProps.build_for(ubq_system)
    TCartResProps.build_for(src)
