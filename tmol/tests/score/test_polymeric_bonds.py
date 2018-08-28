import numpy

from tmol.utility.reactive import reactive_attrs
from tmol.score.total_score import TotalScoreComponentsGraph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.polymeric_bonds import PolymericBonds


@reactive_attrs(auto_attribs=True)
class TCartBonds(
    CartesianAtomicCoordinateProvider,
    PolymericBonds,
    TorchDevice,
    TotalScoreComponentsGraph,
):
    """Cart total."""

    pass


def test_create_torsion_provider(ubq_system):
    src = TCartBonds.build_for(ubq_system)
    assert (src.upper.numpy()[:75] == (numpy.arange(75) + 1)).all()
    assert src.upper.numpy()[75] == -1
    assert (src.lower.numpy()[1:] == numpy.arange(75)).all()
    assert src.lower[0] == -1
