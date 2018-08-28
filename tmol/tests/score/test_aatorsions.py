import torch
import numpy

from tmol.utility.reactive import reactive_attrs
from tmol.score.total_score import TotalScoreComponentsGraph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.torsions import AlphaAABackboneTorsionProvider


@reactive_attrs(auto_attribs=True)
class TCartTorsions(
    CartesianAtomicCoordinateProvider,
    AlphaAABackboneTorsionProvider,
    TorchDevice,
    TotalScoreComponentsGraph,
):
    """Cart total."""

    pass


def test_create_torsion_provider(ubq_system):
    src = TCartTorsions.build_for(ubq_system)
    assert src

    gold_phi = torch.tensor(
        [
            numpy.nan,
            -91.0202,
            -131.0989,
            -115.9912,
            -118.0303,
            -95.2261,
            -99.5796,
            -73.4252,
            -101.3971,
            77.4447,
            -96.2711,
            -119.9447,
            -109.4737,
            -101.4413,
            -126.4132,
            -111.7952,
            -139.0196,
            -120.0428,
            -54.9367,
            -79.8384,
            -71.0060,
            -83.7038,
            -61.3275,
            -57.5667,
            -65.4883,
            -58.4233,
            -60.8123,
            -66.1219,
            -64.2164,
            -69.9996,
            -62.1262,
            -53.4121,
            -93.5800,
            -123.5788,
            81.2164,
            -79.7333,
            -57.0030,
            -57.1842,
            -68.2389,
            -95.7981,
            -84.8311,
            -121.2432,
            -103.5581,
            -122.0749,
            -144.2704,
            48.1713,
            61.7145,
            -115.0664,
            -85.7723,
            -79.5553,
            -101.8294,
            -48.1901,
            -82.9252,
            -85.4496,
            -104.5098,
            -61.2180,
            -63.8914,
            -55.5523,
            -91.0216,
            57.9396,
            -88.6669,
            -103.4373,
            -54.7883,
            66.8974,
            -71.1469,
            -119.1522,
            -103.1357,
            -105.5834,
            -106.9736,
            -108.1151,
            -96.0197,
            -117.6195,
            -83.7244,
            -97.6604,
            120.4148,
            174.1601,
        ]
    )  # yapf: disable
    numpy.testing.assert_allclose(
        (src.phi_tor * 180 / numpy.pi).detach().numpy().squeeze(), gold_phi, atol=1e-4
    )
