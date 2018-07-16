import pytest

import torch
import numpy

from tmol.utility.reactive import reactive_attrs
from tmol.tests.torch import requires_cuda
from tmol.score import (
    CartesianAtomicCoordinateProvider,
)
from tmol.score.total_score import TotalScoreComponentsGraph
from tmol.score.rama.rama_score_graph import RamaScoreGraph


@reactive_attrs(auto_attribs=True)
class TRama(CartesianAtomicCoordinateProvider, RamaScoreGraph,
            TotalScoreComponentsGraph):
    """Cart total."""
    pass


def test_create_torsion_provider(ubq_system):
    trama = TRama.build_for(ubq_system)
    assert trama

    gold_rama_scores = numpy.array([
        -0.7474, -0.3156, -0.7905, -0.3412, -0.3135, -0.2807, 0.3014, 0.1461,
        -1.8280, -0.4346, -0.0333, -0.2466, 0.2117, -0.0184, 0.2571, 0.9067,
        5.7752, 0.6475, -0.8588, -0.7846, -0.7969, 0.3320, -0.4117, 1.4239,
        1.1622, -0.6051, -0.5258, -0.7472, 0.0234, 1.5939, 1.4089, -0.4842,
        2.2174, -1.8470, 3.4690, 5.0221, 0.2429, -0.0765, -0.9289, -0.5931,
        0.3425, -0.3653, -0.8009, 1.2904, 1.6205, 0.2987, -0.4256, -0.6167,
        -1.0429, -0.3423, 2.3625, 0.1690, 0.5093, -0.5014, 0.0158, -0.6732,
        0.8884, -0.0899, -0.7584, -0.7619, 0.7566, 0.5671, 2.0990, -1.1896,
        0.0419, -0.6342, -0.4525, 0.0383, -0.4488, -0.6283, 0.9796, -0.9366,
        1.5038, 2.6801
    ])

    numpy.testing.assert_allclose(
        trama.rama_scores.detach().numpy(), gold_rama_scores, atol=1e-4
    )

    gold_total_score = 16.6269
    assert abs(trama.total_rama - gold_total_score) < 1e-4
