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
    src = TRama.build_for(ubq_system)
    assert src
    print(src.rama_scores)
