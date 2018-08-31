import numpy
import torch

from typing import List
from tmol.system.restypes import Residue
from tmol.system.packed import PackedResidueSystem
import pytest


@pytest.fixture
def partitioned_ubq_system(ubq_res: List[Residue],) -> List[PackedResidueSystem]:
    """Partition ubq system into disconnected 3-mer segments.

    Generate partition ubq system with 3-mer segments separated by 2-mer cuts,
    (ie. _xxx__xxx__xxx__xxx_) as a list of systems.
    """

    ss = 3  # segment size
    cs = 2  # cut size

    ts = ss + cs  # total size of each "section"
    # start index of segment inside section
    # cutting half from "upstream" and "downstream
    si = cs // 2

    return [
        PackedResidueSystem.from_residues(ubq_res[i + si : i + si + ss])
        for i in range(0, (len(ubq_res) // ts) * ts, ts)
    ]


def stack_score_graphs(score_graphs):
    assert len(set(g.system_size for g in score_graphs)) == 1

    start_layers = numpy.cumsum([0] + [g.stack_depth for g in score_graphs])

    stack_depth = start_layers[-1]

    coords = torch.cat([g.coords for g in score_graphs])

    bonds = numpy.concatentate(
        [
            g.bonds + numpy.array([[sl, 0, 0]])
            for sl, g in zip(start_layers, score_graphs)
        ]
    )

    atom_types = numpy.concatentate([g.atom_typs for g in score_graphs])


def test_stacked_scoring(partitioned_ubq_system):
    pass
