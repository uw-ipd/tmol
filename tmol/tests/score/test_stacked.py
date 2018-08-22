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


def test_stacked_scoring(partitioned_ubq_system):
    pass
