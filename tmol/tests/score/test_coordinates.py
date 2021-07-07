import pytest

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.ljlk import LJScore
from tmol.score.modules.stacked_system import StackedSystem
from tmol.score.modules.coords import coords_for
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


def test_coord_clone_factory_from_stacked_systems(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    score_system = ScoreSystem.build_for(twoubq, {LJScore}, {"lj": 1.0})
    coords = coords_for(twoubq, score_system)

    assert coords.shape == (2, 1472, 3)


def test_non_uniform_sized_stacked_system_coord_factory(ubq_res):
    sys1 = PackedResidueSystem.from_residues(ubq_res[:6])
    sys2 = PackedResidueSystem.from_residues(ubq_res[:8])
    sys3 = PackedResidueSystem.from_residues(ubq_res[:4])

    twoubq = PackedResidueSystemStack((sys1, sys2, sys3))

    score_system = ScoreSystem.build_for(twoubq, {LJScore}, {"lj": 1.0})
    coords = coords_for(twoubq, score_system)

    assert coords.shape == (3, sys2.coords.shape[0], 3)
