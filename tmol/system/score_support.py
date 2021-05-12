import math
import numpy
import torch

from typing import Optional

from tmol.database import ParameterDatabase

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor

from tmol.kinematics.operations import inverseKin

from tmol.database.scoring import RamaDatabase

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.system.kinematics import KinematicDescription

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.ljlk import LJScore, LKScore
from tmol.score.modules.lk_ball import LKBallScore
from tmol.score.modules.elec import ElecScore
from tmol.score.modules.cartbonded import CartBondedScore
from tmol.score.modules.dunbrack import DunbrackScore
from tmol.score.modules.hbond import HBondScore
from tmol.score.modules.rama import RamaScore
from tmol.score.modules.omega import OmegaScore


def kincoords_to_coords(
    kincoords, kintree, system_size
) -> Tensor[torch.float][:, :, 3]:
    """System cartesian atomic coordinates."""

    coords = torch.full(
        (system_size, 3),
        math.nan,
        dtype=kincoords.dtype,
        layout=kincoords.layout,
        device=kincoords.device,
        requires_grad=False,
    )

    idIdx = kintree.id[1:].to(dtype=torch.long)
    coords[idIdx] = kincoords[1:]

    return coords.to(torch.float)[None, ...]


def get_full_score_system_for(packed_residue_system: PackedResidueSystem):
    score_system = ScoreSystem.build_for(
        packed_residue_system,
        {
            LJScore,
            LKScore,
            LKBallScore,
            ElecScore,
            CartBondedScore,
            DunbrackScore,
            HBondScore,
            RamaScore,
            OmegaScore,
        },
        weights={
            "lj": 1.0,
            "lk": 1.0,
            "lk_ball": 1.0,
            "elec": 1.0,
            "cartbonded": 1.0,
            "dunbrack": 1.0,
            "hbond": 1.0,
            "rama": 1.0,
            "omega": 1.0,
        },
    )
    return score_system
