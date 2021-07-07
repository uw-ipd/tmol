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

from tmol.score.modules.bases import ScoreSystem, ScoreMethod
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


# TODO add a method to go from TERM (not method) keystrings to required method (XScore) classes


def get_full_score_system_for(packed_residue_system_or_system_stack):
    score_system = ScoreSystem.build_for(
        packed_residue_system_or_system_stack,
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
            "lk_ball_one": 1.0,
            "lk_ball_two": 1.0,
            "lk_ball_three": 1.0,
            "lk_ball_four": 1.0,
            "elec": 1.0,
            "cartbonded_lengths": 1.0,
            "cartbonded_angles": 1.0,
            "cartbonded_torsions": 1.0,
            "cartbonded_impropers": 1.0,
            "cartbonded_hxltorsions": 1.0,
            "dunbrack_one": 1.0,
            "dunbrack_two": 1.0,
            "dunbrack_three": 1.0,
            "hbond": 1.0,
            "rama": 1.0,
            "omega": 1.0,
        },
    )
    return score_system


def weights_keyword_to_score_method(keyword: str) -> ScoreMethod:
    conversion = {
        "lj": LJScore,
        "lk": LKScore,
        "lk_ball_one": LKBallScore,
        "lk_ball_two": LKBallScore,
        "lk_ball_three": LKBallScore,
        "lk_ball_four": LKBallScore,
        "elec": ElecScore,
        "cartbonded_lengths": CartBondedScore,
        "cartbonded_angles": CartBondedScore,
        "cartbonded_torsions": CartBondedScore,
        "cartbonded_impropers": CartBondedScore,
        "cartbonded_hxltorsions": CartBondedScore,
        "dunbrack_one": DunbrackScore,
        "dunbrack_two": DunbrackScore,
        "dunbrack_three": DunbrackScore,
        "hbond": HBondScore,
        "rama": RamaScore,
        "omega": OmegaScore,
    }
    return conversion[keyword]


def score_method_to_even_weights_dict(score_method: ScoreMethod) -> dict:
    conversion = {
        LJScore: {"lj": 1.0},
        LKScore: {"lk": 1.0},
        LKBallScore: {
            "lk_ball_one": 1.0,
            "lk_ball_two": 1.0,
            "lk_ball_three": 1.0,
            "lk_ball_four": 1.0,
        },
        ElecScore: {"elec": 1.0},
        CartBondedScore: {
            "cartbonded_lengths": 1.0,
            "cartbonded_angles": 1.0,
            "cartbonded_torsions": 1.0,
            "cartbonded_impropers": 1.0,
            "cartbonded_hxltorsions": 1.0,
        },
        DunbrackScore: {
            "dunbrack_one": 1.0,
            "dunbrack_two": 1.0,
            "dunbrack_three": 1.0,
        },
        HBondScore: {"hbond": 1.0},
        RamaScore: {"rama": 1.0},
        OmegaScore: {"omega": 1.0},
    }
    return conversion[score_method]
