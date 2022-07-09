import math
import torch

from tmol.types.torch import Tensor

from tmol.score.modules.bases import ScoreSystem, ScoreMethod
from tmol.score.modules.constraint import ConstraintScore
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


# TODO add a method to go from TERM (not method) keystrings
# to required method (XScore) classes
def get_full_score_system_for(packed_residue_system_or_system_stack, device):
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
            "lk_ball": 0.92,
            "lk_ball_iso": -0.38,
            "lk_ball_bridge": -0.33,
            "lk_ball_bridge_uncpl": -0.33,
            "elec": 1.0,
            "cartbonded_lengths": 1.0,
            "cartbonded_angles": 1.0,
            "cartbonded_torsions": 1.0,
            "cartbonded_impropers": 1.0,
            "cartbonded_hxltorsions": 1.0,
            "dunbrack_rot": 0.76,
            "dunbrack_rotdev": 0.69,
            "dunbrack_semirot": 0.78,
            "hbond": 1.0,
            "rama": 1.0,
            "omega": 0.48,
        },
        device=device,
    )
    return score_system


def weights_keyword_to_score_method(keyword: str) -> ScoreMethod:
    conversion = {
        "constraint_atompair": ConstraintScore,
        "constraint_dihedral": ConstraintScore,
        "constraint_angle": ConstraintScore,
        "lj": LJScore,
        "lk": LKScore,
        "lk_ball": LKBallScore,
        "lk_ball_iso": LKBallScore,
        "lk_ball_bridge": LKBallScore,
        "lk_ball_bridge_uncpl": LKBallScore,
        "elec": ElecScore,
        "cartbonded_lengths": CartBondedScore,
        "cartbonded_angles": CartBondedScore,
        "cartbonded_torsions": CartBondedScore,
        "cartbonded_impropers": CartBondedScore,
        "cartbonded_hxltorsions": CartBondedScore,
        "dunbrack_rot": DunbrackScore,
        "dunbrack_rotdev": DunbrackScore,
        "dunbrack_semirot": DunbrackScore,
        "hbond": HBondScore,
        "rama": RamaScore,
        "omega": OmegaScore,
    }
    return conversion[keyword]


def score_method_to_even_weights_dict(score_method: ScoreMethod) -> dict:
    conversion = {
        ConstraintScore: {
            "constraint_atompair": 1.0,
            "constraint_dihedral": 1.0,
            "constraint_angle": 1.0,
        },
        LJScore: {"lj": 1.0},
        LKScore: {"lk": 1.0},
        LKBallScore: {
            "lk_ball": 1.0,
            "lk_ball_iso": 1.0,
            "lk_ball_bridge": 1.0,
            "lk_ball_bridge_uncpl": 1.0,
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
            "dunbrack_rot": 1.0,
            "dunbrack_rotdev": 1.0,
            "dunbrack_semirot": 1.0,
        },
        HBondScore: {"hbond": 1.0},
        RamaScore: {"rama": 1.0},
        OmegaScore: {"omega": 1.0},
    }
    return conversion[score_method]
