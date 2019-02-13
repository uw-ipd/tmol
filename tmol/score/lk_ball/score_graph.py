import torch

from tmol.utility.reactive import reactive_attrs, reactive_property

from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from ..ljlk.score_graph import _LJLKCommonScoreGraph

from .potentials.compiled import AttachedWaters, LKBall


@reactive_attrs
class LKBallIntraScore(IntraScore):
    @reactive_property
    def lkball_scores(target):
        assert target.coords.dim() == 3
        assert target.coords.shape[0] == 1

        assert target.ljlk_atom_types.dim() == 2
        assert target.ljlk_atom_types.shape[0] == 1

        assert target.bonded_path_length.dim() == 3
        assert target.bonded_path_length.shape[0] == 1
        lkball_op = LKBall(target.atom_type_params, target.ljlk_param_resolver)

        return lkball_op.apply(
            target.coords[0],
            target.coords[0],
            target.lkball_waters,
            target.lkball_waters,
            target.ljlk_atom_types[0],
            target.ljlk_atom_types[0],
            target.bonded_path_length[0],
        )

    @reactive_property
    def total_lk_ball_iso(lkball_scores):
        """total hbond score"""
        score_ind, score_val = lkball_scores
        return score_val[..., 0].sum()

    @reactive_property
    def total_lk_ball(lkball_scores):
        """total hbond score"""
        score_ind, score_val = lkball_scores
        return score_val[..., 1].sum()

    @reactive_property
    def total_lk_ball_bridge(lkball_scores):
        """total hbond score"""
        score_ind, score_val = lkball_scores
        return score_val[..., 2].sum()

    @reactive_property
    def total_lk_ball_bridge_uncpl(lkball_scores):
        """total hbond score"""
        score_ind, score_val = lkball_scores
        return score_val[..., 3].sum()


@score_graph
class LKBallScoreGraph(_LJLKCommonScoreGraph):
    @staticmethod
    def factory_for(val, device: torch.device, **_):
        """Overridable clone-constructor.

        Initialize from ``val.ljlk_database`` if possible, otherwise from
        ``parameter_database.scoring.ljlk``.
        """
        if device.type != "cpu":
            raise NotImplementedError("lk_ball not supported on non-cpu devices.")

        return dict()

    total_score_components = [
        ScoreComponentClasses(
            "lk_ball_iso", intra_container=LKBallIntraScore, inter_container=None
        ),
        ScoreComponentClasses(
            "lk_ball", intra_container=LKBallIntraScore, inter_container=None
        ),
        ScoreComponentClasses(
            "lk_ball_bridge", intra_container=LKBallIntraScore, inter_container=None
        ),
        ScoreComponentClasses(
            "lk_ball_bridge_uncpl",
            intra_container=LKBallIntraScore,
            inter_container=None,
        ),
    ]

    @reactive_property
    def lkball_waters(
        coords, ljlk_atom_types, indexed_bonds, atom_type_params, ljlk_param_resolver
    ):
        if coords.device.type != "cpu":
            raise NotImplementedError(
                "lk_ball score graph does not support cuda execution."
            )

        return AttachedWaters(
            atom_type_params, ljlk_param_resolver.global_params
        ).apply(coords[0], ljlk_atom_types[0], indexed_bonds)
