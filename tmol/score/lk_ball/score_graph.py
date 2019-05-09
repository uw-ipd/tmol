import torch

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from ..ljlk.score_graph import _LJLKCommonScoreGraph

from .script_modules import LKBallIntraModule
from tmol.score.ljlk.params import LJLKParamResolver, LJLKGlobalParams
from tmol.score.chemical_database import AtomTypeParamResolver


@reactive_attrs
class LKBallIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def total_lj(target):
        return target.lkball_intra_module(
            target.coords[0], target.ljlk_atom_types[0], target.bonded_path_length[0]
        )


@reactive_attrs
class LKBallIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def lkball_score(target):
        return target.lkball_intra_module(
            target.coords[0],
            target.ljlk_atom_types[0],
            target.bonded_path_length[0],
            target.indexed_bonds.bonds,
            target.indexed_bonds.bond_spans,
        )

    @reactive_property
    def total_lk_ball_iso(lkball_score):
        return lkball_score[0]

    @reactive_property
    def total_lk_ball(lkball_score):
        return lkball_score[1]

    @reactive_property
    def total_lk_ball_bridge(lkball_score):
        return lkball_score[2]

    @reactive_property
    def total_lk_ball_bridge_uncpl(lkball_score):
        return lkball_score[3]


@score_graph
class LKBallScoreGraph(_LJLKCommonScoreGraph):
    @staticmethod
    def factory_for(val, device: torch.device, **_):
        """Overridable clone-constructor.
        """
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
    def lkball_intra_module(
        ljlk_param_resolver: LJLKParamResolver, atom_type_params: AtomTypeParamResolver
    ) -> LKBallIntraModule:
        return LKBallIntraModule(ljlk_param_resolver, atom_type_params)
