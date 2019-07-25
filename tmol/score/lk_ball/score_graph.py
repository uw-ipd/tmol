import torch
import attr

from tmol.utility.reactive import reactive_attrs, reactive_property

from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from ..ljlk.score_graph import _LJLKCommonScoreGraph

from .script_modules import LKBallIntraModule

from tmol.score.ljlk.params import LJLKParamResolver
from tmol.score.chemical_database import AtomTypeParamResolver

from tmol.types.torch import Tensor


@attr.s(auto_attribs=True)
class LKBallPairs:
    polars: Tensor("i8")[:, :]
    occluders: Tensor("i8")[:, :]


@reactive_attrs
class LKBallIntraScore(IntraScore):
    @reactive_property
    # @validate_args
    def lkball_score(target):
        print(target.coords.shape)
        print(target.lkball_pairs.polars.shape)
        print(target.lkball_pairs.occluders.shape)
        print(target.ljlk_atom_types.shape)
        print(target.bonded_path_length.shape)
        print(target.indexed_bonds.bonds.shape)
        print(target.indexed_bonds.bond_spans.shape)

        
        return target.lkball_intra_module(
            target.coords,
            target.lkball_pairs.polars,
            target.lkball_pairs.occluders,
            target.ljlk_atom_types,
            target.bonded_path_length,
            target.indexed_bonds.bonds,
            target.indexed_bonds.bond_spans,
        )

    @reactive_property
    def total_lk_ball_iso(lkball_score):
        return lkball_score[:, 0]

    @reactive_property
    def total_lk_ball(lkball_score):
        return lkball_score[:, 1]

    @reactive_property
    def total_lk_ball_bridge(lkball_score):
        return lkball_score[:, 2]

    @reactive_property
    def total_lk_ball_bridge_uncpl(lkball_score):
        return lkball_score[:, 3]


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

    @reactive_property
    def lkball_pairs(
        ljlk_atom_types: Tensor(torch.int64)[:, :],
        atom_type_params: AtomTypeParamResolver,
        device: torch.device
    ) -> LKBallPairs:
        """Return lists of atoms over which to iterate.
        LK-Ball is only dispatched over polar:heavyatom pairs
        """

        print("ljlk_atom_types.shape",ljlk_atom_types.shape)
        nstacks = ljlk_atom_types.shape[0]

        polars_list = [
            torch.nonzero(
                atom_type_params.params.is_acceptor[ljlk_atom_types[i]]
                + atom_type_params.params.is_donor[ljlk_atom_types[i]]).reshape(-1)
            for i in range(nstacks)]
        occluders_list = [
            torch.nonzero(
                1 - atom_type_params.params.is_hydrogen[ljlk_atom_types[i]]
            ).reshape(-1)
            for i in range(nstacks)]
            
        max_polars = max(len(pols) for pols in polars_list)
        max_occluders = max(len(occs) for occs in occluders_list)
        #print("max_polars", max_polars)
        #print("max_occluders", max_occluders)

        polars = torch.full((nstacks, max_polars), -9999, dtype=torch.int64, device=device)
        occluders = torch.full((nstacks, max_occluders), -9999, dtype=torch.int64, device=device)
        
        for i in range(nstacks):
            polars[i,:polars_list[i].shape[0]] = polars_list[i]
            occluders[i,:occluders_list[i].shape[0]] = occluders_list[i]

        #print("polars", polars)
        #print("occluders", occluders)

        return LKBallPairs(polars=polars, occluders=occluders)
