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


def condense_inds(selection: Tensor(bool)[:, :], device: torch.device):
    """Given a two dimensional boolean tensor, create
    an output tensor holding the column indices of the non-zero
    entries for each row. Pad out the extra entries
    in any given row that do not correspond to a selected
    entry with a sentinel of -1.
    """

    nstacks = selection.shape[0]
    nz_selection = torch.nonzero(selection)
    nkeep = torch.sum(selection, dim=1).view((nstacks, 1))
    max_keep = torch.max(nkeep)
    inds = torch.full((nstacks, max_keep), -1, dtype=torch.int64, device=device)
    counts = torch.arange(max_keep, dtype=torch.int64, device=device).view(
        (1, max_keep)
    )
    lowinds = counts < nkeep

    inds[lowinds] = nz_selection[:, 1]
    return inds


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
        device: torch.device,
    ) -> LKBallPairs:
        """Return lists of atoms over which to iterate.
        LK-Ball is only dispatched over polar:heavyatom pairs
        """

        are_polars = (
            atom_type_params.params.is_acceptor[ljlk_atom_types]
            + atom_type_params.params.is_donor[ljlk_atom_types]
            > 0
        )
        are_occluders = 1 - atom_type_params.params.is_hydrogen[ljlk_atom_types]

        polars = condense_inds(are_polars, device)
        occluders = condense_inds(are_occluders, device)

        return LKBallPairs(polars=polars, occluders=occluders)
