from typing import Optional

import torch

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from ..database import ParamDB
from ..chemical_database import ChemicalDB, AtomTypeParamResolver
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from .params import LJLKParamResolver
from .torch_op import LJOp, LKOp


@attr.s(auto_attribs=True)
class DunbrackParams(TensorGroup):
    ndihe_for_res: Tensor(torch.int32)[:]
    dihedral_indices: Tensor(torch.int32)[..., 4]
    dihedral_offsets: Tensor(torch.int32)[:]
    rottable_set_for_res: Tensor(torch.int32)[:]
    nchi_for_res: Tensor(torch.int32)[:]
    nrotameric_chi_for_res: Tensor(torch.int32)[:]  # ??needed??
    rotres2resid: Tensor(torch.int32)[:]
    prob_table_offset_for_rotresidue: Tensor(torch.int32)[:]
    rotmean_table_offset_for_residue: Tensor(torch.int32)[:]
    rotind2tableind_offset_for_res: Tensor(torch.int32)[:]
    rotameric_chi_desc: Tensor(torch.int32)[:, 2]
    semirotameric_chi_desc: Tensor(torch.int32)[:, 4]


@reactive_attrs
class DunbrackIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def dun(target):
        return target.dunbrack_op.intra(
            target.coords[0, ...],
            target.resolve_indices.ndihe_for_res,
            target.resolve_indices.dihedral_indices,
            target.resolve_indices.dihedral_offsets,
            target.resolve_indices.rottable_set_for_res,
            target.resolve_indices.nchi_for_res,
            target.resolve_indices.nrotameric_chi_for_res,
            target.resolve_indices.rotres2resid,
            target.resolve_indices.prob_table_offset_for_rotresidue,
            target.resolve_indices.rotmean_table_offset_for_residue,
            target.resolve_indices.rotind2tableind_offset_for_res,
            target.resolve_indices.rotameric_chi_desc,
            target.resolve_indices.semirotameric_chi_desc,
        )

    @reactive_property
    def total_dun(dun):
        """total inter-atomic lj"""
        score_ind, score_val = dun
        return score_val.sum()


@score_graph
class DunbrackScoreGraph(BondedAtomScoreGraph, ParamDB, TorchDevice):
    total_score_components = [
        ScoreComponentClasses(
            "dun", intra_container=DunbrackIntraScore, inter_container=None
        )
    ]

    @staticmethod
    @singledispatch
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        dun_database: DunbrackRotamerLibrary,
        **_,
    ):
        """Overridable clone-constructor.
        """

        return dict(
            dun_database=dun_database,
            device=device,
            phis=val.phis,
            psis=val.psis,
            chis=val.chis,
        )

    dun_database: DunbrackRotamerLibrary
    phis: NDArray
    psis: NDArray
    chis: NDArray

    @reactive_property
    @validate_args
    def dun_resolve_indices(
        res_names: NDArray(object)[:], dun_param_resolver: DunbrackParamResolver
    ) -> DunbrackParams:
        """Parameter tensor groups and atom-type to parameter resolver."""
        pass
