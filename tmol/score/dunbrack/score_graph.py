import attr

import torch
import numpy
from typing import Optional
from functools import singledispatch

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from tmol.database import ParameterDatabase
from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from ..database import ParamDB
from ..chemical_database import ChemicalDB, AtomTypeParamResolver
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from .params import DunbrackParamResolver, DunbrackParams
from .torch_op import DunbrackOp


@reactive_attrs
class DunbrackIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def dun(target):
        return target.dunbrack_op.intra(target.coords[0, ...])

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
        val, device: torch.device, dun_database: DunbrackRotamerLibrary, **_
    ):
        """Overridable clone-constructor.
        """

        return dict(
            dun_database=dun_database,
            device=device,
            dun_phi=torch.tensor(val.dun_phi, dtype=torch.int32, device=device),
            dun_psi=torch.tensor(val.dun_psi, dtype=torch.int32, device=device),
            dun_chi=torch.tensor(val.dun_chi, dtype=torch.int32, device=device),
        )

    dun_database: DunbrackRotamerLibrary
    device: torch.device
    dun_phi: Tensor(torch.int32)[:, 5]  # X by 5; resid, at1, at2, at3, at4
    dun_psi: Tensor(torch.int32)[:, 5]  # X by 5; ibid
    dun_chi: Tensor(torch.int32)[:, 6]  # X by 6; resid, chi_ind, at1, at2, at3, at4

    @reactive_property
    @validate_args
    def dunbrack_op(
        dun_param_resolver: DunbrackParamResolver, dun_resolve_indices: DunbrackParams
    ) -> DunbrackOp:
        return DunbrackOp.from_params(dun_param_resolver.packed_db, dun_resolve_indices)

    @reactive_property
    @validate_args
    def dun_param_resolver(
        dun_database: DunbrackRotamerLibrary, device: torch.device
    ) -> DunbrackParamResolver:
        return DunbrackParamResolver.from_database(dun_database, device)

    @reactive_property
    @validate_args
    def dun_resolve_indices(
        dun_param_resolver: DunbrackParamResolver,
        res_names: NDArray(object)[...],
        dun_phi: Tensor(torch.int32)[:, 5],
        dun_psi: Tensor(torch.int32)[:, 5],
        dun_chi: Tensor(torch.int32)[:, 6],
        device: torch.device,
    ) -> DunbrackParams:
        """Parameter tensor groups and atom-type to parameter resolver."""
        return dun_param_resolver.resolve_dunbrack_parameters(
            res_names[0, dun_phi[:, 2]], dun_phi, dun_psi, dun_chi, device
        )
