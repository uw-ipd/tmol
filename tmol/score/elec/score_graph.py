from typing import Optional

import torch

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from tmol.database import ParameterDatabase
from tmol.database.scoring.elec import ElecDatabase

from tmol.types.functional import validate_args

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from .params import ElecParamResolver
from .script_modules import ElecIntraModule


@reactive_attrs
class ElecIntraScore(IntraScore):
    @reactive_property
    # @validate_args
    def total_elec(target):
        V = target.elec_intra_module(
            target.coords, target.elec_partial_charges, target.repatm_bonded_path_length
        )
        return V[:,0]


@score_graph
class ElecScoreGraph(BondedAtomScoreGraph, ParamDB, TorchDevice):
    total_score_components = [
        ScoreComponentClasses(
            "elec", intra_container=ElecIntraScore, inter_container=None
        )
    ]

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        elec_database: Optional[ElecDatabase] = None,
        **_,
    ):
        """Overridable clone-constructor.
        Initialize from ``val.elec_database`` if possible, otherwise from
        ``parameter_database.scoring.ljlk``.
        """
        if elec_database is None:
            if getattr(val, "elec_database", None):
                elec_database = val.elec_database
            else:
                elec_database = parameter_database.scoring.elec

        return dict(elec_database=elec_database)

    elec_database: ElecDatabase

    @reactive_property
    def elec_intra_module(elec_param_resolver: ElecParamResolver) -> ElecIntraModule:
        return ElecIntraModule(elec_param_resolver)

    @reactive_property
    def elec_param_resolver(
        elec_database: ElecDatabase, device: torch.device
    ) -> ElecParamResolver:
        return ElecParamResolver.from_database(elec_database, device)

    # bonded path lengths using 'representative atoms'
    @reactive_property
    # @validate_args
    def repatm_bonded_path_length(
        bonded_path_length: Tensor("f4")[:, :, :],
        res_names: NDArray(object)[:, :],
        res_indices: NDArray(float)[:, :],
        atom_names: NDArray(object)[:, :],
        elec_param_resolver: ElecParamResolver,
    ) -> Tensor(torch.float32)[:, :, :]:
        bpl = bonded_path_length.cpu().numpy()
        return torch.from_numpy(
            elec_param_resolver.remap_bonded_path_lengths(
                bpl, res_names, res_indices, atom_names
            )
        ).to(elec_param_resolver.device)

    @reactive_property
    @validate_args
    def elec_partial_charges(
        res_names: NDArray(object)[:, :],
        atom_names: NDArray(object)[:, :],
        elec_param_resolver: ElecParamResolver,
    ) -> Tensor(torch.float32)[:, :]:
        """Pair parameter tensors for all atoms within system."""
        return torch.from_numpy(
            elec_param_resolver.resolve_partial_charge(res_names, atom_names)
        ).to(elec_param_resolver.device)
