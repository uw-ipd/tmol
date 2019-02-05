from typing import Optional

import torch

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from tmol.database import ParameterDatabase
from tmol.database.scoring.elec import ElecDatabase

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..factory import Factory
from ..score_components import ScoreComponent, ScoreComponentClasses, IntraScore

from .params import ElecParamResolver
from .torch_op import ElecOp


@reactive_attrs
class ElecIntraScore(IntraScore):
    @reactive_property
    def elec(target):
        assert target.coords.dim() == 3
        assert target.coords.shape[0] == 1

        assert target.elec_partial_charges.dim() == 2
        assert target.elec_partial_charges.shape[0] == 1

        assert target.repatm_bonded_path_length.dim() == 3
        assert target.repatm_bonded_path_length.shape[0] == 1

        return target.elec_op.intra(
            target.coords[0],
            target.elec_partial_charges[0],
            target.repatm_bonded_path_length[0],
        )

    @reactive_property
    def total_elec(elec):
        inds, vals = elec
        return vals.sum()


@reactive_attrs(auto_attribs=True)
class ElecScoreGraph(
    BondedAtomScoreGraph, ScoreComponent, ParamDB, TorchDevice, Factory
):
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
    def elec_op(elec_param_resolver: ElecParamResolver) -> ElecOp:
        """elec evaluation op."""
        return ElecOp.from_param_resolver(elec_param_resolver)

    @reactive_property
    def elec_param_resolver(
        elec_database: ElecDatabase, device: torch.device
    ) -> ElecParamResolver:
        return ElecParamResolver.from_database(elec_database, device)

    # bonded path lengths using 'representative atoms'
    @reactive_property
    def repatm_bonded_path_length(
        bonded_path_length: NDArray(object)[...],
        res_names: NDArray(object)[...],
        res_indices: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        elec_param_resolver: ElecParamResolver,
    ) -> Tensor(torch.float32)[:, :]:
        return torch.from_numpy(
            elec_param_resolver.remap_bonded_path_lengths(
                bonded_path_length.numpy(), res_names, res_indices, atom_names
            )
        ).to(elec_param_resolver.device)

    @reactive_property
    def elec_partial_charges(
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        elec_param_resolver: ElecParamResolver,
    ) -> Tensor(torch.float32)[:, :]:
        """Pair parameter tensors for all atoms within system."""
        return torch.from_numpy(
            elec_param_resolver.resolve_partial_charge(res_names, atom_names)
        ).to(elec_param_resolver.device)
