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


@reactive_attrs
class ElecIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def elec(target):
        assert target.coords.dim() == 3
        assert target.coords.shape[0] == 1

        assert target.elec_partial_charges.dim() == 2
        assert target.elec_partial_charges.shape[0] == 1

        assert target.bonded_path_length.dim() == 3
        assert target.bonded_path_length.shape[0] == 1

        return target.elec_op.intra(
            target.coords[0],
            target.elec_partial_charges[0],
            target.bonded_path_length[0],
        )


@reactive_attrs(auto_attribs=True)
class ElecScoreGraph(
    BondedAtomScoreGraph, ScoreComponent, ParamDB, TorchDevice, Factory
):
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
    @validate_args
    def elec_partial_charges(
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        elec_param_resolver: ElecParamResolver,
    ) -> Tensor(torch.float32)[:, :]:
        """Pair parameter tensors for all atoms within system."""
        assert atom_types.shape[0] == 1
        atom_types = atom_types[0]
        return torch.from_numpy(
            elec_param_resolver.resolve_partial_charge(res_names, atom_names)[None, :]
        ).to(elec_param_resolver.device)
