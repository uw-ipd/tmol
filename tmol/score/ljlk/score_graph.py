from typing import Optional

import torch

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from tmol.database import ParameterDatabase
from tmol.database.scoring import LJLKDatabase

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..factory import Factory
from ..score_components import ScoreComponent, ScoreComponentClasses, IntraScore

from .params import LJLKParamResolver
from .torch_op import LJOp


@reactive_attrs
class LJIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def lj(target):
        assert target.coords.dim() == 3
        assert target.coords.shape[0] == 1

        assert target.lj_atom_types.dim() == 2
        assert target.lj_atom_types.shape[0] == 1

        assert target.bonded_path_length.dim() == 3
        assert target.bonded_path_length.shape[0] == 1

        return target.lj_op.intra(
            target.coords[0], target.lj_atom_types[0], target.bonded_path_length[0]
        )

    @reactive_property
    def total_lj(lj):
        """total inter-atomic lj"""
        score_ind, score_val = lj
        return score_val.sum()


@reactive_attrs(auto_attribs=True)
class LJScoreGraph(BondedAtomScoreGraph, ScoreComponent, ParamDB, TorchDevice, Factory):
    total_score_components = [
        ScoreComponentClasses("lj", intra_container=LJIntraScore, inter_container=None)
    ]

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        ljlk_database: Optional[LJLKDatabase] = None,
        **_,
    ):
        """Overridable clone-constructor.

        Initialize from ``val.ljlk_database`` if possible, otherwise from
        ``parameter_database.scoring.ljlk``.
        """
        # Check for disabled tests under "TODO" when enabling cuda.
        assert device.type == "cpu", "Component only supports cpu execution."

        if ljlk_database is None:
            if getattr(val, "ljlk_database", None):
                ljlk_database = val.ljlk_database
            else:
                ljlk_database = parameter_database.scoring.ljlk

        return dict(ljlk_database=ljlk_database)

    ljlk_database: LJLKDatabase

    @reactive_property
    @validate_args
    def param_resolver(
        ljlk_database: LJLKDatabase, device: torch.device
    ) -> LJLKParamResolver:
        """Parameter tensor groups and atom-type to parameter resolver."""
        return LJLKParamResolver.from_database(ljlk_database, device)

    @reactive_property
    @validate_args
    def lj_op(param_resolver: LJLKParamResolver) -> LJOp:
        """LJ evaluation op."""
        return LJOp.from_param_resolver(param_resolver)

    @reactive_property
    @validate_args
    def lj_atom_types(
        atom_types: NDArray(object)[:, :], param_resolver: LJLKParamResolver
    ) -> Tensor(torch.int64)[:, :]:
        """Pair parameter tensors for all atoms within system."""
        assert atom_types.shape[0] == 1
        atom_types = atom_types[0]
        return torch.from_numpy(param_resolver.type_idx(atom_types)[None, :])
