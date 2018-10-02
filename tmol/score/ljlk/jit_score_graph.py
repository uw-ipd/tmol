
from typing import Optional

import math
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
from ..interatomic_distance import BlockedInteratomicDistanceGraph
from .params import LJLKParamResolver
from .torch_op import LJOp


@reactive_attrs
class JitLJIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def lj_pairwise(target) -> Tensor(float)[:, :]:
        assert target.stack_depth == 1

        pscores = target.lj_op.intra(
            target.coords[0],
            target.ljlk_atom_types[0],
            target.ljlk_bonded_path_length[0],
            # Remove block optimization from all compiled kernels
            # for head-to-head comparison of execution times.
            # target.interblock_distance.min_dist[0],
        )

        return pscores

        # split into atr & rep
        # atrE = np.copy(ljE);
        # selector3 = (dists < lj_lk_pair_params["lj_sigma"])
        # atrE[ selector3  ] = -lj_lk_pair_params["lj_wdepth"][ selector3 ]
        # repE = ljE - atrE

        # atrE *= lj_lk_pair_params["weights"]
        # repE *= lj_lk_pair_params["weights"]

    @reactive_property
    def total_lj(lj_pairwise):
        """total inter-atomic lj"""
        return lj_pairwise.sum().reshape(1)


@reactive_attrs(auto_attribs=True)
class JitLJLKScoreGraph(
    BondedAtomScoreGraph,
    BlockedInteratomicDistanceGraph,
    ScoreComponent,
    ParamDB,
    TorchDevice,
    Factory,
):
    total_score_components = [
        ScoreComponentClasses(
            "lj", intra_container=JitLJIntraScore, inter_container=None
        )
    ]

    lj_jit_type: str

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        ljlk_database: Optional[LJLKDatabase] = None,
        lj_jit_type: str = "numba",
        **_,
    ):
        """Overridable clone-constructor.

        Initialize from ``val.ljlk_database`` if possible, otherwise from
        ``parameter_database.scoring.ljlk``.
        """
        if ljlk_database is None:
            if getattr(val, "ljlk_database", None):
                ljlk_database = val.ljlk_database
            else:
                ljlk_database = parameter_database.scoring.ljlk

        return dict(ljlk_database=ljlk_database, lj_jit_type=lj_jit_type)

    ljlk_database: LJLKDatabase

    @reactive_property
    @validate_args
    def ljlk_bonded_path_length(
        bonded_path_length: NDArray("f4")[:, :, :], device: torch.device
    ) -> Tensor(torch.uint8)[:, :, :]:
        """lj&lk interaction weight, bonded cutoff"""

        bpl = torch.from_numpy(bonded_path_length)
        return torch.where(bpl != math.inf, bpl, bpl.new_full((1,), 128)).to(
            device=device, dtype=torch.uint8
        )

    @reactive_property
    @validate_args
    def ljlk_param_resolver(
        ljlk_database: LJLKDatabase, device: torch.device
    ) -> LJLKParamResolver:
        """Parameter tensor groups and atom-type to parameter resolver."""
        return LJLKParamResolver.from_database(ljlk_database, device)

    @reactive_property
    @validate_args
    def ljlk_atom_types(
        atom_types: NDArray(object)[:, :],
        ljlk_param_resolver: LJLKParamResolver,
        device: torch.device,
    ) -> Tensor(torch.int64)[:, :]:
        """lj&lk interaction weight, bonded cutoff"""

        type_idx = ljlk_param_resolver.type_idx(atom_types)
        type_idx[atom_types == None] = -1  # noqa
        return torch.tensor(type_idx).to(device=device, dtype=torch.int64)

    @reactive_property
    @validate_args
    def lj_op(ljlk_param_resolver: LJLKParamResolver, lj_jit_type: str) -> LJOp:
        """Parameter tensor groups and atom-type to parameter resolver."""
        op = LJOp.from_params(
            ljlk_param_resolver, parallel_cpu=False, jit_type=lj_jit_type
        )
        return op
