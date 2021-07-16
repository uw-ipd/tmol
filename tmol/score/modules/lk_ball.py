import attr
from attrs_strict import type_validator
from typing import Set, Type, Optional
import torch
from functools import singledispatch

from tmol.database.scoring import LJLKDatabase

from tmol.score.lk_ball.script_modules import LKBallIntraModule
from tmol.score.ljlk.params import LJLKParamResolver
from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.chemical_database import ChemicalDB
from tmol.score.modules.bonded_atom import BondedAtoms
from tmol.score.common.stack_condense import condense_torch_inds

from tmol.types.torch import Tensor


@attr.s(auto_attribs=True)
class LKBallPairs:
    polars: Tensor[torch.long][:, :]
    occluders: Tensor[torch.long][:, :]


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class LKBallParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {BondedAtoms, ParamDB, ChemicalDB, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(
        val, system: ScoreSystem, *, ljlk_database: Optional[LJLKDatabase] = None, **_
    ):
        """Override constructor.

        Create from provided `ljlk_database``, otherwise from
        ``parameter_database.scoring.ljlk``.
        """
        if ljlk_database is None:
            ljlk_database = ParamDB.get(system).parameter_database.scoring.ljlk

        return LKBallParameters(system=system, ljlk_database=ljlk_database)

    ljlk_database: LJLKDatabase = attr.ib(validator=type_validator())
    ljlk_param_resolver: LJLKParamResolver = attr.ib(init=False)
    ljlk_atom_types: torch.Tensor = attr.ib(init=False)
    lkball_pairs: LKBallPairs = attr.ib(init=False)

    @ljlk_atom_types.default
    def _init_ljlk_atom_types(self) -> Tensor[torch.long][...]:
        return self.ljlk_param_resolver.type_idx(BondedAtoms.get(self).atom_types)

    @ljlk_param_resolver.default
    def _init_ljlk_param_resolver(self) -> LJLKParamResolver:
        return LJLKParamResolver.from_database(
            ParamDB.get(self).parameter_database.chemical,
            self.ljlk_database,
            TorchDevice.get(self.system).device,
        )

    @lkball_pairs.default
    def _init_lkball_pairs(self) -> LKBallPairs:
        """Return lists of atoms over which to iterate.
        LK-Ball is only dispatched over polar:heavyatom pairs
        """

        are_polars = (
            ChemicalDB.get(self).atom_type_params.params.is_acceptor[
                self.ljlk_atom_types
            ]
            + ChemicalDB.get(self).atom_type_params.params.is_donor[
                self.ljlk_atom_types
            ]
            > 0
        )
        are_occluders = ~ChemicalDB.get(self).atom_type_params.params.is_hydrogen[
            self.ljlk_atom_types
        ]

        polars = condense_torch_inds(are_polars, TorchDevice.get(self.system).device)
        occluders = condense_torch_inds(
            are_occluders, TorchDevice.get(self.system).device
        )

        return LKBallPairs(polars=polars, occluders=occluders)


@LKBallParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old, system: ScoreSystem, *, ljlk_database: Optional[LJLKDatabase] = None, **_
):
    """Override constructor.

        Create from ``val.ljlk_database`` if possible, otherwise from
        ``parameter_database.scoring.ljlk``.
        """
    if ljlk_database is None:
        ljlk_database = LKBallParameters.get(old).ljlk_database

    return LKBallParameters(system=system, ljlk_database=ljlk_database)


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class LKBallScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {LKBallParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "LKBallScore":
        return LKBallScore(system=system)

    lk_ball_intra_module: LKBallIntraModule = attr.ib(init=False)

    @lk_ball_intra_module.default
    def _init_lk_ball_intra_module(self):
        return LKBallIntraModule(
            LKBallParameters.get(self).ljlk_param_resolver,
            ChemicalDB.get(self).atom_type_params,
        )

    def intra_forward(self, coords: torch.Tensor):
        result = self.lk_ball_intra_module(
            coords,
            LKBallParameters.get(self).lkball_pairs.polars,
            LKBallParameters.get(self).lkball_pairs.occluders,
            LKBallParameters.get(self).ljlk_atom_types,
            BondedAtoms.get(self).bonded_path_length,
            BondedAtoms.get(self).indexed_bonds.bonds,
            BondedAtoms.get(self).indexed_bonds.bond_spans,
        )
        return {
            "lk_ball_iso": result[:, 0],
            "lk_ball": result[:, 1],
            "lk_ball_bridge": result[:, 2],
            "lk_ball_bridge_uncpl": result[:, 3],
        }
