import attr
from attrs_strict import type_validator
from typing import Set, Type, Optional
import torch
from functools import singledispatch

from tmol.database.scoring import LJLKDatabase

from tmol.score.ljlk.params import LJLKParamResolver
from tmol.score.ljlk.script_modules import LJIntraModule, LKIsotropicIntraModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.chemical_database import ChemicalDB
from tmol.score.modules.stacked_system import StackedSystem
from tmol.score.modules.bonded_atom import BondedAtoms


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class LJLKParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {BondedAtoms, ChemicalDB, ParamDB, StackedSystem, TorchDevice}

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

        return LJLKParameters(system=system, ljlk_database=ljlk_database)

    ljlk_database: LJLKDatabase = attr.ib(validator=type_validator())
    ljlk_param_resolver: LJLKParamResolver = attr.ib(init=False)
    # ljlk_atom_types: Tensor(torch.int64)[:, :] = attr.ib(init=False)
    ljlk_atom_types: torch.Tensor = attr.ib(init=False)

    @ljlk_param_resolver.default
    def _init_ljlk_param_resolver(self) -> LJLKParamResolver:
        # torch.device for param resolver is inherited from chemical db
        return LJLKParamResolver.from_param_resolver(
            ChemicalDB.get(self).atom_type_params, self.ljlk_database
        )

    @ljlk_atom_types.default
    def _init_ljlk_atom_types(self) -> torch.Tensor:
        return self.ljlk_param_resolver.type_idx(BondedAtoms.get(self).atom_types)


@LJLKParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old, system: ScoreSystem, *, ljlk_database: Optional[LJLKDatabase] = None, **_
):
    """Override constructor.

        Create from ``val.ljlk_database`` if possible, otherwise from
        ``parameter_database.scoring.ljlk``.
        """
    if ljlk_database is None:
        ljlk_database = LJLKParameters.get(old).ljlk_database

    return LJLKParameters(system=system, ljlk_database=ljlk_database)


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class LJScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {LJLKParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "LJScore":
        return LJScore(system=system)

    lj_intra_module: LJIntraModule = attr.ib(init=False)

    @lj_intra_module.default
    def _init_lj_intra_module(self):
        return LJIntraModule(LJLKParameters.get(self).ljlk_param_resolver)

    def intra_forward(self, coords: torch.Tensor):
        return {
            "lj": self.lj_intra_module(
                coords,
                LJLKParameters.get(self).ljlk_atom_types,
                BondedAtoms.get(self).bonded_path_length,
            )
        }


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class LKScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {LJLKParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "LKScore":
        return LKScore(system=system)

    lk_intra_module: LKIsotropicIntraModule = attr.ib(init=False)

    @lk_intra_module.default
    def _init_lj_intra_module(self) -> LKIsotropicIntraModule:
        return LKIsotropicIntraModule(LJLKParameters.get(self).ljlk_param_resolver)

    def intra_forward(self, coords: torch.Tensor):
        return {
            "lk": self.lk_intra_module(
                coords,
                LJLKParameters.get(self).ljlk_atom_types,
                BondedAtoms.get(self).bonded_path_length,
            )
        }
