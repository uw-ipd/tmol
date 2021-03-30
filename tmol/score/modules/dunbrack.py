import attr
from attrs_strict import type_validator
from typing import Set, Type, Optional
import torch
from functools import singledispatch

from tmol.database.scoring import PackedDunbrackDatabase, DunbrackRotamerLibrary

from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.score.dunbrack.script_modules import LJIntraModule, LKIsotropicIntraModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.chemical_database import ChemicalDB
from tmol.score.modules.stacked_system import StackedSystem
from tmol.score.modules.bonded_atom import BondedAtoms

from tmol.types.torch import Tensor


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class dunbrackParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {BondedAtoms, ChemicalDB, ParamDB, StackedSystem, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(
        val,
        system: ScoreSystem,
        *,
        dunbrack_database: Optional[dunbrackDatabase] = None,
        **_,
    ):
        """Override constructor.

        Create from provided `dunbrack_database``, otherwise from
        ``parameter_database.scoring.dunbrack``.
        """
        if dunbrack_database is None:
            dunbrack_database = ParamDB.get(system).parameter_database.scoring.dunbrack

        return dunbrackParameters(system=system, dunbrack_database=dunbrack_database)

    dunbrack_rotamer_library: Du
    dunbrack_database: dunbrackDatabase = attr.ib(validator=type_validator())
    dunbrack_param_resolver: DunbrackParamResolver = attr.ib(init=False)
    # dunbrack_atom_types: Tensor[torch.int64][:, :] = attr.ib(init=False)
    dunbrack_atom_types: torch.Tensor = attr.ib(init=False)

    @dunbrack_param_resolver.default
    def _init_dunbrack_param_resolver(self) -> DunbrackParamResolver:
        # torch.device for param resolver is inherited from chemical db
        return DunbrackParamResolver.from_database(
            dunbrack_rotamer_library, TorchDevice.get(self.system).device
        )

    @dunbrack_atom_types.default
    def _init_dunbrack_atom_types(self) -> Tensor[torch.long][...]:
        return self.dunbrack_param_resolver.type_idx(BondedAtoms.get(self).atom_types)


@dunbrackParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old,
    system: ScoreSystem,
    *,
    dunbrack_database: Optional[dunbrackDatabase] = None,
    **_,
):
    """Override constructor.

        Create from ``val.dunbrack_database`` if possible, otherwise from
        ``parameter_database.scoring.dunbrack``.
        """
    if dunbrack_database is None:
        dunbrack_database = dunbrackParameters.get(old).dunbrack_database

    return dunbrackParameters(system=system, dunbrack_database=dunbrack_database)


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class LJScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {dunbrackParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "LJScore":
        return LJScore(system=system)

    lj_intra_module: LJIntraModule = attr.ib(init=False)

    @lj_intra_module.default
    def _init_lj_intra_module(self):
        return LJIntraModule(dunbrackParameters.get(self).dunbrack_param_resolver)

    def intra_forward(self, coords: torch.Tensor):
        return {
            "lj": self.lj_intra_module(
                coords,
                dunbrackParameters.get(self).dunbrack_atom_types,
                BondedAtoms.get(self).bonded_path_length,
            )
        }


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class LKScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {dunbrackParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "LKScore":
        return LKScore(system=system)

    lk_intra_module: LKIsotropicIntraModule = attr.ib(init=False)

    @lk_intra_module.default
    def _init_lj_intra_module(self) -> LKIsotropicIntraModule:
        return LKIsotropicIntraModule(
            dunbrackParameters.get(self).dunbrack_param_resolver
        )

    def intra_forward(self, coords: torch.Tensor):
        return {
            "lk": self.lk_intra_module(
                coords,
                dunbrackParameters.get(self).dunbrack_atom_types,
                BondedAtoms.get(self).bonded_path_length,
            )
        }
