from typing import Set, Type

import attr

from tmol.score.modules.bases import ScoreModule
from tmol.score.modules.database import ParamDB
from tmol.score.modules.device import TorchDevice

from tmol.score.chemical_database import AtomTypeParamResolver


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class ChemicalDB(ScoreModule):
    """Graph component for chemical parameter dispatch."""

    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {TorchDevice, ParamDB}

    @staticmethod
    def build_for(val, system, **_) -> "ChemicalDB":
        return ChemicalDB(system=system)

    atom_type_params: AtomTypeParamResolver = attr.ib(init=False)

    @atom_type_params.default
    def _init_atom_type_params(self) -> AtomTypeParamResolver:
        return AtomTypeParamResolver.from_database(
            ParamDB.get(self).parameter_database.chemical, TorchDevice.get(self).device
        )
