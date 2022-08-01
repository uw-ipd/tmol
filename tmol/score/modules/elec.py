import attr
from attrs_strict import type_validator
from typing import Set, Type, Optional
import torch
from functools import singledispatch

from tmol.database.scoring.elec import ElecDatabase

from tmol.score.elec.params import ElecParamResolver
from tmol.score.elec.script_modules import ElecIntraModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.bonded_atom import BondedAtoms

from tmol.types.torch import Tensor


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class ElecParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {BondedAtoms, ParamDB, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(
        val, system: ScoreSystem, *, elec_database: Optional[ElecDatabase] = None, **_
    ):
        """Override constructor.

        Create from provided `elec_database``, otherwise from
        ``parameter_database.scoring.elec``.
        """
        if elec_database is None:
            elec_database = ParamDB.get(system).parameter_database.scoring.elec

        return ElecParameters(system=system, elec_database=elec_database)

    elec_database: ElecDatabase = attr.ib(validator=type_validator())
    elec_param_resolver: ElecParamResolver = attr.ib(init=False)
    partial_charges: torch.Tensor = attr.ib(init=False)
    elec_bonded_pair_lengths: torch.Tensor = attr.ib(init=False)

    @elec_param_resolver.default
    def _init_elec_param_resolver(self) -> ElecParamResolver:
        return ElecParamResolver.from_database(
            self.elec_database, TorchDevice.get(self.system).device
        )

    @partial_charges.default
    def _init_partial_charges(self) -> Tensor[torch.float32][:, :]:
        return torch.from_numpy(
            self.elec_param_resolver.resolve_partial_charge(
                BondedAtoms.get(self).res_names, BondedAtoms.get(self).atom_names
            )
        ).to(TorchDevice.get(self).device)

    @elec_bonded_pair_lengths.default
    def _init_elec_bonded_pair_lengths(self) -> Tensor[torch.float32][:, :, :]:
        np_bpl = torch.from_numpy(
            self.elec_param_resolver.remap_bonded_path_lengths(
                BondedAtoms.get(self).bonded_path_length.cpu().numpy(),
                BondedAtoms.get(self).res_names,
                BondedAtoms.get(self).res_indices,
                BondedAtoms.get(self).atom_names,
            )
        )
        # this renders into n-poses x max-n-atoms x max-n-atoms?

        # print("bonded path lengths")
        # print(np_bpl.shape)

        return np_bpl.to(TorchDevice.get(self).device)


@ElecParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old, system: ScoreSystem, *, elec_database: Optional[ElecDatabase] = None, **_
):
    """Override constructor.

        Create from ``val.elec_database`` if possible, otherwise from
        ``parameter_database.scoring.elec``.
        """
    if elec_database is None:
        elec_database = ElecParameters.get(old).elec_database

    return ElecParameters(system=system, elec_database=elec_database)


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class ElecScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {ElecParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "ElecScore":
        return ElecScore(system=system)

    elec_intra_module: ElecIntraModule = attr.ib(init=False)

    @elec_intra_module.default
    def _init_elec_intra_module(self):
        return ElecIntraModule(ElecParameters.get(self).elec_param_resolver)

    def intra_forward(self, coords: torch.Tensor):
        # return a dictionary with a single entry
        # value should be a torch.jit.ScriptModule with a .forward method
        return {
            "elec": self.elec_intra_module(
                coords,
                ElecParameters.get(
                    self
                ).partial_charges,  # store the partial charges on elec parameters
                # and pass it in here instead of the usual atom types.
                # See ElecParamResolver.resolve_partial_charge.
                ElecParameters.get(
                    self
                ).elec_bonded_pair_lengths  # pass this member of elec
                # parameters, different from the usual bonded_path_lengths.
                # See ElecParamResolver.remap_bonded_path_lengths.
            )
        }
