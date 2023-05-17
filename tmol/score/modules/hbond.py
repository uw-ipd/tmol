import attr
from attrs_strict import type_validator
from typing import Set, Type, Optional
import torch
from functools import singledispatch

from tmol.database.scoring import HBondDatabase

from tmol.score.hbond.identification import HBondElementAnalysis
from tmol.score.hbond.params import HBondParamResolver, CompactedHBondDatabase
from tmol.score.hbond.script_modules import HBondIntraModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.bonded_atom import BondedAtoms

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup


@attr.s(auto_attribs=True)
class HBondDonorIndices(TensorGroup):
    D: Tensor[torch.long][..., 3]
    H: Tensor[torch.long][..., 3]
    donor_type: Tensor[torch.int][...]


@attr.s(auto_attribs=True)
class HBondAcceptorIndices(TensorGroup):
    A: Tensor[torch.long][..., 3]
    B: Tensor[torch.long][..., 3]
    B0: Tensor[torch.long][..., 3]
    acceptor_type: Tensor[torch.int][...]


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class HBondParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {BondedAtoms, ParamDB, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(
        val, system: ScoreSystem, *, hbond_database: Optional[HBondDatabase] = None, **_
    ):
        """Override constructor.

        Create from provided `hbond_database``, otherwise from
        ``parameter_database.scoring.hbond``.
        """
        if hbond_database is None:
            hbond_database = ParamDB.get(system).parameter_database.scoring.hbond

        return HBondParameters(system=system, hbond_database=hbond_database)

    hbond_database: HBondDatabase = attr.ib(validator=type_validator())
    hbond_param_resolver: HBondParamResolver = attr.ib(init=False)
    hbond_elements: HBondElementAnalysis = attr.ib(init=False)
    hbond_donor_indices: HBondDonorIndices = attr.ib(init=False)
    hbond_acceptor_indices: HBondAcceptorIndices = attr.ib(init=False)
    compacted_hbond_database: CompactedHBondDatabase = attr.ib(init=False)

    @hbond_param_resolver.default
    def _init_hbond_param_resolver(self) -> HBondParamResolver:
        return HBondParamResolver.from_database(
            ParamDB.get(self).parameter_database.chemical,
            self.hbond_database,
            TorchDevice.get(self.system).device,
        )

    @hbond_elements.default
    def _init_hbond_elements(self):
        return HBondElementAnalysis.setup_from_database(
            ParamDB.get(self).parameter_database.chemical,
            hbond_database=self.hbond_database,
            atom_types=BondedAtoms.get(self).atom_types,
            bonds=BondedAtoms.get(self).bonds,
        )

    @hbond_donor_indices.default
    def _init_hbond_donor_indices(self) -> HBondDonorIndices:
        """hbond donor indicies and type indicies."""

        donor_type = self.hbond_param_resolver.resolve_donor_type(
            self.hbond_elements.donors["donor_type"]
        ).to(torch.int32)
        D = torch.from_numpy(self.hbond_elements.donors["d"]).to(
            device=donor_type.device
        )
        H = torch.from_numpy(self.hbond_elements.donors["h"]).to(
            device=donor_type.device
        )

        return HBondDonorIndices(D=D, H=H, donor_type=donor_type)

    @hbond_acceptor_indices.default
    def _init_hbond_acceptor_indices(self) -> HBondAcceptorIndices:
        """hbond acceptor indicies and type indicies."""

        acceptor_type = self.hbond_param_resolver.resolve_acceptor_type(
            self.hbond_elements.acceptors["acceptor_type"]
        ).to(torch.int32)
        A = torch.from_numpy(self.hbond_elements.acceptors["a"]).to(
            device=acceptor_type.device
        )
        B = torch.from_numpy(self.hbond_elements.acceptors["b"]).to(
            device=acceptor_type.device
        )
        B0 = torch.from_numpy(self.hbond_elements.acceptors["b0"]).to(
            device=acceptor_type.device
        )

        return HBondAcceptorIndices(A=A, B=B, B0=B0, acceptor_type=acceptor_type)

    @compacted_hbond_database.default
    def _init_compacted_hbond_database(self) -> CompactedHBondDatabase:
        "two-tensor representation of hbond parameters on the device"
        return CompactedHBondDatabase.from_database(
            ParamDB.get(self).parameter_database.chemical,
            self.hbond_database,
            TorchDevice.get(self.system).device,
        )


@HBondParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old, system: ScoreSystem, *, hbond_database: Optional[HBondDatabase] = None, **_
):
    """Override constructor.

    Create from ``val.hbond_database`` if possible, otherwise from
    ``parameter_database.scoring.hbond``.
    """
    if hbond_database is None:
        hbond_database = HBondParameters.get(old).hbond_database

    return HBondParameters(system=system, hbond_database=hbond_database)


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class HBondScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {HBondParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "HBondScore":
        return HBondScore(system=system)

    hbond_intra_module: HBondIntraModule = attr.ib(init=False)

    @hbond_intra_module.default
    def _init_hbond_intra_module(self):
        return HBondIntraModule(HBondParameters.get(self).compacted_hbond_database)

    def intra_forward(self, coords: torch.Tensor):
        result = self.hbond_intra_module(
            coords,
            coords,
            HBondParameters.get(self).hbond_donor_indices.D,
            HBondParameters.get(self).hbond_donor_indices.H,
            HBondParameters.get(self).hbond_donor_indices.donor_type,
            HBondParameters.get(self).hbond_acceptor_indices.A,
            HBondParameters.get(self).hbond_acceptor_indices.B,
            HBondParameters.get(self).hbond_acceptor_indices.B0,
            HBondParameters.get(self).hbond_acceptor_indices.acceptor_type,
        )
        return {"hbond": result}
