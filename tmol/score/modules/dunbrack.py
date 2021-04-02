import attr
from attrs_strict import type_validator
from typing import Set, Type, Optional
import torch
import numpy
from functools import singledispatch

from tmol.database.scoring import DunbrackRotamerLibrary

from tmol.score.dunbrack.params import (
    DunbrackParamResolver,
    DunbrackParams,
    DunbrackScratch,
)
from tmol.score.dunbrack.script_modules import DunbrackScoreModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.bonded_atom import BondedAtoms

from tmol.system.score_support import (
    get_dunbrack_phi_psi_chi,
    get_dunbrack_phi_psi_chi_for_stack,
    PhiPsiChi,
)
from tmol.system.packed import PackedResidueSystemStack


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class DunbrackParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {ParamDB, BondedAtoms, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(
        val,
        system: ScoreSystem,
        *,
        dunbrack_rotamer_library: Optional[DunbrackRotamerLibrary] = None,
        **_,
    ):
        """Override constructor.

        Create from provided `dunbrack_rotamer_library``, otherwise from
        ``parameter_database.scoring.dun``.
        """
        if dunbrack_rotamer_library is None:
            dunbrack_rotamer_library = ParamDB.get(
                system
            ).parameter_database.scoring.dun

        dunbrack_phi_psi_chi = get_dunbrack_phi_psi_chi(
            val, TorchDevice.get(system).device
        )

        return DunbrackParameters(
            system=system,
            dunbrack_rotamer_library=dunbrack_rotamer_library,
            dunbrack_phi_psi_chi=dunbrack_phi_psi_chi,
        )

    dunbrack_rotamer_library: DunbrackRotamerLibrary = attr.ib(
        validator=type_validator()
    )
    dunbrack_phi_psi_chi: PhiPsiChi = attr.ib(validator=type_validator())
    dunbrack_param_resolver: DunbrackParamResolver = attr.ib(init=False)
    dunbrack_params: DunbrackParams = attr.ib(init=False)
    dunbrack_scratch: DunbrackScratch = attr.ib(init=False)

    @dunbrack_param_resolver.default
    def _init_dunbrack_param_resolver(self) -> DunbrackParamResolver:
        return DunbrackParamResolver.from_database(
            self.dunbrack_rotamer_library, TorchDevice.get(self.system).device
        )

    @dunbrack_params.default
    def _init_dunbrack_params(self) -> DunbrackParams:
        dun_phi = self.dunbrack_phi_psi_chi.phi
        dun_psi = self.dunbrack_phi_psi_chi.psi
        dun_chi = self.dunbrack_phi_psi_chi.chi
        """Parameter tensor groups and atom-type to parameter resolver."""
        dun_res_names = numpy.full(
            (dun_phi.shape[0], dun_phi.shape[1]), None, dtype=object
        )

        # select the name for each residue that potentially qualifies
        # for dunbrack scoring by using the 2nd atom that defines the
        # phi torsion. This atom will be non-negative even if other
        # atoms that define phi are negative.
        res_names = BondedAtoms.get(self).res_names
        dun_at2_inds = dun_phi[:, :, 2].cpu().numpy()
        dun_at2_real = dun_at2_inds != -1
        nz_at2_real = numpy.nonzero(dun_at2_real)
        dun_res_names[dun_at2_real] = res_names[
            nz_at2_real[0], dun_at2_inds[dun_at2_real]
        ]

        return self.dunbrack_param_resolver.resolve_dunbrack_parameters(
            dun_res_names,
            dun_phi,  # stack this always
            dun_psi,  # stack this always
            dun_chi,  # stack this always
            TorchDevice.get(self.system).device,
        )

    @dunbrack_scratch.default
    def _init_dun_scratch(self) -> DunbrackScratch:
        return self.dunbrack_param_resolver.allocate_dunbrack_scratch_space(
            self.dunbrack_params
        )


@DunbrackParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old,
    system: ScoreSystem,
    *,
    dunbrack_rotamer_library: Optional[DunbrackRotamerLibrary] = None,
    **_,
):
    """Override constructor.

        Create from ``val.dunbrack_rotamer_library`` if possible, otherwise from
        ``parameter_database.scoring.dunbrack``.
        """
    if dunbrack_rotamer_library is None:
        dunbrack_rotamer_library = DunbrackParameters.get(old).dunbrack_rotamer_library

    return DunbrackParameters(
        system=system,
        dunbrack_rotamer_library=dunbrack_rotamer_library,
        dunbrack_phi_psi_chi=DunbrackParameters.get(old).dunbrack_phi_psi_chi,
    )


@DunbrackParameters.build_for.register(PackedResidueSystemStack)
def _build_for_stack(
    stack,
    system: ScoreSystem,
    *,
    dunbrack_rotamer_library: Optional[DunbrackRotamerLibrary] = None,
    **_,
):
    """Override constructor.

    Create from provided `dunbrack_rotamer_library``, otherwise from
    ``parameter_database.scoring.dun``.
    """
    if dunbrack_rotamer_library is None:
        dunbrack_rotamer_library = ParamDB.get(system).parameter_database.scoring.dun

    dunbrack_phi_psi_chi = get_dunbrack_phi_psi_chi_for_stack(
        stack, TorchDevice.get(system).device
    )

    return DunbrackParameters(
        system=system,
        dunbrack_rotamer_library=dunbrack_rotamer_library,
        dunbrack_phi_psi_chi=dunbrack_phi_psi_chi,
    )


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class DunbrackScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {DunbrackParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "DunbrackScore":
        return DunbrackScore(system=system)

    dunbrack_score_module: DunbrackScoreModule = attr.ib(init=False)

    @dunbrack_score_module.default
    def _init_dunbrack_score_module(self):
        return DunbrackScoreModule(
            DunbrackParameters.get(self).dunbrack_param_resolver.packed_db,
            DunbrackParameters.get(self).dunbrack_params,
            DunbrackParameters.get(self).dunbrack_scratch,
        )

    def intra_forward(self, coords: torch.Tensor):
        return {"dunbrack": self.dunbrack_score_module(coords)}
