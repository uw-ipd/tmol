import attr
from attrs_strict import type_validator
from collections import namedtuple
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

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


PhiPsiChi = namedtuple("PhiPsiChi", ["phi", "psi", "chi"])


def get_dunbrack_phi_psi_chi(
    system: PackedResidueSystem, device: torch.device
) -> PhiPsiChi:
    dun_phi_raw = numpy.array(
        [
            [
                x["residue_index"],
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "phi"]
        ],
        dtype=numpy.int32,
    )

    dun_psi_raw = numpy.array(
        [
            [
                x["residue_index"],
                x["atom_index_a"],
                x["atom_index_b"],
                x["atom_index_c"],
                x["atom_index_d"],
            ]
            for x in system.torsion_metadata[system.torsion_metadata["name"] == "psi"]
        ],
        dtype=numpy.int32,
    )

    # fd: make a consistent phi/psi array
    resids = numpy.concatenate((dun_phi_raw[:, 0], dun_psi_raw[:, 0]))
    resids = numpy.unique(resids)
    dun_phi = numpy.full((resids.shape[0], 5), -1)
    dun_psi = numpy.full((resids.shape[0], 5), -1)
    resids2idx = {x: i for i, x in enumerate(resids)}
    phi_idx = numpy.vectorize(lambda x: resids2idx[x])(dun_phi_raw[:, 0])
    psi_idx = numpy.vectorize(lambda x: resids2idx[x])(dun_psi_raw[:, 0])
    dun_phi[phi_idx] = dun_phi_raw
    dun_psi[psi_idx] = dun_psi_raw
    dun_phi[:, 0] = resids
    dun_psi[:, 0] = resids

    def nab_chi(chi_name, chi_ind):
        dun_chi = numpy.array(
            [
                [
                    x["residue_index"],
                    chi_ind,
                    x["atom_index_a"],
                    x["atom_index_b"],
                    x["atom_index_c"],
                    x["atom_index_d"],
                ]
                for x in system.torsion_metadata[
                    system.torsion_metadata["name"] == chi_name
                ]
            ],
            dtype=numpy.int32,
        )
        if dun_chi.shape[0] == 0:
            dun_chi = numpy.zeros((0, 6), dtype=numpy.int32)
        return dun_chi

    dun_chi1 = nab_chi("chi1", 0)
    dun_chi2 = nab_chi("chi2", 1)
    dun_chi3 = nab_chi("chi3", 2)
    dun_chi4 = nab_chi("chi4", 3)

    # merge the 4 chi tensors, sorting by residue index and chi index
    join_chi = numpy.concatenate((dun_chi1, dun_chi2, dun_chi3, dun_chi4), 0)
    chi_res = join_chi[:, 0]
    chi_inds = join_chi[:, 1]
    sort_inds = numpy.lexsort((chi_inds, chi_res))
    dun_chi = join_chi[sort_inds, :]

    return PhiPsiChi(
        torch.tensor(dun_phi[None, :], dtype=torch.int32, device=device),
        torch.tensor(dun_psi[None, :], dtype=torch.int32, device=device),
        torch.tensor(dun_chi[None, :], dtype=torch.int32, device=device),
    )


def get_dunbrack_phi_psi_chi_for_stack(
    systemstack: PackedResidueSystemStack, device: torch.device
) -> PhiPsiChi:
    phi_psi_chis = [
        get_dunbrack_phi_psi_chi(sys, device) for sys in systemstack.systems
    ]

    max_nres = max(phi_psi_chi.phi.shape[1] for phi_psi_chi in phi_psi_chis)
    max_nchi = max(phi_psi_chi.chi.shape[1] for phi_psi_chi in phi_psi_chis)

    def expand_dihe(t, max_size):
        ext = torch.full(
            (1, max_size, t.shape[2]), -1, dtype=torch.int32, device=t.device
        )
        ext[0, : t.shape[1], :] = t[0]
        return ext

    phi_psi_chi = PhiPsiChi(
        torch.cat(
            [expand_dihe(phi_psi_chi.phi, max_nres) for phi_psi_chi in phi_psi_chis]
        ),
        torch.cat(
            [expand_dihe(phi_psi_chi.psi, max_nres) for phi_psi_chi in phi_psi_chis]
        ),
        torch.cat(
            [expand_dihe(phi_psi_chi.chi, max_nchi) for phi_psi_chi in phi_psi_chis]
        ),
    )

    return phi_psi_chi


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
        # TEMP!
        # Instead of saying that the criterion for being scored by dunbrack
        # is that at2 is real for your phi torsion, let's say every
        # real residue for this structure is a valid dunbrack residue.

        res_inds = BondedAtoms.get(self).res_indices
        first_for_pose = numpy.ones((res_inds.shape[0], 1), dtype=bool)
        at_is_first_for_res = res_inds[:, :-1] != res_inds[:, 1:]
        at_is_real = res_inds[:, 1:] == res_inds[:, 1:]
        at_is_first_for_res = numpy.logical_and(at_is_first_for_res, at_is_real)
        first_at_for_res = numpy.concatenate(
            (first_for_pose, at_is_first_for_res), axis=1
        )
        n_poses = res_inds.shape[0]
        n_res_for_pose = numpy.sum(first_at_for_res, axis=1, dtype=int)
        max_n_res = numpy.max(n_res_for_pose)
        dun_res_names = numpy.empty((n_poses, max_n_res), dtype=object)
        res_is_good = (
            numpy.arange(max_n_res).reshape(1, max_n_res) < n_res_for_pose[:, None]
        )
        dun_res_names[res_is_good] = res_names[first_at_for_res]

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
            DunbrackParameters.get(self).dunbrack_param_resolver.scoring_db,
            DunbrackParameters.get(self).dunbrack_params,
            DunbrackParameters.get(self).dunbrack_scratch,
        )

    def intra_forward(self, coords: torch.Tensor):
        result = self.dunbrack_score_module(coords)
        return {
            "dunbrack_rot": result[:, 0],
            "dunbrack_rotdev": result[:, 1],
            "dunbrack_semirot": result[:, 2],
        }
