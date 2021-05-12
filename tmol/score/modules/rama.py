import attr
from attrs_strict import type_validator
from collections import namedtuple
from typing import Set, Type, Optional
import torch
import numpy
import pandas
from functools import singledispatch

from tmol.database.scoring import RamaDatabase

from tmol.score.rama.params import RamaParamResolver, RamaParams
from tmol.score.rama.script_modules import RamaScoreModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.bonded_atom import BondedAtoms

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


AllPhisPsis = namedtuple("AllPhisPsis", ["allphis", "allpsis"])


def get_rama_all_phis_psis(system):
    phis = numpy.array(
        [
            [
                [
                    x["residue_index"],
                    x["atom_index_a"],
                    x["atom_index_b"],
                    x["atom_index_c"],
                    x["atom_index_d"],
                ]
                for x in system.torsion_metadata[
                    system.torsion_metadata["name"] == "phi"
                ]
            ]
        ]
    )

    psis = numpy.array(
        [
            [
                [
                    x["residue_index"],
                    x["atom_index_a"],
                    x["atom_index_b"],
                    x["atom_index_c"],
                    x["atom_index_d"],
                ]
                for x in system.torsion_metadata[
                    system.torsion_metadata["name"] == "psi"
                ]
            ]
        ]
    )

    return AllPhisPsis(phis, psis)


def get_rama_all_phis_psis_for_stack(stackedsystem):
    all_phis_psis_list = [
        get_rama_all_phis_psis(system) for system in stackedsystem.systems
    ]

    max_nres = max(
        all_phis_psis.allphis.shape[1] for all_phis_psis in all_phis_psis_list
    )

    def expand(t):
        ext = numpy.full((1, max_nres, 5), -1, dtype=int)
        ext[0, : t.shape[1], :] = t[0]
        return ext

    all_phis_psis_stacked = AllPhisPsis(
        numpy.concatenate(
            [expand(all_phis_psis.allphis) for all_phis_psis in all_phis_psis_list]
        ),
        numpy.concatenate(
            [expand(all_phis_psis.allpsis) for all_phis_psis in all_phis_psis_list]
        ),
    )

    return all_phis_psis_stacked


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class RamaParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {BondedAtoms, ParamDB, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(
        val, system: ScoreSystem, *, rama_database: Optional[RamaDatabase] = None, **_
    ):
        """Override constructor.

        Create from provided `rama_database``, otherwise from
        ``parameter_database.scoring.rama``.
        """
        if rama_database is None:
            rama_database = ParamDB.get(system).parameter_database.scoring.rama

        all_phis_psis = get_rama_all_phis_psis(val)

        return RamaParameters(
            system=system, rama_database=rama_database, rama_all_phis_psis=all_phis_psis
        )

    rama_database: RamaDatabase = attr.ib(validator=type_validator())
    rama_all_phis_psis: AllPhisPsis = attr.ib(validator=type_validator())
    rama_param_resolver: RamaParamResolver = attr.ib(init=False)
    rama_params: RamaParams = attr.ib(init=False)

    @rama_param_resolver.default
    def _init_rama_param_resolver(self) -> RamaParamResolver:
        return RamaParamResolver.from_database(
            self.rama_database, TorchDevice.get(self.system).device
        )

    @rama_params.default
    def _init_rama_params(self) -> RamaParams:
        # find all phi/psis where BOTH are defined
        phi_list = []
        psi_list = []
        param_inds_list = []

        for i in range(self.rama_all_phis_psis.allphis.shape[0]):
            dfphis = pandas.DataFrame(self.rama_all_phis_psis.allphis[i])
            dfpsis = pandas.DataFrame(self.rama_all_phis_psis.allpsis[i])
            phipsis = dfphis.merge(
                dfpsis, left_on=0, right_on=0, suffixes=("_phi", "_psi")
            ).values[:, 1:]

            # resolve parameter indices
            ramatable_indices = self.rama_param_resolver.resolve_ramatables(
                BondedAtoms.get(self).res_names[i, phipsis[:, 5]],  # psi atom 'b'
                BondedAtoms.get(self).res_names[i, phipsis[:, 7]],  # psi atom 'd'
            )

            # remove undefined indices and send to device
            rama_defined = numpy.all(phipsis != -1, axis=1)

            phi_list.append(phipsis[rama_defined, :4])
            psi_list.append(phipsis[rama_defined, 4:])
            param_inds_list.append(ramatable_indices[rama_defined])

        max_size = max(x.shape[0] for x in phi_list)
        phi_inds = torch.full(
            (self.rama_all_phis_psis.allphis.shape[0], max_size, 4),
            -1,
            device=self.rama_param_resolver.device,
            dtype=torch.int32,
        )
        psi_inds = torch.full(
            (self.rama_all_phis_psis.allphis.shape[0], max_size, 4),
            -1,
            device=self.rama_param_resolver.device,
            dtype=torch.int32,
        )
        param_inds = torch.full(
            (self.rama_all_phis_psis.allphis.shape[0], max_size),
            -1,
            device=self.rama_param_resolver.device,
            dtype=torch.int32,
        )

        def copyem(dest, arr, i):
            iarr = arr[i]
            dest[i, : iarr.shape[0]] = torch.tensor(
                iarr, dtype=torch.int32, device=self.rama_param_resolver.device
            )

        for i in range(self.rama_all_phis_psis.allphis.shape[0]):
            copyem(phi_inds, phi_list, i)
            copyem(psi_inds, psi_list, i)
            copyem(param_inds, param_inds_list, i)

        return RamaParams(
            phi_indices=phi_inds, psi_indices=psi_inds, param_indices=param_inds
        )


@RamaParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old, system: ScoreSystem, *, rama_database: Optional[RamaDatabase] = None, **_
):
    """Override constructor.

        Create from ``val.rama_database`` if possible, otherwise from
        ``parameter_database.scoring.rama``.
        """
    if rama_database is None:
        rama_database = RamaParameters.get(old).rama_database

    return RamaParameters(
        system=system,
        rama_database=rama_database,
        rama_all_phis_psis=RamaParameters.get(old).rama_all_phis_psis,
    )


@RamaParameters.build_for.register(PackedResidueSystemStack)
def _build_for_stack(
    stack, system: ScoreSystem, *, rama_database: Optional[RamaDatabase] = None, **_
):
    """Override constructor.

    Create from provided `dunbrack_rotamer_library``, otherwise from
    ``parameter_database.scoring.dun``.
    """
    if rama_database is None:
        rama_database = ParamDB.get(system).parameter_database.scoring.rama

    rama_all_phis_psis = get_rama_all_phis_psis_for_stack(stack)

    return RamaParameters(
        system=system,
        rama_database=rama_database,
        rama_all_phis_psis=rama_all_phis_psis,
    )


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class RamaScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {RamaParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "RamaScore":
        return RamaScore(system=system)

    rama_score_module: RamaScoreModule = attr.ib(init=False)

    @rama_score_module.default
    def _init_rama_score_module(self) -> RamaScoreModule:
        return RamaScoreModule(
            RamaParameters.get(self).rama_params,
            RamaParameters.get(self).rama_param_resolver,
        )

    def intra_forward(self, coords: torch.Tensor):
        return {"rama": self.rama_score_module(coords)}
