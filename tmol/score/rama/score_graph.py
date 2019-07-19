import pandas
import numpy

from typing import Optional
from functools import singledispatch

import torch

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from tmol.database import ParameterDatabase
from tmol.database.scoring import RamaDatabase
from .params import RamaParamResolver, RamaParams
from .script_modules import RamaScoreModule

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.array import NDArray


@reactive_attrs
class RamaIntraScore(IntraScore):
    @reactive_property
    def total_rama(target):
        return target.rama_module(target.coords)


@score_graph
class RamaScoreGraph(BondedAtomScoreGraph, ParamDB, TorchDevice):
    # Data member instructing the ScoreComponent class which classes to construct when
    # attempting to evaluate "one body" vs "two body" energies with the Rama term.
    total_score_components = [
        ScoreComponentClasses(
            "rama", intra_container=RamaIntraScore, inter_container=None
        )
    ]

    @staticmethod
    @singledispatch
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        rama_database: Optional[RamaDatabase] = None,
        **_,
    ):
        return dict(
            rama_database=val.rama_database, allphis=val.allphis, allpsis=val.allpsis
        )

    rama_database: RamaDatabase
    allphis: NDArray(int)[:, :, 5]
    allpsis: NDArray(int)[:, :, 5]

    @reactive_property
    @validate_args
    def rama_module(
        rama_param_resolver: RamaParamResolver, rama_resolve_indices: RamaParams
    ) -> RamaScoreModule:
        return RamaScoreModule(rama_resolve_indices, rama_param_resolver)

    @reactive_property
    @validate_args
    def rama_param_resolver(
        rama_database: RamaDatabase, device: torch.device
    ) -> RamaParamResolver:
        "phi/psi resolver"
        return RamaParamResolver.from_database(rama_database, device)

    @reactive_property
    @validate_args
    def rama_resolve_indices(
        res_names: NDArray(object)[:, :],
        rama_param_resolver: RamaParamResolver,
        allphis: NDArray(int)[:, :, 5],
        allpsis: NDArray(int)[:, :, 5],
    ) -> RamaParams:
        # find all phi/psis where BOTH are defined
        phi_list = []
        psi_list = []
        param_inds_list = []

        for i in range(allphis.shape[0]):
            
            dfphis = pandas.DataFrame(allphis[i])
            dfpsis = pandas.DataFrame(allpsis[i])
            phipsis = dfphis.merge(
                dfpsis, left_on=0, right_on=0, suffixes=("_phi", "_psi")
            ).values[:, 1:]

            # resolve parameter indices
            ramatable_indices = rama_param_resolver.resolve_ramatables(
                res_names[i, phipsis[:, 5]],  # psi atom 'b'
                res_names[i, phipsis[:, 7]],  # psi atom 'd'
            )

            # remove undefined indices and send to device
            rama_defined = numpy.all(phipsis != -1, axis=1)
            
            phi_list.append(phipsis[rama_defined, :4])
            psi_list.append(phipsis[rama_defined, 4:])
            param_inds_list.append(ramatable_indices[rama_defined])

        max_size = max(x.shape[0] for x in phi_list)
        phi_inds = torch.full(
            (allphis.shape[0], max_size, 4), -1,
            device=rama_param_resolver.device, dtype=torch.int32)
        psi_inds = torch.full(
            (allphis.shape[0], max_size, 4), -1,
            device=rama_param_resolver.device, dtype=torch.int32)
        param_inds = torch.full(
            (allphis.shape[0], max_size), -1,
            device=rama_param_resolver.device, dtype=torch.int32)

        def copyem(dest, arr, i):
            iarr = arr[i]
            dest[i, :iarr.shape[0]] = torch.tensor(
                iarr, dtype=torch.int32,
                device=rama_param_resolver.device)
        
        for i in range(allphis.shape[0]):
            copyem(phi_inds, phi_list, i)
            copyem(psi_inds, psi_list, i)
            copyem(param_inds, param_inds_list, i)


        return RamaParams(
            phi_indices=phi_inds,
            psi_indices=psi_inds,
            param_indices=param_inds,
        )
