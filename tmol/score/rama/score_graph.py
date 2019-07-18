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
        return target.rama_module(target.coords[0, ...])[0]


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
    allphis: NDArray(int)[:, 5]
    allpsis: NDArray(int)[:, 5]

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
        res_names: NDArray(object)[...],
        rama_param_resolver: RamaParamResolver,
        allphis: NDArray(int)[:, 5],
        allpsis: NDArray(int)[:, 5],
    ) -> RamaParams:
        # find all phi/psis where BOTH are defined
        dfphis = pandas.DataFrame(allphis)
        dfpsis = pandas.DataFrame(allpsis)
        phipsis = dfphis.merge(
            dfpsis, left_on=0, right_on=0, suffixes=("_phi", "_psi")
        ).values[:, 1:]

        # resolve parameter indices
        ramatable_indices = rama_param_resolver.resolve_ramatables(
            res_names[0, phipsis[:, 5]],  # psi atom 'b'
            res_names[0, phipsis[:, 7]],  # psi atom 'd'
        )

        # remove undefined indices and send to device
        rama_defined = numpy.all(phipsis != -1, axis=1)
        phi_indices = torch.from_numpy(phipsis[rama_defined, :4]).to(
            device=rama_param_resolver.device, dtype=torch.int32
        )
        psi_indices = torch.from_numpy(phipsis[rama_defined, 4:]).to(
            device=rama_param_resolver.device, dtype=torch.int32
        )
        param_indices = torch.from_numpy(ramatable_indices[rama_defined]).to(
            device=rama_param_resolver.device, dtype=torch.int32
        )

        return RamaParams(
            phi_indices=phi_indices,
            psi_indices=psi_indices,
            param_indices=param_indices,
        )
