import torch
import numpy

from functools import singledispatch

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

from tmol.types.torch import Tensor
from tmol.types.array import NDArray

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from .params import DunbrackParamResolver, DunbrackParams, DunbrackScratch
from .script_modules import DunbrackScoreModule


@reactive_attrs
class DunbrackIntraScore(IntraScore):
    @reactive_property
    # @validate_args
    def dun_score(target):
        return target.dun_module(target.coords)

    @reactive_property
    def total_dun_rot(dun_score):
        return dun_score[:, 0]

    @reactive_property
    def total_dun_dev(dun_score):
        return dun_score[:, 1]

    @reactive_property
    def total_dun_semi(dun_score):
        return dun_score[:, 2]


@score_graph
class DunbrackScoreGraph(BondedAtomScoreGraph, ParamDB, TorchDevice):
    total_score_components = [
        ScoreComponentClasses(
            "dun_rot", intra_container=DunbrackIntraScore, inter_container=None
        ),
        ScoreComponentClasses(
            "dun_dev", intra_container=DunbrackIntraScore, inter_container=None
        ),
        ScoreComponentClasses(
            "dun_semi", intra_container=DunbrackIntraScore, inter_container=None
        ),
    ]

    @staticmethod
    @singledispatch
    def factory_for(
        val, device: torch.device, dun_database: DunbrackRotamerLibrary, **_
    ):
        """Overridable clone-constructor.
        """

        return dict(
            dun_database=dun_database,
            device=device,
            dun_phi=torch.tensor(val.dun_phi, dtype=torch.int32, device=device),
            dun_psi=torch.tensor(val.dun_psi, dtype=torch.int32, device=device),
            dun_chi=torch.tensor(val.dun_chi, dtype=torch.int32, device=device),
        )

    dun_database: DunbrackRotamerLibrary
    device: torch.device
    dun_phi: Tensor[torch.int32][:, :, 5]  # X by 5; resid, at1, at2, at3, at4
    dun_psi: Tensor[torch.int32][:, :, 5]  # X by 5; ibid
    dun_chi: Tensor[torch.int32][:, :, 6]  # X by 6; resid, chi_ind, at1, at2, at3, at4

    @reactive_property
    @validate_args
    def dun_module(
        dun_param_resolver: DunbrackParamResolver,
        dun_resolve_indices: DunbrackParams,
        dun_scratch: DunbrackScratch,
    ) -> DunbrackScoreModule:
        return DunbrackScoreModule(
            dun_param_resolver.scoring_db, dun_resolve_indices, dun_scratch
        )

    @reactive_property
    @validate_args
    def dun_param_resolver(
        dun_database: DunbrackRotamerLibrary, device: torch.device
    ) -> DunbrackParamResolver:
        return DunbrackParamResolver.from_database(dun_database, device)

    @reactive_property
    @validate_args
    def dun_resolve_indices(
        dun_param_resolver: DunbrackParamResolver,
        res_names: NDArray[object][...],
        dun_phi: Tensor[torch.int32][:, :, 5],
        dun_psi: Tensor[torch.int32][:, :, 5],
        dun_chi: Tensor[torch.int32][:, :, 6],
        device: torch.device,
    ) -> DunbrackParams:
        """Parameter tensor groups and atom-type to parameter resolver."""
        dun_res_names = numpy.full(
            (dun_phi.shape[0], dun_phi.shape[1]), None, dtype=object
        )

        # select the name for each residue that potentially qualifies
        # for dunbrack scoring by using the 2nd atom that defines the
        # phi torsion. This atom will be non-negative even if other
        # atoms that define phi are negative.
        dun_at2_inds = dun_phi[:, :, 2].cpu().numpy()
        dun_at2_real = dun_at2_inds != -1
        nz_at2_real = numpy.nonzero(dun_at2_real)
        dun_res_names[dun_at2_real] = res_names[
            nz_at2_real[0], dun_at2_inds[dun_at2_real]
        ]

        return dun_param_resolver.resolve_dunbrack_parameters(
            dun_res_names, dun_phi, dun_psi, dun_chi, device
        )

    @reactive_property
    @validate_args
    def dun_scratch(
        dun_param_resolver: DunbrackParamResolver, dun_resolve_indices: DunbrackParams
    ) -> DunbrackScratch:
        return dun_param_resolver.allocate_dunbrack_scratch_space(dun_resolve_indices)
