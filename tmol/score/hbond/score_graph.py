import attr
from typing import Optional

import torch
import numpy

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from .identification import HBondElementAnalysis
from .params import HBondParamResolver

# from .torch_op import HBondOp
from .script_modules import HBondIntraModule

from tmol.database import ParameterDatabase
from tmol.database.scoring import HBondDatabase

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.utility.nvtx import nvtx_range


@attr.s(auto_attribs=True)
class HBondDonorIndices(TensorGroup):
    D: Tensor("i8")[..., 3]
    H: Tensor("i8")[..., 3]
    donor_type: Tensor("i4")[...]


@attr.s(auto_attribs=True)
class HBondAcceptorIndices(TensorGroup):
    A: Tensor("i8")[..., 3]
    B: Tensor("i8")[..., 3]
    B0: Tensor("i8")[..., 3]
    acceptor_type: Tensor("i4")[...]


@attr.s(auto_attribs=True)
class HBondDescr(TensorGroup):
    donor: HBondDonorIndices
    acceptor: HBondAcceptorIndices
    score: Tensor("f")[...]


@reactive_attrs
class HBondIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def total_hbond(target):
        """total hbond score"""
        with nvtx_range("HBondIntraScore.hbond"):
            with nvtx_range("HBondIntraScore.hbond.coords"):
                return hbond_score[0]

    @reactive_property
    @validate_args
    def hbond_score(target):
        with nvtx_range("HBondIntraScore.hbond.coords"):
            coords = target.coords[0]
            return target.hbond_intra_module(
                coords,
                coords,
                target.hbond_donor_indices.D,
                target.hbond_donor_indices.H,
                target.hbond_donor_indices.donor_type,
                target.hbond_acceptor_indices.A,
                target.hbond_acceptor_indices.B,
                target.hbond_acceptor_indices.B0,
                target.hbond_acceptor_indices.acceptor_type,
            )

    @reactive_property
    @validate_args
    def hbond_descr(target, hbond) -> HBondDescr:
        """All hbond pairs, in order of "sp2"/"sp3"/"ring"."""
        (di, ai), score = hbond
        return HBondDescr(
            donor=target.hbond_donor_indices[di],
            acceptor=target.hbond_acceptor_indices[ai],
            score=score,
        )


@score_graph
class HBondScoreGraph(BondedAtomScoreGraph, ParamDB, TorchDevice):
    """Compute graph for the HBond term.

    Uses the reactive system to compute the list of donors and acceptors
    (via the HBondElementAnalysis class) and then reuses these lists.
    """

    total_score_components = [
        ScoreComponentClasses(
            "hbond", intra_container=HBondIntraScore, inter_container=None
        )
    ]

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        hbond_database: Optional[HBondDatabase] = None,
        **_,
    ):
        """Overridable clone-constructor.

        Initialize from ``val.hbond_database`` if possible, otherwise from
        ``parameter_database.scoring.hbond``.
        """

        if hbond_database is None:
            if getattr(val, "hbond_database", None):
                hbond_database = val.hbond_database
            else:
                hbond_database = parameter_database.scoring.hbond

        return dict(hbond_database=hbond_database)

    hbond_database: HBondDatabase

    @reactive_property
    @validate_args
    def hbond_param_resolver(
        parameter_database: ParameterDatabase,
        hbond_database: HBondDatabase,
        device: torch.device,
    ) -> HBondParamResolver:
        "hbond pair parameter resolver"
        return HBondParamResolver.from_database(
            parameter_database.chemical, hbond_database, device
        )

    @reactive_property
    @validate_args
    def hbond_intra_module(
        hbond_database: HBondDatabase, hbond_param_resolver: HBondParamResolver
    ) -> HBondIntraModule:
        return HBondIntraModule(hbond_database, hbond_param_resolver)

    @reactive_property
    @validate_args
    def hbond_elements(
        parameter_database: ParameterDatabase,
        hbond_database: HBondDatabase,
        atom_types: NDArray(object)[:, :],
        bonds: NDArray(int)[:, 3],
    ) -> HBondElementAnalysis:
        """hbond score elements in target graph"""
        assert atom_types.shape[0] == 1
        assert numpy.all(bonds[:, 0] == 0)

        return HBondElementAnalysis.setup_from_database(
            chemical_database=parameter_database.chemical,
            hbond_database=hbond_database,
            atom_types=atom_types[0],
            bonds=bonds[:, 1:],
        )

    @reactive_property
    @validate_args
    def hbond_donor_indices(
        hbond_elements: HBondElementAnalysis, hbond_param_resolver: HBondParamResolver
    ) -> HBondDonorIndices:
        """hbond donor indicies and type indicies."""

        donor_type = hbond_param_resolver.resolve_donor_type(
            hbond_elements.donors["donor_type"]
        ).to(torch.int32)
        D = torch.from_numpy(hbond_elements.donors["d"]).to(device=donor_type.device)
        H = torch.from_numpy(hbond_elements.donors["h"]).to(device=donor_type.device)

        return HBondDonorIndices(D=D, H=H, donor_type=donor_type)

    @reactive_property
    @validate_args
    def hbond_acceptor_indices(
        hbond_elements: HBondElementAnalysis, hbond_param_resolver: HBondParamResolver
    ) -> HBondAcceptorIndices:
        """hbond acceptor indicies and type indicies."""

        acceptor_type = hbond_param_resolver.resolve_acceptor_type(
            hbond_elements.acceptors["acceptor_type"]
        ).to(torch.int32)
        A = torch.from_numpy(hbond_elements.acceptors["a"]).to(
            device=acceptor_type.device
        )
        B = torch.from_numpy(hbond_elements.acceptors["b"]).to(
            device=acceptor_type.device
        )
        B0 = torch.from_numpy(hbond_elements.acceptors["b0"]).to(
            device=acceptor_type.device
        )

        return HBondAcceptorIndices(A=A, B=B, B0=B0, acceptor_type=acceptor_type)
