import pandas
import numpy
import attr

from typing import Optional
from functools import singledispatch

import torch

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from .torch_op import OmegaOp

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup


@attr.s(auto_attribs=True)
class OmegaParams(TensorGroup):
    omega_indices: Tensor(torch.int32)[..., 4]


@reactive_attrs
class OmegaIntraScore(IntraScore):
    @reactive_property
    @validate_args
    def total_omega(omega_score):
        score_val = omega_score
        return score_val.sum()

    @reactive_property
    @validate_args
    def omega_score(target):
        return target.omega_op.intra(
            target.coords[0, ...],
            target.omega_resolve_indices.omega_indices,
            target.spring_constant,
        )


@score_graph
class OmegaScoreGraph(BondedAtomScoreGraph, TorchDevice):
    total_score_components = [
        ScoreComponentClasses(
            "omega", intra_container=OmegaIntraScore, inter_container=None
        )
    ]

    @staticmethod
    @singledispatch
    def factory_for(val, device: torch.device, **_):
        return dict(allomegas=val.allomegas, device=device)

    allomegas: NDArray(int)[:, 4]
    device: torch.device

    @reactive_property
    def spring_constant(device: torch.device) -> float:
        """ The spring constant for omega (in radians)"""
        return torch.tensor(32.8, device=device, dtype=torch.float)

    @reactive_property
    @validate_args
    def omega_op(device: torch.device) -> OmegaOp:
        return OmegaOp.from_device(device)

    @reactive_property
    @validate_args
    def omega_resolve_indices(
        device: torch.device, allomegas: NDArray(int)[:, 4]
    ) -> OmegaParams:
        # remove undefined indices and send to device
        omega_defined = numpy.all(allomegas != -1, axis=1)
        omega_indices = torch.from_numpy(allomegas[omega_defined, :]).to(
            device=device, dtype=torch.int32
        )

        return OmegaParams(omega_indices=omega_indices)
