import attr
import numpy
import torch

from tmol.types import ValidateAttrs
from tmol.types import NDArray
from tmol.types import Tensor


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LKBallBlockTypeParams(ValidateAttrs):
    tile_n_polar_atoms: NDArray[numpy.int32][:]
    tile_n_occluder_atoms: NDArray[numpy.int32][:]
    tile_pol_occ_inds: NDArray[numpy.int32][:, :]
    tile_lk_ball_params: NDArray[numpy.float32][:, :, 9]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LKBallPackedBlockTypesParams(ValidateAttrs):
    tile_n_polar_atoms: Tensor[torch.int32][:, :]
    tile_n_occluder_atoms: Tensor[torch.int32][:, :]
    tile_pol_occ_inds: Tensor[torch.int32][:, :, :]
    tile_lk_ball_params: Tensor[torch.float32][:, :, :, 9]
