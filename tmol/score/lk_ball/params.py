import attr
import numpy
import torch

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LKBallBlockTypeParams(ValidateAttrs):
    tile_n_polar_atoms: NDArray[numpy.int32][:]
    tile_n_occluder_atoms: NDArray[numpy.int32][:]
    tile_pol_occ_inds: NDArray[numpy.int32][:, :]
    tile_lk_ball_params: NDArray[numpy.float32][:, :, 9]

    # lk_ball is _slightly_ different than hbond
    #   so in addition to hbond parameters, it needs to know polar H and attached hvy atoms
    #   e.g., CYS SG is NOT an hbond donor, but generates a "donor water"
    tile_n_donH: NDArray[numpy.int32][:]
    tile_donH_inds: NDArray[numpy.int32][:, :]
    tile_donH_hvy_inds: NDArray[numpy.int32][:, :]
    tile_which_donH_of_donH_hvy: NDArray[numpy.int32][:, :]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class LKBallPackedBlockTypesParams(ValidateAttrs):
    tile_n_polar_atoms: Tensor[torch.int32][:, :]
    tile_n_occluder_atoms: Tensor[torch.int32][:, :]
    tile_pol_occ_inds: Tensor[torch.int32][:, :, :]
    tile_lk_ball_params: Tensor[torch.float32][:, :, :, 9]

    tile_n_donH: Tensor[torch.int32][:, :]
    tile_donH_inds: Tensor[torch.int32][:, :, :]
    tile_donH_hvy_inds: Tensor[torch.int32][:, :, :]
    tile_which_donH_of_donH_hvy: Tensor[torch.int32][:, :, :]
