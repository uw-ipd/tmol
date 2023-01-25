import attr

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@attr.s(auto_attribs=True, froze=True, slots=True)
class LKBallBlockTypeParams(ValidateAttrs):
    tile_n_polar_atoms: NDArray[numpy.int32][:]
    tile_n_occluder_atoms: NDArray[numpy.int32][:]
    tile_pol_occ_inds: NDArray[numpy.int32][:, :]
    tile_n_occluder_atoms: NDArray[numpy.int32][:]
    tile_lk_ball_params: NDArray[numpy.float32][:, :, :]


@attr.s(auto_attribs=True, froze=True, slots=True)
class LKBallPackedBlockTypeParams(ValidateAttrs):
    tile_n_polar_atoms: Tensor[torch.int32][:]
    tile_n_occluder_atoms: Tensor[torch.int32][:]
    tile_pol_occ_inds: Tensor[torch.int32][:, :]
    tile_n_occluder_atoms: Tensor[torch.int32][:]
    tile_lk_ball_params: Tensor[torch.float32][:, :, :]
