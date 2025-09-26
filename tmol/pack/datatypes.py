import torch
import attr

from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ConvertAttrs


@attr.s(auto_attribs=True, frozen=True)
class PackerEnergyTables(TensorGroup, ConvertAttrs):
    max_n_rotamers_per_pose: int
    pose_n_res: Tensor[torch.int32][:]  # [n-poses]
    pose_n_rotamers: Tensor[torch.int32][:]  # [n-poses]
    pose_rotamer_offset: Tensor[torch.int32][:]  # [n-poses]
    nrotamers_for_res: Tensor[torch.int32][:, :]  # [n-poses x n-res]
    oneb_offsets: Tensor[torch.int32][:, :]  # [n-poses x n-res]
    res_for_rot: Tensor[torch.int32][:]  # [n-rotamers-total]
    chunk_size: int
    chunk_offset_offsets: Tensor[torch.int64][:, :, :]  # [n-poses x n-res x n-res]
    chunk_offsets: Tensor[torch.int64][:]  # [n-interacting-chunk-pairs]
    energy1b: Tensor[torch.float32][:]  # [nrotamers_total]
    energy2b: Tensor[torch.float32][:]  # [ntwob_energies]
