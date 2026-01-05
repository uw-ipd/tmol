import attr
import torch
import numpy
from typing import Optional

from tmol.types.torch import Tensor
from tmol.types.array import NDArray


@attr.s(auto_attribs=True, frozen=False, slots=True)
class CanonicalForm:
    chain_id: Tensor[torch.int64][:, :]  # n_poses x max_n_res
    res_types: Tensor[torch.int64][:, :]  # n_poses x max_n_res
    coords: Tensor[torch.float32][
        :, :, :, 3
    ]  # n_poses x max_n_res x max_n_canonical_atoms x 3
    res_labels: NDArray[int][:, :]  # n_poses x max_n_res
    residue_insertion_codes: NDArray[object][:, :]  # n_poses x max_n_res
    chain_labels: NDArray[object][:, :]  # n_poses x max_n_res
    atom_occupancy: Optional[
        NDArray[numpy.float32][:, :, :]
    ]  # n_poses x max_n_res x max_n_canonical_atoms
    atom_b_factor: Optional[
        NDArray[numpy.float32][:, :, :]
    ]  # n_poses x max_n_res x max_n_canonical_atoms
    disulfides: Optional[Tensor[torch.int64][:, 3]]  # n_disulfides x 3
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]]  # n_poses x max_n_res x 2

    def __iter__(self):
        yield self.chain_id
        yield self.res_types
        yield self.coords
        yield self.res_labels
        yield self.residue_insertion_codes
        yield self.chain_labels
        yield self.atom_occupancy
        yield self.atom_b_factor
        yield self.disulfides
        yield self.res_not_connected
