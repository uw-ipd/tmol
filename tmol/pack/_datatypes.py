import torch
import attr

from tmol.types import (
    Tensor,
    TensorGroup,
    ConvertAttrs,
)


@attr.s(auto_attribs=True, frozen=True)
class PackerEnergyTables(TensorGroup, ConvertAttrs):
    """One- and two-body energy tables plus their packed rotamer indexing."""

    max_n_rotamers_per_pose: int
    pose_n_res: Tensor[torch.int32][:]  # [n-poses]
    pose_n_rotamers: Tensor[torch.int32][:]  # [n-poses]
    pose_rotamer_offset: Tensor[torch.int32][:]  # [n-poses]
    nrotamers_for_res: Tensor[torch.int32][:, :]  # [n-poses x n-res]
    oneb_offsets: Tensor[torch.int32][:, :]  # [n-poses x n-res]
    res_for_rot: Tensor[torch.int32][:]  # [n-rotamers-total]
    chunk_size: int
    neighbor_row_offsets: Tensor[torch.int64][:]  # CSR rows, [n-poses*n-res + 1]
    neighbor_blocks: Tensor[torch.int32][:]  # CSR columns, [n-directed-edges]
    # Chunk-table offset per directed edge.
    neighbor_chunk_offset_offsets: Tensor[torch.int64][:]
    chunk_offsets: Tensor[torch.int64][:]  # [n-interacting-chunk-pairs]
    energy1b: Tensor[torch.float32][:]  # [nrotamers_total]
    energy2b: Tensor[torch.float32][:]  # [ntwob_energies]
