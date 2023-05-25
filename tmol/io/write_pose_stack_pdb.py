import numpy
import torch
from tmol.pose.pose_stack import PoseStack
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from typing import Optional, Union


def atom_records_from_pose_stack(
    pose_stack: PoseStack,
    chain_labels: Optional[Union[NDArray[str][:], NDArray[str][:, :]]] = None,
):
    from tmol.io.chain_deduction import chain_inds_for_pose_stack

    return atom_records_from_coords(
        pose_stack.packed_block_types,
        chain_inds_for_pose_stack(pose_stack),
        pose_stack.block_type_ind64,
        pose_stack.coords,
        pose_stack.block_coord_offset,
        chain_labels,
    )


@validate_args
def atom_records_from_coords(
    pbt: "PackedBlockTypes",
    chain_ind_for_block: Union[Tensor[torch.int32][:, :], NDArray[numpy.int32][:, :]],
    block_types64: Tensor[torch.int32][:, :],
    pose_like_coords: Tensor[torch.int32][:, :, 3],
    block_coord_offset: Tensor[torch.int32][:, :],
    chain_labels: Optional[Union[NDArray[str][:], NDArray[str][:, :]]] = None,
):
    from tmol.io.pdb_parsing import atom_record_dtype
    from tmol.utility.tensor.common_operations import exclusive_cumsum1d

    assert pose_like_coords.shape[0] == chain_ind_for_block.shape[0]
    assert pose_like_coords.shape[0] == block_types64.shape[0]
    assert pose_like_coords.shape[0] == block_coord_offset.shape[0]

    assert block_types64.shape[1] == block_coord_offset.shape[1]
    assert block_types64.shape[1] == chain_ind_for_block.shape[1]

    n_poses = block_coord_offset.shape[0]
    max_n_blocks = block_coord_offset.shape[1]
    max_n_pose_atoms = pose_like_coords.shape[1]

    is_real_block = block_types64 != -1
    real_block_pose, real_block_block = torch.nonzero(is_real_block, as_tuple=True)
    # n_real_blocks = torch.sum(is_real_block.to(torch.int64), dim=1)
    # pose_for_real_block = torch.arange(
    #     n_poses, dtype=torch.int64, device=pbt.device
    # ).repeat_interleave(n_real_blocks, dim=0)
    # print("pose_for_real_block.shape")
    # print(pose_for_real_block.shape)
    # print("n_real_blocks")
    # print(n_real_blocks)

    n_block_atoms = torch.zeros(
        (n_poses, max_n_blocks), dtype=torch.int32, device=pbt.device
    )
    n_block_atoms[is_real_block] = pbt.n_atoms[block_types64[is_real_block]]
    block_coord_offset64 = block_coord_offset.to(torch.int64)
    block_for_pose_atom = torch.zeros(
        (n_poses, max_n_pose_atoms), dtype=torch.int64, device=pbt.device
    )
    block_for_pose_atom[
        real_block_pose, block_coord_offset64[real_block_pose, real_block_block]
    ] = 1
    block_for_pose_atom = torch.cumsum(block_for_pose_atom, dim=1) - 1
    pose_for_pose_atom = torch.arange(
        n_poses, dtype=torch.int64, device=pbt.device
    ).repeat((1, max_n_pose_atoms))

    n_pose_atoms = torch.sum(n_block_atoms, dim=1)
    n_atoms_total = torch.sum(n_pose_atoms)

    # so we can index just the atoms out of coords tensor, e.g.
    # we need to know which atoms are real and which are padding
    atom_is_real = torch.tile(
        torch.arange(max_n_pose_atoms, dtype=torch.int64, device=pbt.device),
        (n_poses, 1),
    ) < n_pose_atoms.unsqueeze(dim=1)

    pose_for_real_atom = torch.arange(
        n_poses,
        dtype=torch.int64,
        device=pbt.device,
    ).repeat_interleave(n_pose_atoms, dim=0)
    block_for_atom = torch.zeros(
        (n_poses, max_n_pose_atoms), dtype=torch.int64, device=pbt.device
    )
    block_for_atom[
        real_block_pose, block_coord_offset64[real_block_pose, real_block_block]
    ] = 1
    block_for_atom = torch.cumsum(block_for_atom, dim=1) - 1
    block_for_real_atom = block_for_atom[atom_is_real]
    block_local_atom_index_for_pose_atom = (
        torch.arange(max_n_pose_atoms, dtype=torch.int32, device=pbt.device).repeat(
            (n_poses, 1)
        )
        - block_coord_offset[pose_for_pose_atom, block_for_pose_atom]
    )
    block_local_atom_index_for_real_atom = block_local_atom_index_for_pose_atom[
        atom_is_real
    ]

    pose_atom_offsets = exclusive_cumsum1d(n_pose_atoms)
    # chain_ind_for_block = torch.zeros(
    #     (n_poses, max_n_blocks),
    #     dtype=torch.int32,
    #     device=pbt.device
    # )
    # chain_ind_for_block[chain_begin != 0] = 1
    # chain_ind_for_block = torch.cumsum(chain_ind_for_block, dim=1) - 1

    # ok, let's move everything to the cpu/numpy from here forward
    # chain_begin = chain_begin.cpu().numpy()
    block_types64 = block_types64.cpu().numpy()
    pose_like_coords = pose_like_coords.cpu().numpy()
    block_coord_offset = block_coord_offset.cpu().numpy()
    is_real_block = is_real_block.cpu().numpy()
    n_block_atoms = n_block_atoms.cpu().numpy()
    n_pose_atoms = n_pose_atoms.cpu().numpy()
    pose_for_real_atom = pose_for_real_atom.cpu().numpy()
    atom_is_real = atom_is_real.cpu().numpy()
    # chain_ind_for_block = chain_ind_for_block.cpu().numpy()
    # chain_ind_for_real_atom = chain_ind_for_real_atom.cpu().numpy()
    block_for_atom = block_for_atom.cpu().numpy()
    block_for_real_atom = block_for_real_atom.cpu().numpy()
    block_local_atom_index_for_real_atom = (
        block_local_atom_index_for_real_atom.cpu().numpy()
    )

    # n_res = numpy.cumsum(is_real_block, axis=1)

    chain_ind_for_real_atom = chain_ind_for_block[
        pose_for_real_atom, block_for_real_atom
    ]

    # n_pose_arange = numpy.tile(numpy.arange(max_n_blocks, dtype=int), (n_poses,1))
    # pose_res_is_real

    # n_atoms = pose_like_coords.shape[1]
    results = numpy.empty(n_atoms_total, dtype=atom_record_dtype)
    results["record_name"] = numpy.full((n_atoms_total,), "ATOM  ", dtype=str)
    results["modeli"] = pose_for_real_atom
    results["chaini"] = chain_ind_for_real_atom
    # chain_begin = chain_begin.cpu().numpy()
    # res_begin = numpy.full((n_atoms,), 0, dtype=int)
    # res_begin[block_coord_offset[0]] = 1
    # res_for_atom = numpy.cumsum(res_begin) - 1
    # print("res for atom")
    # print(res_for_atom)
    results["resi"] = block_for_atom[atom_is_real] + 1
    results["atomi"] = (
        numpy.arange(n_atoms_total, dtype=numpy.int)
        + 1
        - pose_atom_offsets[pose_for_real_atom]
    )
    results["model"] = pose_for_real_atom + 1

    if chain_labels is None:
        chain_labels = numpy.array([x for x in "ABCDEFGHIJKLKMNOPQRSTUVWXY"])

    if len(chain_labels.shape) == 1:
        results["chain"] = chain_labels[chain_ind_for_real_atom]
    elif len(chain_labels.shape) == 2:
        results["chain"] = chain_labels[pose_for_real_atom, chain_ind_for_real_atom]

    # print("block_types64[0, i]")
    # print([block_types64[0, i] for i in res_for_atom])

    # create lookup for atom names
    bt_names = numpy.array([bt.name[:3] for bt in pbt.active_block_types])
    bt_atom_names = numpy.empty((pbt.n_types, pbt.max_n_atoms), dtype=numpy.object_)
    for i, bt in enumerate(pbt.active_block_types):
        for j, at in enumerate(bt.atoms):
            bt_atom_names[i, j] = at.name

    bt_for_real_atom = block_types64[pose_for_real_atom, block_for_real_atom]
    results["resn"] = bt_names[bt_for_real_atom]
    results["atomn"] = bt_atom_names[
        bt_for_real_atom, block_local_atom_index_for_real_atom
    ]
    real_atom_coords = pose_like_coords[atom_is_real]
    results["x"] = real_atom_coords[:, 0]
    results["y"] = real_atom_coords[:, 1]
    results["z"] = real_atom_coords[:, 2]
    results["insert"] = " "
    results["occupancy"] = 1
    results["b"] = 0

    return results
