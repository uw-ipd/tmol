import torch
import numpy

from tmol.utility.tensor.common_operations import (
    stretch,
    exclusive_cumsum2d_and_totals,
)
from tmol.score.common.stack_condense import (
    condense_subset,
    take_values_w_sentineled_index,
)

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.chemical.restypes import Residue
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


# to dump pdbs
from tmol.system.packed import PackedResidueSystem
from tmol.utility.reactive import reactive_property

# from tmol.score.score_graph import score_graph
# from tmol.score.bonded_atom import BondedAtomScoreGraph
# from tmol.score.coordinates import CartesianAtomicCoordinateProvider
# from tmol.score.device import TorchDevice
# from tmol.score.score_components import ScoreComponentClasses, IntraScore
from tmol.io.generic import to_pdb


@validate_args
def poses_from_assigned_rotamers(
    orig_poses: PoseStack,
    packed_block_types: PackedBlockTypes,
    pose_id_for_context: Tensor[torch.int32][:],
    context_coords: Tensor[torch.float32][:, :, 3],
    context_coord_offsets: Tensor[torch.int32][:, :],
    context_block_type: Tensor[torch.int32][:, :],
) -> PoseStack:
    pbt = packed_block_types
    device = pbt.device

    n_poses = context_coords.shape[0]
    max_context_coords_n_atoms = context_coords.shape[1]
    # print("max_context_coords_n_atoms")
    # print(max_context_coords_n_atoms)

    real_context_blocks = context_block_type != -1
    (real_context_block_context_ind, real_context_block_block_ind) = torch.nonzero(
        real_context_blocks, as_tuple=True
    )
    # real_context_block_type = context_block_type[context_block_type != -1]
    # print("real_context_block_type")
    # print(real_context_block_type.shape)
    # n_context_atoms = pbt.n_atoms[real_context_block_type.to(torch.int64)]
    context_block_type64 = context_block_type.to(torch.int64)
    n_context_atoms = take_values_w_sentineled_index(
        pbt.n_atoms, context_block_type64, default_fill=0
    )
    # print("n_context_atoms")
    # print(n_context_atoms)
    n_atoms_offset, n_ats_total = exclusive_cumsum2d_and_totals(n_context_atoms)
    max_n_atoms = torch.max(n_ats_total).item()

    atom_begin = torch.zeros(
        (n_poses, max_context_coords_n_atoms), dtype=torch.int32, device=device
    )
    # print("atom_begin.shape")
    # print(atom_begin.shape)
    (nz_context_coord_offsets, _) = torch.nonzero(
        context_coord_offsets != -1, as_tuple=True
    )
    # print("nz_context_coord_offsets")
    # print(nz_context_coord_offsets)
    context_coord_offsets64 = context_coord_offsets.to(torch.int64)
    # print("context_coord_offsets64[context_coord_offsets!=-1]")
    # print(context_coord_offsets64[context_coord_offsets!=-1])
    atom_begin[
        nz_context_coord_offsets, context_coord_offsets64[context_coord_offsets != -1]
    ] = 1
    # atom_begin = atom_begin.flatten()
    cs_atom_begin = torch.cumsum(atom_begin, dim=1)
    # print("cs_atom_begin")
    # print(cs_atom_begin)
    block_for_atom = cs_atom_begin - 1

    # print("block_for_atom")
    # print(block_for_atom)
    # print(block_for_atom.shape)

    context_for_atom64 = stretch(
        torch.arange(n_poses, dtype=torch.int64), max_context_coords_n_atoms
    ).view(n_poses, max_context_coords_n_atoms)
    block_type_for_atom64 = context_block_type64[
        context_for_atom64, block_for_atom
    ].view(n_poses, max_context_coords_n_atoms)

    block_n_atoms_for_atom = pbt.n_atoms[block_type_for_atom64]
    # print("block_n_atoms_for_atom")
    # print(block_n_atoms_for_atom.shape)

    context_block_offset_for_atom = torch.gather(
        context_coord_offsets, dim=1, index=block_for_atom
    )

    block_ind_for_atom = (
        torch.remainder(
            torch.arange(
                n_poses * max_context_coords_n_atoms, dtype=torch.int32, device=device
            ),
            max_context_coords_n_atoms,
        ).view(n_poses, max_context_coords_n_atoms)
        - context_block_offset_for_atom
    )

    # print("block_ind_for_atom")
    # print(block_ind_for_atom)

    context_atom_is_legit = block_ind_for_atom < block_n_atoms_for_atom
    # print("context_atom_is_legit")
    # print(context_atom_is_legit)

    # condensed_coords = torch.zeros((n_poses, max_n_atoms, 3), dtype=torch.float32, device=device)
    condensed_coords = condense_subset(
        context_coords, context_atom_is_legit, default_fill=0.0
    )

    # atoms_to_keep =

    # condensed_coords =
    # torch.zeros((n_poses, max_n_atoms, 3), dtype=torch.float32, device=device)

    nats = pbt.n_atoms.cpu()

    nres = torch.sum(context_block_type != -1, dim=1).cpu()
    cbt_cpu = context_block_type.cpu()
    # coords_numpy = context_coords.cpu().numpy().astype(numpy.float64)
    coords_numpy = condensed_coords.cpu().numpy().astype(numpy.float64)
    n_atoms_offset_cpu = n_atoms_offset.cpu().numpy()

    residues = [
        [
            Residue(
                residue_type=pbt.active_block_types[cbt_cpu[i, j]],
                coords=coords_numpy[
                    i,
                    n_atoms_offset_cpu[i, j] : (
                        n_atoms_offset_cpu[i, j] + nats[cbt_cpu[i, j]]
                    ),
                    :,
                ],
            )
            for j in range(nres[i])
        ]
        for i in range(context_coords.shape[0])
    ]

    pid4c_64 = pose_id_for_context.to(torch.int64)

    return PoseStack(
        packed_block_types=packed_block_types,
        residues=residues,
        residue_coords=coords_numpy,
        coords=condensed_coords,
        block_coord_offset=n_atoms_offset,
        block_coord_offset64=n_atoms_offset.to(torch.int64),
        inter_residue_connections=orig_poses.inter_residue_connections[pid4c_64],
        inter_residue_connections64=orig_poses.inter_residue_connections64[pid4c_64],
        inter_block_bondsep=orig_poses.inter_block_bondsep[pid4c_64],
        inter_block_bondsep64=orig_poses.inter_block_bondsep64[pid4c_64],
        block_type_ind=context_block_type,
        block_type_ind64=context_block_type64,
        device=orig_poses.device,
    )


# find replacement @validate_args
# find replacement def pdb_lines_for_pose(poses: PoseStack, ind: int) -> str:
# find replacement     @score_graph
# find replacement     class DummyIntra(IntraScore):
# find replacement         @reactive_property
# find replacement         def total_dummy(target):
# find replacement             return target.coords.sum()
# find replacement
# find replacement     @score_graph
# find replacement     class BASGCart(
# find replacement         CartesianAtomicCoordinateProvider, BondedAtomScoreGraph, TorchDevice
# find replacement     ):
# find replacement         total_score_components = [
# find replacement             ScoreComponentClasses(
# find replacement                 "dummy", intra_container=DummyIntra, inter_container=None
# find replacement             )
# find replacement         ]
# find replacement
# find replacement     packed_system = PackedResidueSystem.from_residues(poses.residues[ind])
# find replacement     bonded_atom_score_graph = BASGCart.build_for(packed_system)
# find replacement     return to_pdb(bonded_atom_score_graph)
