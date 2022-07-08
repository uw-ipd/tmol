import torch
import numpy

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
    context_coords: Tensor[torch.float32][:, :, :, 3],
    context_block_type: Tensor[torch.int32][:, :],
) -> PoseStack:
    pbt = packed_block_types
    nats = pbt.n_atoms.cpu()
    nres = torch.sum(context_block_type != -1, dim=1).cpu()
    cbt_cpu = context_block_type.cpu()
    coords_numpy = context_coords.cpu().numpy().astype(numpy.float64)
    residues = [
        [
            Residue(
                residue_type=pbt.active_block_types[cbt_cpu[i, j]],
                coords=coords_numpy[i, j, : (nats[cbt_cpu[i, j]])],
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
        coords=context_coords,
        inter_residue_connections=orig_poses.inter_residue_connections[pid4c_64],
        inter_block_bondsep=orig_poses.inter_block_bondsep[pid4c_64],
        block_type_ind=context_block_type,
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
