import torch
import numpy

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.system.restypes import Residue
from tmol.system.pose import PackedBlockTypes, Pose, Poses


# to dump pdbs
from tmol.system.packed import PackedResidueSystem
from tmol.utility.reactive import reactive_property
from tmol.score.score_graph import score_graph
from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.device import TorchDevice
from tmol.score.score_components import ScoreComponentClasses, IntraScore
from tmol.io.generic import to_pdb


@validate_args
def poses_from_assigned_rotamers(
    orig_poses: Poses,
    packed_block_types: PackedBlockTypes,
    pose_id_for_context: Tensor[torch.int32][:],
    context_coords: Tensor[torch.float32][:, :, :, 3],
    context_block_type: Tensor[torch.int32][:, :],
) -> Poses:
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

    return Poses(
        packed_block_types=packed_block_types,
        residues=residues,
        residue_coords=coords_numpy,
        coords=context_coords,
        inter_residue_connections=orig_poses.inter_residue_connections[pid4c_64],
        inter_block_bondsep=orig_poses.inter_block_bondsep[pid4c_64],
        block_type_ind=context_block_type,
        device=orig_poses.device,
    )


@validate_args
def pdb_lines_for_pose(poses: Poses, ind: int) -> str:
    @score_graph
    class DummyIntra(IntraScore):
        @reactive_property
        def total_dummy(target):
            return target.coords.sum()

    @score_graph
    class BASGCart(
        CartesianAtomicCoordinateProvider, BondedAtomScoreGraph, TorchDevice
    ):
        total_score_components = [
            ScoreComponentClasses(
                "dummy", intra_container=DummyIntra, inter_container=None
            )
        ]

    packed_system = PackedResidueSystem.from_residues(poses.residues[ind])
    bonded_atom_score_graph = BASGCart.build_for(packed_system)
    return to_pdb(bonded_atom_score_graph)
