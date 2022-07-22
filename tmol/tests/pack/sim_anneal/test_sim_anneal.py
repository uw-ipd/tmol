import torch
import attr
import numpy

from tmol.utility.tensor.common_operations import stretch
from tmol.chemical.restypes import ResidueTypeSet
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

# to dump pdbs
from tmol.system.packed import PackedResidueSystem
from tmol.utility.reactive import reactive_property

# deprecated from tmol.score.score_graph import score_graph
# deprecated from tmol.score.bonded_atom import BondedAtomScoreGraph
# deprecated from tmol.score.coordinates import CartesianAtomicCoordinateProvider
# from tmol.score.device import TorchDevice
# from tmol.score.score_components import ScoreComponentClasses, IntraScore
from tmol.io.generic import to_pdb

from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.pack.rotamer.build_rotamers import RotamerSet, build_rotamers
from tmol.pack.sim_anneal.annealer import MCAcceptRejectModule, SelectRanRotModule
from tmol.pack.sim_anneal.accept_final import (
    poses_from_assigned_rotamers,
    #    pdb_lines_for_pose,
)


def test_random_rotamer_module(ubq_res, default_database, torch_device):
    # torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)

    # replace them with residues constructed from the residue types
    # that live in our locally constructed set of refined residue types
    ubq_res = [
        attr.evolve(
            res,
            residue_type=next(
                rt for rt in rts.residue_types if rt.name == res.residue_type.name
            ),
        )
        for res in ubq_res
    ]

    p = PoseStack.one_structure_from_polymeric_residues(ubq_res[:3], torch_device)
    poses = PoseStack.from_poses([p] * 5, torch_device)

    contexts = poses.coords.clone()

    arange3 = torch.arange(3, dtype=torch.int32, device=torch_device)
    arange5 = torch.arange(5, dtype=torch.int32, device=torch_device)
    max_n_atoms = poses.coords.shape[-2]

    faux_coords = torch.arange(
        max_n_atoms * 5 * 6 * 3, dtype=torch.float32, device=torch_device
    ).view(30, max_n_atoms, 3)

    alternate_coords = torch.zeros(
        (5 * 2, max_n_atoms, 3), dtype=torch.float32, device=torch_device
    )
    alternate_block_id = torch.zeros((5 * 2, 3), dtype=torch.int32, device=torch_device)
    random_rots = torch.zeros((5,), dtype=torch.int32, device=torch_device)

    selector = SelectRanRotModule(
        n_traj_per_pose=1,
        pose_id_for_context=arange5,
        n_rots_for_pose=torch.full((5,), 6, dtype=torch.int32, device=torch_device),
        rot_offset_for_pose=arange5 * 6,
        block_type_ind_for_rot=stretch(poses.block_type_ind.view(-1), 2),
        block_ind_for_rot=stretch(arange3, 2).repeat(5),
        rotamer_coords=faux_coords,
        alternate_coords=alternate_coords,
        alternate_block_id=alternate_block_id,
        random_rots=random_rots,
    )

    context_coords = poses.coords.clone()
    context_block_type = poses.block_type_ind.clone()

    selector.go(context_coords, context_block_type)

    # what to assert
    # assert alt_coords.shape == (10, poses.coords.shape[2], 3)
    # assert alt_coords.device == torch_device
    # assert alt_ids.shape == (10, 3)
    # assert alt_ids.device == torch_device

    rr = random_rots.to(torch.int64)
    alt_coords = alternate_coords.cpu().numpy()
    rotamer_alt_coords = alt_coords.reshape(5, 2, max_n_atoms, 3)[:, 1]

    gold_rotamer_alt_coords = faux_coords[rr, :, :].cpu().numpy()
    assert rotamer_alt_coords.shape == gold_rotamer_alt_coords.shape

    numpy.testing.assert_equal(gold_rotamer_alt_coords, rotamer_alt_coords)


def test_mc_accept_reject_module(ubq_res, default_database, torch_device):
    # torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)

    # replace them with residues constructed from the residue types
    # that live in our locally constructed set of refined residue types
    ubq_res = [
        attr.evolve(
            res,
            residue_type=next(
                rt for rt in rts.residue_types if rt.name == res.residue_type.name
            ),
        )
        for res in ubq_res
    ]

    p = PoseStack.one_structure_from_polymeric_residues(ubq_res[:3], torch_device)
    poses = PoseStack.from_poses([p] * 5, torch_device)

    contexts = poses.coords.clone()

    arange3 = torch.arange(3, dtype=torch.int64, device=torch_device)
    arange5 = torch.arange(5, dtype=torch.int64, device=torch_device)
    max_n_atoms = poses.coords.shape[-2]

    n_traj_per_pose = (1,)
    # pose_id_for_context = arange5,
    # n_rots_for_pose = torch.full((5,), 2, dtype=torch.int32, device=torch_device),
    # rot_offset_for_pose = arange5 * 2,
    # block_type_ind_for_rot = stretch(poses.block_type_ind.view(-1), 2),
    # block_ind_for_rot = stretch(arange3, 2).repeat(5),

    block_ind_for_alt = torch.remainder(
        stretch(torch.arange(5, dtype=torch.int64, device=torch_device), 2), 3
    )

    # context_coords = poses.coords[
    #     stretch(arange5,6),
    #     stretch(arange3,2).repeat(5)
    # ]
    context_coords = poses.coords.clone()
    context_block_type = poses.block_type_ind

    # take the coordinates from pose 0 for residues 0, 1, 2, 0, & 1
    # two rotamers each
    ten0s = torch.zeros((10,), dtype=torch.int64, device=torch_device)

    alternate_coords = poses.coords[ten0s, block_ind_for_alt]
    alternate_ids = torch.zeros((10, 3), dtype=torch.int32, device=torch_device)
    alternate_ids[:, 0] = stretch(arange5, 2).to(torch.int32)
    alternate_ids[:, 1] = block_ind_for_alt.to(torch.int32)
    alternate_ids[:, 2] = poses.block_type_ind[ten0s, block_ind_for_alt]
    faux_energies = torch.arange(10, dtype=torch.float32, device=torch_device).view(
        1, 10
    )
    accepted = torch.zeros((5,), dtype=torch.int32, device=torch_device)
    temperature = torch.ones((1,), dtype=torch.float32, device=torch.device("cpu"))

    mc_accept_reject = MCAcceptRejectModule()

    # print("context_coords", context_coords.shape)
    # print("alternate_coords", alternate_coords.shape)
    # print("alternate_ids", alternate_ids.shape)
    # print("faux_energies", faux_energies.shape)

    accept = mc_accept_reject.go(
        temperature,
        context_coords,
        context_block_type,
        alternate_coords,
        alternate_ids,
        faux_energies,
        accepted,
    )
    print(accept)


def test_accept_final(
    default_database, fresh_default_restype_set, rts_ubq_res, torch_device, dun_sampler
):
    # torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)

    max_n_blocks = 10
    n_poses = 3
    p = PoseStack.one_structure_from_polymeric_residues(
        rts_ubq_res[:max_n_blocks], torch_device
    )
    poses = PoseStack.from_poses([p] * n_poses, torch_device)

    palette = PackerPalette(fresh_default_restype_set)
    task = PackerTask(poses, palette)
    # task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    poses, rotamer_set = build_rotamers(poses, task, default_database.chemical)

    print("built rotamers", rotamer_set.pose_for_rot.shape)

    pose_id_for_context = torch.arange(n_poses, dtype=torch.int32, device=torch_device)
    context_coords = poses.coords.clone()
    rand_rot = torch.floor(
        torch.rand((n_poses, max_n_blocks), dtype=torch.float, device=torch_device)
        * rotamer_set.n_rots_for_block.to(torch.float32)
    ).to(torch.int64)
    rand_rot_global = rotamer_set.rot_offset_for_block + rand_rot
    packable_blocks = rotamer_set.n_rots_for_block != 0
    context_coords[packable_blocks] = rotamer_set.coords[
        rand_rot_global[packable_blocks]
    ]
    context_block_type = poses.block_type_ind.clone()
    context_block_type[packable_blocks] = rotamer_set.block_type_ind_for_rot[
        rand_rot_global[packable_blocks]
    ].to(torch.int32)

    randomized_poses = poses_from_assigned_rotamers(
        poses,
        poses.packed_block_types,
        pose_id_for_context,
        context_coords,
        context_block_type,
    )

    # @score_graph
    # class DummyIntra(IntraScore):
    #     @reactive_property
    #     def total_dummy(target):
    #         return target.coords.sum()
    #
    # @score_graph
    # class BASGCart(CartesianAtomicCoordinateProvider, BondedAtomScoreGraph, TorchDevice) :
    #     total_score_components = [
    #         ScoreComponentClasses("dummy", intra_container=DummyIntra, inter_container=None)
    #     ]

    # debug output for i in range(n_poses):
    # debug output     # packed_system = PackedResidueSystem.from_residues(randomized_poses.residues[i])
    # debug output     # bonded_atom_score_graph = BASGCart.build_for(packed_system)
    # debug output     # pdb = to_pdb(bonded_atom_score_graph)
    # debug output
    # debug output     pdb = pdb_lines_for_pose(randomized_poses, i)
    # debug output     with open("temp_repacked_pdb_{:04d}.pdb".format(i), "w") as fid:
    # debug output         fid.write(pdb)
