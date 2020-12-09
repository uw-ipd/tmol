import numpy
import torch
import attr

from tmol.pack.rotamer.build_rotamers import build_rotamers
from tmol.pack.rotamer.bounding_spheres import create_rotamer_bounding_spheres

from tmol.system.pose import PackedBlockTypes, Pose, Poses
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler

from tmol.utility.tensor.common_operations import exclusive_cumsum1d


def test_create_rotamer_bounding_spheres_smoke(
    default_database, fresh_default_restype_set, rts_ubq_res, torch_device, dun_sampler
):
    # torch_device = torch.device("cpu")

    p1 = Pose.from_residues_one_chain(rts_ubq_res[:3], torch_device)
    p2 = Pose.from_residues_one_chain(rts_ubq_res[:2], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)
    palette = PackerPalette(fresh_default_restype_set)
    task = PackerTask(poses, palette)
    # leu_set = set(["LEU"])
    # for one_pose_rlts in task.rlts:
    #     for rlt in one_pose_rlts:
    #         rlt.restrict_absent_name3s(leu_set)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    rotamer_set = build_rotamers(poses, task, default_database.chemical)

    bounding_spheres = create_rotamer_bounding_spheres(poses, rotamer_set)
    # print("bounding spheres")
    # print(bounding_spheres.shape)

    rot_coords = rotamer_set.coords.cpu()
    bounding_spheres = bounding_spheres.cpu()
    n_atoms = poses.packed_block_types.n_atoms.cpu()
    fudge = 1e-4
    for i in range(rotamer_set.coords.shape[0]):
        i_bti = rotamer_set.block_type_ind_for_rot[i]
        i_pose = rotamer_set.pose_for_rot[i]
        i_bi = rotamer_set.block_ind_for_rot[i]

        i_sphere = bounding_spheres[i_pose, i_bi]
        i_sphere_coord = i_sphere[:3]
        i_radius = i_sphere[3]
        for j in range(n_atoms[i_bti]):
            dist = torch.norm(i_sphere_coord - rot_coords[i, j], dim=0)
            assert dist <= i_radius + fudge
