import torch

from tmol.pack.rotamer.build_rotamers import build_rotamers
from tmol.pack.rotamer.bounding_spheres import create_rotamer_bounding_spheres

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler

# from tmol.io import pose_stack_from_pdb
from tmol.tests.data import no_termini_pose_stack_from_pdb


def test_create_rotamer_bounding_spheres_smoke(
    default_database, fresh_default_restype_set, ubq_pdb, torch_device, dun_sampler
):
    p1 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=4
    )
    p2 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=3
    )
    p3 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=5
    )

    assert p1.packed_block_types is p2.packed_block_types
    assert p1.packed_block_types is p3.packed_block_types

    poses = PoseStackBuilder.from_poses([p1, p2, p3], torch_device)
    palette = PackerPalette(fresh_default_restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()
    task.rlts[0][1].disable_packing()
    task.rlts[2][2].disable_packing()

    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    poses, rotamer_set = build_rotamers(poses, task, default_database.chemical)

    bounding_spheres = create_rotamer_bounding_spheres(poses, rotamer_set)

    assert bounding_spheres.shape == (3, 4, 4)
    assert bounding_spheres.dtype == torch.float32
    assert bounding_spheres.device == torch_device

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

    bti = poses.block_type_ind.cpu()
    pcoords = poses.coords.cpu()
    pbco = poses.block_coord_offset.cpu()
    for i, j in [(0, 1), (2, 2)]:
        ij_n_atoms = n_atoms[bti[i, j]]
        ij_sphere_coord = bounding_spheres[i, j, :3]
        ij_radius = bounding_spheres[i, j, 3]
        for k in range(ij_n_atoms):
            kcoord = pcoords[i, pbco[i, j] + k]
            dist = torch.norm(ij_sphere_coord - kcoord, dim=0)
            assert dist < ij_radius + fudge


def test_build_spheres_for_lots_of_rotamers(
    default_database, fresh_default_restype_set, ubq_pdb, torch_device, dun_sampler
):
    n_poses = 10

    # fd TEMP: NO TERM VARIANTS
    # There are 76 residues in 1ubq
    p = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=75
    )
    poses = PoseStackBuilder.from_poses([p] * n_poses, torch_device)

    palette = PackerPalette(fresh_default_restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    poses, rotamer_set = build_rotamers(poses, task, default_database.chemical)

    bounding_spheres = create_rotamer_bounding_spheres(poses, rotamer_set)

    assert bounding_spheres is not None
