import torch
import attr

from tmol.utility.tensor.common_operations import stretch
from tmol.system.pose import Poses
from tmol.pack.rotamer.build_rotamers import RotamerSet


def create_rotamer_bounding_spheres(poses: Poses, rotamer_set: RotamerSet):
    torch_device = poses.coords.device
    n_poses = poses.coords.shape[0]
    max_n_blocks = poses.coords.shape[1]
    n_rots = rotamer_set.pose_for_rot.shape[0]

    bounding_spheres = torch.full(
        (n_poses, max_n_blocks, 4), 0, dtype=torch.float32, device=torch_device
    )

    # what is the center of the smallest sphere that encloses all the rotamers?
    # let's just take the center of mass for the rotamers

    global_block_ind_for_rot = (
        rotamer_set.pose_for_rot * max_n_blocks
        + rotamer_set.block_ind_for_rot.to(torch.int64)
    )
    max_n_atoms = poses.packed_block_types.max_n_atoms
    centers_of_mass = torch.zeros(
        (n_poses * max_n_blocks, 3), dtype=torch.float32, device=torch_device
    )
    centers_of_mass.index_add_(
        0,
        stretch(global_block_ind_for_rot, max_n_atoms),
        rotamer_set.coords.reshape(-1, 3),
    )
    n_ats_for_rot = poses.packed_block_types.n_atoms[rotamer_set.block_type_ind_for_rot]
    n_ats = torch.zeros(
        (n_poses * max_n_blocks,), dtype=torch.int32, device=torch_device
    )
    # print("global_block_ind_for_rot")
    # print(global_block_ind_for_rot.shape)
    # print("n_ats_for_rot")
    # print(n_ats_for_rot.shape)
    n_ats.index_add_(0, global_block_ind_for_rot, n_ats_for_rot)

    # print("centers_of_mass")
    # print(centers_of_mass.shape)
    # print("n_ats")
    # print(n_ats)
    # print("n_ats != 0")
    # print(n_ats[n_ats != 0])

    centers_of_mass[n_ats != 0] = centers_of_mass[n_ats != 0] / n_ats[
        n_ats != 0
    ].unsqueeze(1).to(torch.float32)
    # print("centers_of_mass[:10]")
    # print(centers_of_mass[:10])
    at_is_real = torch.arange(
        max_n_atoms, dtype=torch.int32, device=torch_device
    ).repeat(n_rots).reshape(n_rots, max_n_atoms) < n_ats_for_rot.unsqueeze(dim=1)
    diff_w_com = torch.zeros_like(rotamer_set.coords)

    diff_w_com[at_is_real] = (
        centers_of_mass[stretch(global_block_ind_for_rot, max_n_atoms)].reshape(
            n_rots, max_n_atoms, 3
        )[at_is_real]
        - rotamer_set.coords[at_is_real]
    )
    # print("diff_w_com[:10]")
    # print(diff_w_com[:10])
    atom_dist_to_com = torch.norm(diff_w_com, dim=2)
    # print("atom_dist_to_com")
    # print(atom_dist_to_com[:10])
    rot_dist_to_com = torch.max(atom_dist_to_com, dim=1)[0]
    # print("rot_dist_to_com")
    # print(rot_dist_to_com[:10])

    # now I need to get the max for all the rotamers at a single position, and I
    # don't know how to do that except 1) segmented scan on max in c++, or
    # 2) create an overly-large tensor of n-poses x max-n-blocks x max-n-rots
    # and then populate that tensor with rot_dist_to_com
    max_n_rots_per_block = torch.max(rotamer_set.n_rots_for_block)
    rot_dist_to_com_big = torch.zeros(
        (n_poses * max_n_blocks, max_n_rots_per_block),
        dtype=torch.float32,
        device=torch_device,
    )

    rot_dist_to_com_big[
        global_block_ind_for_rot,
        (
            torch.arange(n_rots, dtype=torch.int64, device=torch_device)
            - rotamer_set.rot_offset_for_block.flatten()[global_block_ind_for_rot]
        ),
    ] = rot_dist_to_com
    sphere_radius = torch.max(rot_dist_to_com_big, dim=1)[0]
    # print("sphere_radius")
    # print(sphere_radius)
    bounding_spheres = torch.zeros(
        (n_poses * max_n_blocks, 4), dtype=torch.float32, device=torch_device
    )
    bounding_spheres[:, 3][n_ats != 0] = sphere_radius[n_ats != 0]
    bounding_spheres[:, :3][n_ats != 0] = centers_of_mass[n_ats != 0]
    # print("bounding_spheres 1")
    # print(bounding_spheres)

    background_centers_of_mass = torch.sum(poses.coords, dim=2).reshape(-1, 3)
    pbti = poses.block_type_ind.to(torch.int64).flatten()
    background_n_ats = torch.zeros_like(pbti)
    background_n_ats[pbti != -1] = poses.packed_block_types.n_atoms[
        pbti[pbti != -1]
    ].to(torch.int64)

    background_centers_of_mass[pbti != -1] = background_centers_of_mass[
        pbti != -1
    ] / background_n_ats[pbti != -1].unsqueeze(1).to(torch.float32)
    pose_at_is_real = torch.arange(
        max_n_atoms, dtype=torch.int64, device=torch_device
    ).repeat(n_poses * max_n_blocks).reshape(
        n_poses * max_n_blocks, max_n_atoms
    ) < background_n_ats.unsqueeze(
        dim=1
    )
    pose_at_is_real = pose_at_is_real.flatten()
    pose_diff_w_com = torch.zeros_like(poses.coords).reshape(-1, 3)
    pose_diff_w_com[pose_at_is_real] = (
        background_centers_of_mass[
            stretch(
                torch.arange(
                    n_poses * max_n_blocks, dtype=torch.int64, device=torch_device
                ),
                max_n_atoms,
            )
        ][pose_at_is_real]
        - poses.coords.view(-1, 3)[pose_at_is_real]
    )
    pose_diff_w_com = pose_diff_w_com.view(n_poses * max_n_blocks, max_n_atoms, 3)
    pose_dist_to_com = torch.norm(pose_diff_w_com, dim=2)
    pose_bounding_radius = torch.max(pose_dist_to_com, dim=1)[0]

    bounding_spheres[:, 3][n_ats == 0] = pose_bounding_radius[n_ats == 0]
    bounding_spheres[:, :3][n_ats == 0] = background_centers_of_mass.reshape(-1, 3)[
        n_ats == 0
    ]

    # print("bounding spheres")
    # print(bounding_spheres)

    return bounding_spheres.view(n_poses, max_n_blocks, 4)
