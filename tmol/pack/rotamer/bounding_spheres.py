import torch
import attr

from tmol.utility.tensor.common_operations import stretch


def create_rotamer_bounding_spheres(poses: Poses, rotamer_set: RotamerSet):
    torch_device = poses.coords.device
    n_poses = poses.coords.shape[0]
    max_n_blocks = poses.coords.shape[1]

    bounding_spheres = torch.full(
        (n_poses, max_n_blocks, 4), 0, dtype=torch.float32, device=torch_device
    )

    # what is the center of the smallest sphere that encloses all the rotamers?
    # let's just take the center of mass for the rotamers

    global_block_ind_for_rot = (pose_for_rot * max_n_bocks + block_ind_for_rot).to(
        torch.int64
    )
    max_n_atoms = poses.packed_block_types.max_n_atoms
    centers_of_mass = torch.zeros(
        (n_poses * max_n_blocks, 3), dtype=torch.float32, device=torch_device
    )
    centers_of_mass.index_add_(
        0, stretch(global_block_ind_for_rot, max_n_atoms), rotamer_coords.reshape(-1, 3)
    )
    n_ats_for_rot = poses.packed_block_types.n_atoms[block_type_ind_for_rot]
    n_ats = torch.zeros(
        (n_poses * max_n_blocks, 3), dtype=torch.int32, device=torch_device
    )
    n_ats.index_add_(0, global_block_ind_for_rot, n_ats_for_rot)
    centers_of_mass[n_ats != 0] = centers_of_mass[n_ats != 0] / n_ats[n_ats != 0]
    at_is_real = torch.arange(
        max_n_atoms, dtype=torch.int32, device=torch_device
    ).repeat(n_rots).reshape(n_rots, max_n_atoms) < n_ats_for_rot.unsqueeze(dim=1)
    diff_w_com = torch.zeros_like(rotamer_coords)
    diff_w_com[at_is_real] = (
        centers_of_mass[stretch(global_block_ind_for_rot, max_n_atoms)][at_is_real]
        - rotamer_coords[at_is_real]
    )
    atom_dist_to_com = torch.norm(diff_w_com, dim=2)
    rot_dist_to_com = torch.max(atom_dist_to_com, dim=1)

    # now I need to get the max for all the rotamers at a single position, and I
    # don't know how to do that except 1) segmented scan on max in c++, or
    # 2) create an overly-large tensor of n-poses x max-n-blocks x max-n-rots
    # and then populate that tensor with rot_dist_to_com
    rot_dist_to_com_big = torch.zeros(
        (n_poses * max_n_blocks, max_n_rots), dtype=torch.float32, device=torch_device
    )
    rot_dist_to_com_big[
        global_block_ind_for_rot,
        (
            torch.arange(n_rots, dtype=torch.int64, device=torch_device)
            - rot_offset_for_block[block_ind_for_rot.to(torch.int64)]
        ),
    ] = rot_dist_to_com_big
    sphere_radius = torch.max(rot_dist_to_com_big, dim=1)
    bounding_spheres = torch.zeros(
        (n_poses * max_n_blocks, 4), dtype=torch.float32, device=torch_device
    )
    bounding_spheres[n_ats != 0, 3] = sphere_radius[n_ats != 0]
    bounding_spheres[n_ats != 0, :3] = centers_of_mass[n_ats != 0]

    background_centers_of_mass = torch.sum(poses.coords, dim=2).flatten()
    pbti = poses.block_type_ind
    background_n_ats = torch.zeros_like(pbti)
    background_n_ats[pbti != -1] = poses.packed_block_types.n_atoms[pbti[pbti != -1]]
    background_centers_of_mass = (
        background_centers_of_mass[pbti != -1] / background_n_ats[pbti != -1]
    )
    pose_at_is_real = torch.arange(
        max_n_atoms, dtype=torch.int32, device=torch_device
    ).repeat(n_poses * max_n_blocks) < background_n_ats.unsqueeze(dim=1)
    pose_diff_w_com = torch.zeros_like(pose.coords).reshape(-1, 3)
    pose_diff_w_com[pose_at_is_real] = (
        background_centers_of_mass[
            stretch(
                arange(n_poses * max_n_blocks, dtype=torch.int64, device=torch_device),
                max_n_atoms,
            )
        ][pose_at_is_real]
        - pose.coords.view(-1, 3)[pose_at_is_real]
    )
    pose_dist_to_com = torch.norm(pose_diff_w_com, dim=2)
    pose_bounding_radius = torch.max(pose_dist_to_com, dim=1)

    bounding_spheres[n_ats == 0, 3] = pose_bounding_radius[n_ats == 0]
    bounding_spheres[n_ats == 0, :3] = background_centers_of_mass.reshape(-1, 3)[
        n_ats == 0
    ]

    return bounding_spheres
