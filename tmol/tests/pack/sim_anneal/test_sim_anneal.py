import torch
import attr

from tmol.utility.tensor.common_operations import stretch
from tmol.system.restypes import ResidueTypeSet
from tmol.system.pose import PackedBlockTypes, Pose, Poses

from tmol.pack.sim_anneal.annealer import MCAcceptRejectModule, SelectRanRotModule


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

    p = Pose.from_residues_one_chain(ubq_res[:3], torch_device)
    poses = Poses.from_poses([p] * 5, torch_device)

    contexts = poses.coords.clone()

    arange3 = torch.arange(3, dtype=torch.int32, device=torch_device)
    arange5 = torch.arange(5, dtype=torch.int32, device=torch_device)
    max_n_atoms = poses.coords.shape[-2]

    selector = SelectRanRotModule(
        n_traj_per_pose=1,
        pose_id_for_context=arange5,
        n_rots_for_pose=torch.full((5,), 2, dtype=torch.int32, device=torch_device),
        rot_offset_for_pose=arange5 * 2,
        block_type_ind_for_rot=stretch(poses.block_type_ind.view(-1), 2),
        block_ind_for_rot=stretch(arange3, 2).repeat(5),
        rotamer_coords=poses.coords[
            stretch(arange5, 6).to(torch.int64),
            stretch(arange3, 2).repeat(5).to(torch.int64),
        ],
    )

    context_coords = poses.coords.clone()
    context_block_type = poses.block_type_ind.clone()

    alt_coords, alt_ids, rr = selector(context_coords, context_block_type)
    # what to assert
    assert alt_coords.shape == (10, poses.coords.shape[2], 3)
    assert alt_coords.device == torch_device
    assert alt_ids.shape == (10, 3)
    assert alt_ids.device == torch_device


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

    p = Pose.from_residues_one_chain(ubq_res[:3], torch_device)
    poses = Poses.from_poses([p] * 5, torch_device)

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
        10, 1
    )
    temperature = torch.ones((1,), dtype=torch.float32, device=torch_device)

    mc_accept_reject = MCAcceptRejectModule()

    accept = mc_accept_reject(
        temperature,
        context_coords,
        context_block_type,
        alternate_coords,
        alternate_ids,
        faux_energies,
    )
    print(accept)
