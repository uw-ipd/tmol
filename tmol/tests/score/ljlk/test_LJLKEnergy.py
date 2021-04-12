import numpy
import torch
import pytest

from tmol.score.ljlk.LJLKEnergy import LJLKEnergy
from tmol.score.ljlk.params import LJLKParamResolver
from tmol.system.pose import Pose, Poses, residue_types_from_residues, PackedBlockTypes
from tmol.score.chemical_database import AtomTypeParamResolver


def test_smoke(default_database, torch_device):

    resolver = AtomTypeParamResolver.from_database(
        default_database.chemical, torch_device
    )

    ljlk_params = LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, device=torch_device
    )

    ljlk_energy = LJLKEnergy(
        atom_type_resolver=resolver,
        type_params=ljlk_params.type_params,
        global_params=ljlk_params.global_params,
        # atom_type_index=ljlk_params.atom_type_index,
        device=torch_device,
    )

    assert ljlk_energy.type_params.lj_radius.device == torch_device
    assert ljlk_energy.global_params.max_dis.device == torch_device


def test_annotate_heavy_ats_in_tile(ubq_res, default_database, torch_device):

    resolver = AtomTypeParamResolver.from_database(
        default_database.chemical, torch_device
    )

    ljlk_params = LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, device=torch_device
    )

    ljlk_energy = LJLKEnergy(
        atom_type_resolver=resolver,
        type_params=ljlk_params.type_params,
        global_params=ljlk_params.global_params,
        # atom_type_index=ljlk_params.atom_type_index,
        device=torch_device,
    )

    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(rt_list, torch_device)

    for rt in rt_list:
        ljlk_energy.setup_block_type(rt)
        assert hasattr(rt, "ljlk_heavy_atoms_in_tile")
        assert hasattr(rt, "ljlk_n_heavy_atoms_in_tile")
    ljlk_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "ljlk_heavy_atoms_in_tile")
    assert hasattr(pbt, "ljlk_n_heavy_atoms_in_tile")


def test_create_neighbor_list(ubq_res, default_database, torch_device):
    #
    # torch_device = torch.device("cpu")
    resolver = AtomTypeParamResolver.from_database(
        default_database.chemical, torch_device
    )
    ljlk_params = LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, device=torch_device
    )

    ljlk_energy = LJLKEnergy(
        atom_type_resolver=resolver,
        type_params=ljlk_params.type_params,
        global_params=ljlk_params.global_params,
        # atom_type_index=ljlk_params.atom_type_index,
        device=torch_device,
    )

    p1 = Pose.from_residues_one_chain(ubq_res[:4], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:6], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)

    # for i, rt in enumerate(poses.packed_block_types.active_block_types):
    #     for j, atom in enumerate(rt.atoms):
    #         print(rt.name, j, atom.name)
    # return

    # nab the ca coords for these residues
    bounding_spheres = numpy.full((2, 6, 4), numpy.nan, dtype=numpy.float32)
    for i in range(2):
        for j in range(4 if i == 0 else 6):
            bounding_spheres[i, j, :3] = ubq_res[j].coords[2, :]
    bounding_spheres[:, :, 3] = 3.0
    bounding_spheres = torch.tensor(
        bounding_spheres, dtype=torch.float32, device=torch_device
    )

    neighbor_list = ljlk_energy.create_block_neighbor_lists(poses, bounding_spheres)

    # check that the listed neighbors are in striking distance
    for i in range(neighbor_list.shape[0]):
        for j in range(neighbor_list.shape[1]):
            j_coord = bounding_spheres[i, j, 0:3]
            for k in range(neighbor_list.shape[2]):
                ijk_neighbor = neighbor_list[i, j, k]
                if ijk_neighbor == -1:
                    continue
                k_coord = bounding_spheres[i, ijk_neighbor, 0:3]
                dist = torch.norm(j_coord - k_coord)
                assert dist < 6 + ljlk_energy.global_params.max_dis

    # check that any pair of residues in striking distance is
    # listed as a neighbor
    for i in range(2):
        n_res = 4 if i == 0 else 6
        for j in range(n_res):
            j_count = 0
            j_coord = bounding_spheres[i, j, 0:3]
            for k in range(n_res):
                k_coord = bounding_spheres[i, k, 0:3]
                dis = torch.norm(j_coord - k_coord)
                if dis < 6 + ljlk_energy.global_params.max_dis:
                    assert neighbor_list[i, j, j_count] == k
                    j_count += 1
            for k in range(j_count, 6):
                assert neighbor_list[i, j, k] == -1


def test_inter_module(ubq_res, default_database, torch_device):
    #
    # torch_device = torch.device("cpu")
    resolver = AtomTypeParamResolver.from_database(
        default_database.chemical, torch_device
    )
    ljlk_params = LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, device=torch_device
    )

    ljlk_energy = LJLKEnergy(
        atom_type_resolver=resolver,
        type_params=ljlk_params.type_params,
        global_params=ljlk_params.global_params,
        # atom_type_index=ljlk_params.atom_type_index,
        device=torch_device,
    )

    p1 = Pose.from_residues_one_chain(ubq_res[:4], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:6], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)

    # nab the ca coords for these residues
    bounding_spheres = numpy.full((2, 6, 4), numpy.nan, dtype=numpy.float32)
    for i in range(2):
        for j in range(4 if i == 0 else 6):
            bounding_spheres[i, j, :3] = ubq_res[j].coords[2, :]
    bounding_spheres[:, :, 3] = 3.0
    bounding_spheres = torch.tensor(
        bounding_spheres, dtype=torch.float32, device=torch_device
    )

    # five trajectories for each system
    context_system_ids = torch.div(
        torch.arange(10, dtype=torch.int32, device=torch_device), 5
    )

    weights = {"lj": 1.0, "lk": 1.0}
    ljlk_energy.setup_packed_block_types(poses.packed_block_types)
    ljlk_energy.setup_poses(poses)
    inter_module = ljlk_energy.inter_module(
        poses.packed_block_types, poses, context_system_ids, bounding_spheres, weights
    )

    max_n_atoms = poses.packed_block_types.max_n_atoms
    # ok, let's create the contexts
    context_coords = torch.zeros(
        (10, 6, max_n_atoms, 3), dtype=torch.float32, device=torch_device
    )
    context_coords[:5, :, :, :] = torch.tensor(
        poses.coords[0:1, :, :, :], device=torch_device
    )
    context_coords[5:, :, :, :] = torch.tensor(
        poses.coords[1:2, :, :, :], device=torch_device
    )

    context_block_type = torch.zeros((10, 6), dtype=torch.int32, device=torch_device)
    context_block_type[:5, :] = torch.tensor(
        poses.block_type_ind[0:1, :], device=torch_device
    )
    context_block_type[5:, :] = torch.tensor(
        poses.block_type_ind[1:2, :], device=torch_device
    )

    alternate_coords = torch.zeros(
        (20, max_n_atoms, 3), dtype=torch.float32, device=torch_device
    )
    alternate_coords[:10, :, :] = torch.tensor(
        poses.coords[0:1, 1:2, :], device=torch_device
    )
    alternate_coords[10:, :, :] = torch.tensor(
        poses.coords[1:2, 3:4, :], device=torch_device
    )

    alternate_ids = torch.zeros((20, 3), dtype=torch.int32, device=torch_device)
    alternate_ids[:, 0] = torch.div(torch.arange(20, dtype=torch.int32), 2)
    alternate_ids[:10, 1] = 1
    alternate_ids[10:, 1] = 3
    alternate_ids[:10, 2] = torch.tensor(
        poses.block_type_ind[0, 1], device=torch_device
    )
    alternate_ids[10:, 2] = torch.tensor(
        poses.block_type_ind[1, 3], device=torch_device
    )

    # TEMP! Just score one residue
    # alternate_coords = alternate_coords[15:16]
    # alternate_ids = alternate_ids[15:16]

    def run_once():
        rpes = inter_module.go(
            context_coords, context_block_type, alternate_coords, alternate_ids
        )
        assert rpes is not None
        print()
        print(rpes)

    run_once()
    run_once()

    # rpes2 = inter_module.go(
    #     context_coords, context_block_type, alternate_coords, alternate_ids
    # )
    # assert rpes2 is not None
    # print(rpes2)


@pytest.mark.benchmark(group="time_rpe")
@pytest.mark.parametrize("n_alts", [2])
@pytest.mark.parametrize("n_traj", [1])
def test_inter_module_timing(benchmark, ubq_res, default_database, n_alts, n_traj):
    # n_traj = 100
    n_poses = 100
    # n_alts = 10

    #
    torch_device = torch.device("cuda")
    resolver = AtomTypeParamResolver.from_database(
        default_database.chemical, torch_device
    )
    ljlk_params = LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, device=torch_device
    )

    ljlk_energy = LJLKEnergy(
        atom_type_resolver=resolver,
        type_params=ljlk_params.type_params,
        global_params=ljlk_params.global_params,
        # atom_type_index=ljlk_params.atom_type_index,
        device=torch_device,
    )

    p1 = Pose.from_residues_one_chain(ubq_res, torch_device)
    nres = p1.coords.shape[0]
    poses = Poses.from_poses([p1] * n_poses, torch_device)

    one_bounding_sphere_set = numpy.full((1, nres, 4), numpy.nan, dtype=numpy.float32)
    for i in range(nres):
        atnames = set([at.name for at in ubq_res[i].residue_type.atoms])
        cb = ubq_res[i].coords[5, :] if "CB" in atnames else ubq_res[i].coords[2, :]
        max_cb_dist = torch.max(
            torch.norm(
                torch.tensor(
                    ubq_res[i].coords[2:3, :] - ubq_res[i].coords[:, :],
                    dtype=torch.float32,
                    device=torch_device,
                ),
                dim=1,
            )
        )
        one_bounding_sphere_set[0, i, :3] = cb
        one_bounding_sphere_set[0, i, 3] = max_cb_dist.item()
    # print("one bounding sphere set")
    # print(one_bounding_sphere_set)

    # nab the ca coords for these residues
    bounding_spheres = numpy.repeat(one_bounding_sphere_set, n_poses, axis=0)
    bounding_spheres = torch.tensor(
        bounding_spheres, dtype=torch.float32, device=torch_device
    )

    # n_traj trajectories for each system
    context_system_ids = torch.div(
        torch.arange(n_traj * n_poses, dtype=torch.int32, device=torch_device), n_traj
    )

    weights = {"lj": 1.0, "lk": 1.0}
    for bt in poses.packed_block_types.active_block_types:
        ljlk_energy.setup_block_type(bt)
    ljlk_energy.setup_packed_block_types(poses.packed_block_types)
    ljlk_energy.setup_poses(poses)
    inter_module = ljlk_energy.inter_module(
        poses.packed_block_types, poses, context_system_ids, bounding_spheres, weights
    )

    max_n_atoms = poses.packed_block_types.max_n_atoms
    # ok, let's create the contexts
    context_coords = torch.zeros(
        (n_traj * n_poses, nres, max_n_atoms, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    context_coords[:, :, :, :] = torch.tensor(
        poses.coords[0:1, :, :, :], device=torch_device
    )

    context_block_type = torch.zeros(
        (n_traj * n_poses, nres), dtype=torch.int32, device=torch_device
    )
    context_block_type[:, :] = torch.tensor(
        poses.block_type_ind[0:1, :], device=torch_device
    )

    alternate_coords = torch.zeros(
        (n_alts * n_traj * n_poses, max_n_atoms, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    which_block = torch.remainder(
        torch.div(
            torch.arange(
                n_alts * n_traj * n_poses, dtype=torch.int32, device=torch_device
            ),
            n_alts,
        ),
        nres,
    )
    alternate_coords[:, :, :] = torch.tensor(
        poses.coords[0:1, which_block.cpu().numpy(), :, :], device=torch_device
    )

    alternate_ids = torch.zeros(
        (n_alts * n_traj * n_poses, 3), dtype=torch.int32, device=torch_device
    )
    alternate_ids[:, 0] = torch.div(
        torch.arange(n_alts * n_traj * n_poses, dtype=torch.int32, device=torch_device),
        n_alts,
    )
    alternate_ids[:, 1] = which_block
    alternate_ids[:, 2] = torch.tensor(
        poses.block_type_ind[0, which_block.cpu().numpy()], device=torch_device
    )

    @benchmark
    def run():
        rpes = inter_module.go(
            context_coords, context_block_type, alternate_coords, alternate_ids
        )
        return rpes

    vals = run
    assert vals is not None
