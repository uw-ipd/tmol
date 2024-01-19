import numpy
import torch
import pytest
from torch._C import device

from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase
from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device):
    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)

    assert ljlk_energy.type_params.lj_radius.device == torch_device
    assert ljlk_energy.global_params.max_dis.device == torch_device


def test_annotate_heavy_ats_in_tile(ubq_res, default_database, torch_device):
    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)

    rt_list = residue_types_from_residues(ubq_res)
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, rt_list, torch_device
    )

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
    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)

    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:4], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:6], torch_device
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

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


def test_render_inter_module(ubq_res, default_database, torch_device):
    #
    # torch_device = torch.device("cpu")

    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)

    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:4], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:6], torch_device
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

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
    context_system_ids = torch.floor_divide(
        torch.arange(10, dtype=torch.int32, device=torch_device), 5
    )

    weights = {"lj": 1.0, "lk": 1.0}
    for bt in poses.packed_block_types.active_block_types:
        ljlk_energy.setup_block_type(bt)
    ljlk_energy.setup_packed_block_types(poses.packed_block_types)
    ljlk_energy.setup_poses(poses)
    inter_module = ljlk_energy.render_inter_module(
        poses.packed_block_types, poses, context_system_ids, bounding_spheres, weights
    )

    max_n_atoms_per_block = poses.packed_block_types.max_n_atoms
    # ok, let's create the contexts
    context_coords = torch.zeros(
        (10, 6, max_n_atoms_per_block, 3), dtype=torch.float32, device=torch_device
    )
    # this should be fine
    poses_expanded_coords, real_expanded_pose_ats = poses.expand_coords()
    context_coords[:5, :, :, :] = poses_expanded_coords[0:1]
    context_coords[5:, :, :, :] = poses_expanded_coords[1:2]
    context_coords = context_coords.view(10, -1, 3)
    context_coord_offsets = max_n_atoms_per_block * torch.remainder(
        torch.arange(60, dtype=torch.int32, device=torch_device).view(10, 6), 6
    )

    context_block_type = torch.zeros((10, 6), dtype=torch.int32, device=torch_device)
    context_block_type[:5, :] = torch.tensor(
        poses.block_type_ind[0:1, :], device=torch_device
    )
    context_block_type[5:, :] = torch.tensor(
        poses.block_type_ind[1:2, :], device=torch_device
    )

    alternate_coords = torch.zeros(
        (20, max_n_atoms_per_block, 3), dtype=torch.float32, device=torch_device
    )
    alternate_coords[:10, :, :] = poses_expanded_coords[0:1, 1:2, :]
    alternate_coords[10:, :, :] = poses_expanded_coords[1:2, 3:4, :]
    alternate_coords = alternate_coords.view(-1, 3)
    alternate_coord_offsets = max_n_atoms_per_block * torch.arange(
        20, dtype=torch.int32, device=torch_device
    )

    alternate_ids = torch.zeros((20, 3), dtype=torch.int32, device=torch_device)
    alternate_ids[:, 0] = torch.floor_divide(torch.arange(20, dtype=torch.int32), 2)
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
            context_coords,
            context_coord_offsets,
            context_block_type,
            alternate_coords,
            alternate_coord_offsets,
            alternate_ids,
        )
        assert rpes is not None
        # print()
        # print(rpes)

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
@pytest.mark.parametrize("n_poses", [10, 30, 100])
def test_inter_module_timing(
    benchmark, ubq_res, default_database, n_alts, n_traj, n_poses, torch_device
):
    # n_traj = 100
    # n_poses = 100
    # n_alts = 10

    # this is slow on CPU?
    # torch_device = torch.device("cuda")

    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)

    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res, torch_device
    )
    nres = p1.max_n_blocks
    poses = PoseStackBuilder.from_poses([p1] * n_poses, torch_device)

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
    context_system_ids = torch.floor_divide(
        torch.arange(n_traj * n_poses, dtype=torch.int32, device=torch_device), n_traj
    )

    weights = {"lj": 1.0, "lk": 1.0}
    for bt in poses.packed_block_types.active_block_types:
        ljlk_energy.setup_block_type(bt)
    ljlk_energy.setup_packed_block_types(poses.packed_block_types)
    ljlk_energy.setup_poses(poses)
    inter_module = ljlk_energy.render_inter_module(
        poses.packed_block_types, poses, context_system_ids, bounding_spheres, weights
    )

    max_n_atoms_per_block = poses.packed_block_types.max_n_atoms
    # ok, let's create the contexts
    context_coords = torch.zeros(
        (n_traj * n_poses, nres, max_n_atoms_per_block, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    poses_expanded_coords, real_expanded_pose_ats = poses.expand_coords()
    context_coords[:, :, :, :] = poses_expanded_coords[0:1, :, :, :]
    context_coords = context_coords.view(n_poses, -1, 3)
    context_coord_offsets = max_n_atoms_per_block * torch.remainder(
        torch.arange(nres * n_poses, dtype=torch.int32, device=torch_device).view(
            n_poses, nres
        ),
        nres,
    )

    context_block_type = torch.zeros(
        (n_traj * n_poses, nres), dtype=torch.int32, device=torch_device
    )
    context_block_type[:, :] = torch.tensor(
        poses.block_type_ind[0:1, :], device=torch_device
    )

    alternate_coords = torch.zeros(
        (n_alts * n_traj * n_poses, max_n_atoms_per_block, 3),
        dtype=torch.float32,
        device=torch_device,
    )
    which_block = torch.remainder(
        torch.floor_divide(
            torch.arange(
                n_alts * n_traj * n_poses, dtype=torch.int32, device=torch_device
            ),
            n_alts,
        ),
        nres,
    )
    alternate_coords[:, :, :] = poses_expanded_coords[
        0:1, which_block.type(torch.int64), :, :
    ]
    alternate_coords = alternate_coords.view(-1, 3)
    alternate_coord_offsets = max_n_atoms_per_block * torch.arange(
        n_alts * n_traj * n_poses, dtype=torch.int32, device=torch_device
    )

    alternate_ids = torch.zeros(
        (n_alts * n_traj * n_poses, 3), dtype=torch.int32, device=torch_device
    )
    alternate_ids[:, 0] = torch.floor_divide(
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
            context_coords,
            context_coord_offsets,
            context_block_type,
            alternate_coords,
            alternate_coord_offsets,
            alternate_ids,
        )
        return rpes

    vals = run
    assert vals is not None


def test_whole_pose_scoring_module_smoke(rts_ubq_res, default_database, torch_device):
    gold_vals = numpy.array([[-9.207252], [1.51558], [3.61822]], dtype=numpy.float32)

    ljlk_energy = LJLKEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_ubq_res[0:4], device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        ljlk_energy.setup_block_type(bt)
    ljlk_energy.setup_packed_block_types(p1.packed_block_types)
    ljlk_energy.setup_poses(p1)

    ljlk_pose_scorer = ljlk_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = ljlk_pose_scorer(
        coords,
    )

    numpy.testing.assert_allclose(gold_vals, scores.cpu().detach().numpy(), atol=1e-4)


class TestLJLKEnergyTerm(EnergyTermTestBase):
    energy_term_class = LJLKEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(
        cls, rts_ubq_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_whole_pose_scoring_10(
            rts_ubq_res, default_database, torch_device, update_baseline
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(
        cls, rts_ubq_res, default_database, torch_device
    ):
        return super().test_whole_pose_scoring_gradcheck(
            rts_ubq_res[0:4], default_database, torch_device
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        rts_ubq_res,
        default_database,
        torch_device: torch.device,
        update_baseline=False,
    ):
        return super().test_whole_pose_scoring_jagged(
            rts_ubq_res, default_database, torch_device, update_baseline
        )

    @classmethod
    def test_block_scoring(
        cls, rts_ubq_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_block_scoring(
            rts_ubq_res[0:4], default_database, torch_device, update_baseline
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, rts_ubq_res, default_database, torch_device
    ):
        return super().test_block_scoring_reweighted_gradcheck(
            rts_ubq_res[0:4],
            default_database,
            torch_device,
            eps=1e-3,
            atol=1e-3,
            nondet_tol=1e-6,
        )
