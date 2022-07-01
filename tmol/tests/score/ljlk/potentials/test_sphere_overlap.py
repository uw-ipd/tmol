import pytest

import torch

from tmol.utility.cpp_extension import load, relpaths, modulename

from tmol.tests.torch import requires_cuda


@requires_cuda
@pytest.fixture
def extension():
    return load(modulename(f"{__name__}.cuda"), relpaths(__file__, "sphere_overlap.cu"))


@requires_cuda
def test_compute_block_spheres(extension):
    """test that spheres are correctly computed given a tensor of coordinates"""
    dev = torch.device("cuda")
    coords = some_coords()[None, :, :].to(dev)
    block_type_n_atoms = torch.tensor([5, 6, 4, 3, 6, 4], dtype=torch.int32, device=dev)
    pose_stack_block_type = torch.tensor(
        [[0, 1, 2, 1, 0, 2, 1]], dtype=torch.int32, device=dev
    )
    pose_stack_block_coord_offset = torch.tensor(
        [[0, 5, 11, 15, 21, 26, 30]], dtype=torch.int32, device=dev
    )
    spheres = extension.compute_block_spheres_float(
        coords, pose_stack_block_coord_offset, pose_stack_block_type, block_type_n_atoms
    )

    gold_com = torch.zeros((1, 7, 3), dtype=torch.float32, device=dev)
    block_start = torch.zeros((coords.shape[1],), dtype=torch.int32, device=dev)
    block_start[pose_stack_block_coord_offset.type(torch.int64)] = 1

    block_id = torch.cumsum(block_start, 0) - 1
    gold_com.index_add_(1, block_id, coords)
    pose_block_natoms = block_type_n_atoms[pose_stack_block_type.type(torch.int64)]
    gold_com = gold_com / pose_block_natoms[:, :, None]

    som_for_block_atom = gold_com[:, block_id, :]
    dist = torch.norm(coords - som_for_block_atom, dim=2)

    max_n_atoms = torch.max(block_type_n_atoms).item()
    dist_by_res = torch.zeros((7, max_n_atoms), dtype=torch.float32, device=dev)
    ar = torch.arange(max_n_atoms, dtype=torch.int32, device=dev)[None, :]
    at_is_real = ar < pose_block_natoms[0, :, None]

    dist_by_res[at_is_real] = dist.flatten()
    gold_sphere_radii, _ = torch.max(dist_by_res, dim=1)

    gold_spheres = torch.cat(
        (gold_com.reshape((-1, 3)), gold_sphere_radii.reshape((-1, 1))), dim=1
    ).reshape((1, 7, 4))

    torch.testing.assert_allclose(gold_spheres, spheres, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_compute_block_spheres2(extension):
    """test that spheres are correctly computed given a tensor of coordinates"""
    dev = torch.device("cuda")
    coords = some_coords()[None, :, :].to(dev)
    block_type_n_atoms = torch.tensor([5, 6, 4, 3, 6, 4], dtype=torch.int32, device=dev)
    pose_stack_block_type = torch.tensor(
        [[0, 1, 2, 1, 0, 2, 1]], dtype=torch.int32, device=dev
    )
    pose_stack_block_coord_offset = torch.tensor(
        [[0, 5, 11, 15, 21, 26, 30]], dtype=torch.int32, device=dev
    )

    def double_stack_depth(x):
        return torch.cat((x, x), dim=0)

    coords2 = double_stack_depth(coords)
    block_type_n_atoms2 = double_stack_depth(block_type_n_atoms)
    pose_stack_block_type2 = double_stack_depth(pose_stack_block_type)
    pose_stack_block_coord_offset2 = double_stack_depth(pose_stack_block_coord_offset)

    spheres = extension.compute_block_spheres_float(
        coords2,
        pose_stack_block_coord_offset2,
        pose_stack_block_type2,
        block_type_n_atoms2,
    )

    gold_com = torch.zeros((1, 7, 3), dtype=torch.float32, device=dev)
    block_start = torch.zeros((coords.shape[1],), dtype=torch.int32, device=dev)
    block_start[pose_stack_block_coord_offset.type(torch.int64)] = 1

    block_id = torch.cumsum(block_start, 0) - 1
    gold_com.index_add_(1, block_id, coords)
    pose_block_natoms = block_type_n_atoms[pose_stack_block_type.type(torch.int64)]
    gold_com = gold_com / pose_block_natoms[:, :, None]

    som_for_block_atom = gold_com[:, block_id, :]
    dist = torch.norm(coords - som_for_block_atom, dim=2)

    max_n_atoms = torch.max(block_type_n_atoms).item()
    dist_by_res = torch.zeros((7, max_n_atoms), dtype=torch.float32, device=dev)
    ar = torch.arange(max_n_atoms, dtype=torch.int32, device=dev)[None, :]
    at_is_real = ar < pose_block_natoms[0, :, None]

    dist_by_res[at_is_real] = dist.flatten()
    gold_sphere_radii, _ = torch.max(dist_by_res, dim=1)

    gold_spheres = torch.cat(
        (gold_com.reshape((-1, 3)), gold_sphere_radii.reshape((-1, 1))), dim=1
    ).reshape((1, 7, 4))
    gold_spheres2 = double_stack_depth(gold_spheres)

    torch.testing.assert_allclose(gold_spheres2, spheres, rtol=1e-5, atol=1e-5)


def some_coords():
    return torch.tensor(
        [
            [27.340, 24.430, 2.614],
            [26.266, 25.413, 2.842],
            [26.913, 26.639, 3.531],
            [27.886, 26.463, 4.263],
            [25.112, 24.880, 3.649],  # 5
            [25.353, 24.860, 5.134],
            [23.930, 23.959, 5.904],
            [24.447, 23.984, 7.620],
            [27.282, 23.521, 3.027],
            [25.864, 25.717, 1.875],  # 10
            [24.227, 25.486, 3.461],
            [24.886, 23.861, 3.332],
            [26.298, 24.359, 5.342],
            [25.421, 25.882, 5.505],
            [23.700, 23.479, 8.233],  # 15
            [25.405, 23.472, 7.719],
            [24.552, 25.017, 7.954],
            [26.335, 27.770, 3.258],
            [26.850, 29.021, 3.898],
            [26.100, 29.253, 5.202],  # 20
            [24.865, 29.024, 5.330],
            [26.733, 30.148, 2.905],
            [26.882, 31.546, 3.409],
            [26.786, 32.562, 2.270],
            [27.783, 33.160, 1.870],  # 25
            [25.562, 32.733, 1.806],
            [25.549, 27.828, 2.627],
            [27.898, 28.870, 4.158],
            [27.488, 30.028, 2.128],
            [25.757, 30.107, 2.422],  # 30
            [26.089, 31.748, 4.128],
            [27.856, 31.647, 3.889],
            [25.395, 33.378, 1.059],
            [24.801, 32.218, 2.200],
            [26.849, 29.656, 6.217],  # 35
            [26.235, 30.058, 7.497],
        ],
        dtype=torch.float32,
    )
