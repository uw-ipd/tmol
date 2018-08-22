import torch
from math import nan

from tmol.score.interatomic_distance import Sphere, SphereDistance, IntraLayerAtomPairs


def test_sphere_from_coord_blocks(multilayer_test_coords, multilayer_test_offsets):
    """Sphere calculates mean and radius of layers, respecting nans."""

    ### Block size 8
    blocks = Sphere.from_coord_blocks(8, multilayer_test_coords)
    assert blocks.shape == (4, 3)

    assert blocks.center.shape == (4, 3, 3)
    torch.testing.assert_allclose(blocks.center, multilayer_test_offsets)

    assert blocks.radius.shape == (4, 3)
    torch.testing.assert_allclose(blocks.radius, torch.tensor([1.0]).expand((4, 3)))

    ### Block size 4
    blocks = Sphere.from_coord_blocks(4, multilayer_test_coords)
    assert blocks.shape == (4, 6)

    assert blocks.center.shape == (4, 6, 3)
    torch.testing.assert_allclose(
        blocks.center,
        # Interleave test offsets on 2nd dimension
        torch.stack([multilayer_test_offsets] * 2, 2).view(4, 6, 3),
    )

    assert blocks.radius.shape == (4, 6)
    torch.testing.assert_allclose(blocks.radius, torch.tensor([1.0]).expand((4, 6)))

    ### Block size 2, every 4th block is nan
    blocks = Sphere.from_coord_blocks(2, multilayer_test_coords)
    assert blocks.shape == (4, 12)

    assert blocks.center.shape == (4, 12, 3)
    torch.testing.assert_allclose(
        blocks.center,
        # Interleave test offsets on 2nd dimension
        torch.stack(
            [multilayer_test_offsets] * 3
            + [torch.full_like(multilayer_test_offsets, nan)],
            2,
        ).view(4, 12, 3),
    )

    assert blocks.radius.shape == (4, 12)
    torch.testing.assert_allclose(
        blocks.radius, torch.tensor([1.0, 1.0, 1.0, 0.0] * 3)[None, :].expand((4, 12))
    )


def test_blocked_interatomic_distance_nulls(multilayer_test_coords):
    """Test that interatomic distance properly handles fully null blocks."""
    null_padded = multilayer_test_coords.new_full((4, 24 + 8, 3), nan)
    null_padded[:, :24, :] = multilayer_test_coords

    ilap = IntraLayerAtomPairs.for_coord_blocks(
        block_size=8,
        coord_blocks=Sphere.from_coord_blocks(8, multilayer_test_coords),
        threshold_distance=4.0,
    )

    np_ilap = IntraLayerAtomPairs.for_coord_blocks(
        block_size=8,
        coord_blocks=Sphere.from_coord_blocks(8, null_padded),
        threshold_distance=4.0,
    )

    assert (ilap.inds == np_ilap.inds).all()


def test_blocked_interatomic_distance_layered(multilayer_test_coords):
    """Sphere-radius calculation uses triangle inequality for interaction distance."""

    threshold_distance = 6.0

    # fmt: off
    dense_expected_block_interactions = torch.Tensor([
        [  # 0-------1------2
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        [  # 2-------1------0
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        [  # ----0---1---2---
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        [  # -------012------
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    ]).to(dtype=torch.uint8)
    # fmt: on

    blocks = Sphere.from_coord_blocks(8, multilayer_test_coords)
    bdist = SphereDistance.for_spheres(blocks[:, None, :], blocks[:, :, None])

    torch.testing.assert_allclose(
        (bdist.min_dist < threshold_distance).to(torch.float),
        dense_expected_block_interactions.to(torch.float),
    )
