import pytest
import torch
from math import nan


@pytest.fixture
def multilayer_test_coords():
    """A stacked test system with random coordinate clusters.

    clusters:
        8 coordinate block, 6 populated w/ unit vectors x,y,z,-x,-y,-z

    layers:
        0-------1------2
        1-------2------0
        ----0---1---2---
        -------012------
    """

    cluster_coords = torch.Tensor(
        [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (nan, nan, nan),
            (nan, nan, nan),
        ]
    )

    layout = torch.Tensor([[-8, 0, 8], [8, 0, -8], [-4, 0, 4], [-1, 0, 1]])

    # Convert to coordinates offsets along the x axis.
    offsets = layout[..., None] * torch.Tensor([1, 0, 0])
    assert offsets.shape == layout.shape + (3,)
    assert not (offsets[..., 0] == 0).all()
    assert (offsets[..., 1] == 0).all()
    assert (offsets[..., 2] == 0).all()

    coords = offsets[:, :, None, :] + cluster_coords
    assert coords.shape == layout.shape + cluster_coords.shape
    assert ((coords[0, 0, :6] == cluster_coords[:6] + torch.Tensor([-8, 0, 0]))).all()

    return coords.reshape((4, 8 * 3, 3))
