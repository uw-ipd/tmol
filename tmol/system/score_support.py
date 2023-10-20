import math
import torch

from tmol.types.torch import Tensor


def kincoords_to_coords(
    kincoords, kinforest, system_size
) -> Tensor[torch.float][:, :, 3]:
    """System cartesian atomic coordinates."""

    coords = torch.full(
        (system_size, 3),
        math.nan,
        dtype=kincoords.dtype,
        layout=kincoords.layout,
        device=kincoords.device,
        requires_grad=False,
    )

    idIdx = kinforest.id[1:].to(dtype=torch.long)
    coords[idIdx] = kincoords[1:]

    return coords.to(torch.float)[None, ...]
