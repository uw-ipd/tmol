import numpy
import torch

from tmol.types.functional import validate_args

from .packed import PackedResidueSystem
from tmol.score.types import RealTensor


@validate_args
def system_real_graph_params(
        system: PackedResidueSystem,
        drop_missing_atoms: bool = False,
        requires_grad: bool = True,
):
    bonds = system.bonds
    coords = (
        torch.from_numpy(system.coords).clone()
        .to(RealTensor.dtype)
        .requires_grad_(requires_grad)
    ) # yapf: disable

    atom_types = system.atom_metadata["atom_type"].copy()

    if drop_missing_atoms:
        atom_types[numpy.any(numpy.isnan(system.coords), axis=-1)] = None

    return dict(
        system_size=len(coords),
        bonds=bonds,
        coords=coords,
        atom_types=atom_types,
    )
