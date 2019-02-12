import numpy
import torch

from tmol.score.bonded_atom import IndexedBonds
from tmol.tests.autograd import gradcheck


def test_water_generation(default_database):
    from tmol.score.lk_ball.potentials.compiled import AttachedWaters

    op = AttachedWaters.from_database(default_database, torch.device("cpu"))

    tensor = torch.FloatTensor

    # test 3: ring acceptor--sp3 acceptor
    coords = tensor(
        [
            [-5.250, -1.595, -2.543],  # SER CB
            [-6.071, -0.619, -3.193],  # SER OG
            [-5.489, 0.060, -3.542],  # SER HG
            [-10.628, 2.294, -1.933],  # HIS CG
            [-9.991, 1.160, -1.435],  # HIS ND1
            [-10.715, 0.960, -0.319],  # HIS CE1
        ]
    )

    bonds = numpy.array([[0, 1], [1, 2], [3, 4], [4, 5]])
    indexed_bonds = IndexedBonds.from_bonds(IndexedBonds.to_directed(bonds))

    atom_type_names = ["CH2", "OH", "Hpol", "CH0", "NhisDDepro", "Caro"]
    atom_types = op.atom_resolver.type_idx(atom_type_names)

    waters = op.apply(coords, atom_types, indexed_bonds)

    assert list((~torch.isnan(waters[..., 0])).sum(dim=-1)) == [0, 3, 0, 0, 1, 0]

    # gradcheck(
    #     lambda coords: op.apply(coords, atom_types, indexed_bonds),
    #     (coords.requires_grad_(True),),
    # )
