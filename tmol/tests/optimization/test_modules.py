from torch import BoolTensor

from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.optimization.modules import (
    CartesianEnergyNetwork,
    TorsionalEnergyNetwork,
    torsional_energy_network_from_system,
)

from tmol.system.kinematics import KinematicDescription

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.coords import coords_for

from tmol.system.score_support import get_full_score_system_for


def test_cart_network_min(ubq_system, torch_device):
    score_system = get_full_score_system_for(ubq_system)
    coords = coords_for(ubq_system, score_system)

    model = CartesianEnergyNetwork(score_system, coords)
    optimizer = LBFGS_Armijo(model.parameters(), lr=0.8, max_iter=20)

    E0 = score_system.intra_total(coords)

    def closure():
        optimizer.zero_grad()

        E = model()
        E.backward()
        return E

    optimizer.step(closure)  # this optimizes coords, the tensor

    E1 = score_system.intra_total(coords)
    assert E1 < E0


def test_cart_network_min(ubq_system, torch_device):
    score_system = get_full_score_system_for(ubq_system)
    coords = coords_for(ubq_system, score_system)

    coord_mask = BoolTensor(coords.shape)
    for i in range(coord_mask.shape[1]):
        for j in range(coord_mask.shape[2]):
            coord_mask[0, i, j] = i % 2 and (j + i) % 2

    model = CartesianEnergyNetwork(score_system, coords, coord_mask=coord_mask)
    optimizer = LBFGS_Armijo(model.parameters(), lr=0.8, max_iter=20)

    E0 = score_system.intra_total(coords)

    def closure():
        optimizer.zero_grad()

        E = model()
        E.backward()
        return E

    optimizer.step(closure)  # this optimizes coords, the tensor

    E1 = score_system.intra_total(coords)
    assert E1 < E0


def test_dof_network_min(ubq_system, torch_device):
    score_system = get_full_score_system_for(ubq_system)
    coords = coords_for(ubq_system, score_system)

    model = torsional_energy_network_from_system(score_system, ubq_system, torch_device)

    # "kincoords" is for each atom, 9 values,
    # but only 3 for regular atom, 9 for jump
    optimizer = LBFGS_Armijo(model.parameters(), lr=0.8, max_iter=20)

    E0 = score_system.intra_total(model.coords())

    def closure():
        optimizer.zero_grad()

        E = model()
        E.backward()
        return E

    optimizer.step(closure)
    E1 = score_system.intra_total(model.coords())
    assert E1 < E0


def test_dof_network_min_masked(ubq_system, torch_device):
    score_system = get_full_score_system_for(ubq_system)
    coords = coords_for(ubq_system, score_system)

    sys_kin = KinematicDescription.for_system(
        ubq_system.bonds, ubq_system.torsion_metadata
    )
    kintree = sys_kin.kintree.to(torch_device)
    dofs = sys_kin.extract_kincoords(ubq_system.coords).to(torch_device)
    system_size = ubq_system.system_size

    dof_mask = BoolTensor(dofs.shape)
    for i in range(dof_mask.shape[0]):
        for j in range(dof_mask.shape[1]):
            dof_mask[i, j] = i % 2 and (j + i) % 2

    model = TorsionalEnergyNetwork(
        score_system, dofs, kintree, system_size, dof_mask=dof_mask
    )

    # "kincoords" is for each atom, 9 values,
    # but only 3 for regular atom, 9 for jump
    optimizer = LBFGS_Armijo(model.parameters(), lr=0.8, max_iter=20)

    E0 = score_system.intra_total(model.coords())

    def closure():
        optimizer.zero_grad()

        E = model()
        E.backward()
        return E

    optimizer.step(closure)
    E1 = score_system.intra_total(model.coords())
    assert E1 < E0
