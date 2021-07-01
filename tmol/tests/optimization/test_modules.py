from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.optimization.modules import CartesianEnergyNetwork, TorsionalEnergyNetwork

from tmol.system.kinematics import KinematicDescription

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.coords import coords_for

from tmol.system.score_support import get_full_score_system_for


def test_cart_network_min(ubq_system, torch_device):
    score_system = get_full_score_system_for(ubq_system)
    coords = coords_for(ubq_system, score_system)

    model = CartesianEnergyNetwork(score_system, coords)
    optimizer = LBFGS_Armijo(model.parameters(), lr=0.8, max_iter=20)

    E0 = score_system.intra_score_only(coords)

    def closure():
        optimizer.zero_grad()

        E = model()
        E.backward()
        return E

    optimizer.step(closure)  # this optimizes coords, the tensor

    E1 = score_system.intra_score_only(coords)
    assert E1 < E0


def test_dof_network_min(ubq_system, torch_device):
    score_system = get_full_score_system_for(ubq_system)
    coords = coords_for(ubq_system, score_system)

    model = TorsionalEnergyNetwork(score_system, ubq_system, torch_device)

    # "kincoords" is for each atom, 9 values,
    # but only 3 for regular atom, 9 for jump
    optimizer = LBFGS_Armijo(model.parameters(), lr=0.8, max_iter=20)

    E0 = score_system.intra_score_only(model.coords())

    def closure():
        optimizer.zero_grad()

        E = model()
        E.backward()
        return E

    optimizer.step(closure)
    E1 = score_system.intra_score_only(model.coords())
    assert E1 < E0
