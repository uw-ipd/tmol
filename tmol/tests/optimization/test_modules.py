from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.optimization.modules import CartesianEnergyNetwork, TorsionalEnergyNetwork

from tmol.system.kinematics import KinematicDescription

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.coords import coords_for
from tmol.score.modules.ljlk import LJScore, LKScore
from tmol.score.modules.lk_ball import LKBallScore
from tmol.score.modules.elec import ElecScore
from tmol.score.modules.cartbonded import CartBondedScore
from tmol.score.modules.dunbrack import DunbrackScore
from tmol.score.modules.hbond import HBondScore
from tmol.score.modules.rama import RamaScore
from tmol.score.modules.omega import OmegaScore

from tmol.score.device import TorchDevice


def test_cart_network_min(ubq_system, torch_device):
    score_system = ScoreSystem.build_for(
        ubq_system,
        {
            LJScore,
            LKScore,
            LKBallScore,
            ElecScore,
            CartBondedScore,
            DunbrackScore,
            HBondScore,
            RamaScore,
            OmegaScore,
        },
        weights={
            "lj": 1.0,
            "lk": 1.0,
            "lk_ball": 1.0,
            "elec": 1.0,
            "cartbonded": 1.0,
            "dunbrack": 1.0,
            "hbond": 1.0,
            "rama": 1.0,
            "omega": 1.0,
        },
    )
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
    score_system = ScoreSystem.build_for(
        ubq_system,
        {
            LJScore,
            LKScore,
            LKBallScore,
            ElecScore,
            CartBondedScore,
            DunbrackScore,
            HBondScore,
            RamaScore,
            OmegaScore,
        },
        weights={
            "lj": 1.0,
            "lk": 1.0,
            "lk_ball": 1.0,
            "elec": 1.0,
            "cartbonded": 1.0,
            "dunbrack": 1.0,
            "hbond": 1.0,
            "rama": 1.0,
            "omega": 1.0,
        },
    )
    coords = coords_for(ubq_system, score_system)

    # Initialize kinematic tree for the system
    sys_kin = KinematicDescription.for_system(
        ubq_system.bonds, ubq_system.torsion_metadata
    )
    tkintree = sys_kin.kintree.to(torch_device)
    tdofmetadata = sys_kin.dof_metadata.to(torch_device)
    # compute dofs from xyzs
    kincoords = sys_kin.extract_kincoords(ubq_system.coords).to(torch_device)

    model = TorsionalEnergyNetwork(
        score_system, kincoords, tkintree, tdofmetadata, ubq_system.system_size
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
