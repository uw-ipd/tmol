import numpy
import torch

from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase
from tmol.tests.score.common.test_energy_term import pose_stack_from_pdb_and_resnums
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.pose.constraint_set import ConstraintSet


def test_add_constraints(
    ubq_pdb,
    default_database,
    torch_device: torch.device,
    resnums=None,
    update_baseline=False,
    atol=1e-5,
    rtol=1e-3,
):
    n_poses = 10

    resnums = [(0, 5)]
    p1 = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, resnums)
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    constraints = ConstraintSet(device=torch_device)

    def harmfunc(atoms, params):
        atoms1 = atoms[:, 0]
        atoms2 = atoms[:, 1]
        diff = atoms1 - atoms2
        return (diff.pow(2).sum(1).sqrt() - params[:, 0]) ** 2

    def constfn10(atoms, params):
        return torch.full(
            (atoms.size(0),), 10, dtype=torch.float32, device=atoms.device
        )

    def constfn5(atoms, params):
        return torch.full((atoms.size(0),), 3, dtype=torch.float32, device=atoms.device)

    cnstr_atoms = torch.full((3, 2, 3), 0, dtype=torch.int32, device=torch_device)

    cnstr_atoms[0, 0, 0] = 0
    cnstr_atoms[0, 0, 1] = 0
    cnstr_atoms[0, 0, 2] = 0
    cnstr_atoms[0, 1, 0] = 0
    cnstr_atoms[0, 1, 1] = 1
    cnstr_atoms[0, 1, 2] = 1

    cnstr_atoms[1, 0, 0] = 2
    cnstr_atoms[1, 0, 1] = 0
    cnstr_atoms[1, 0, 2] = 0
    cnstr_atoms[1, 1, 0] = 2
    cnstr_atoms[1, 1, 1] = 1
    cnstr_atoms[1, 1, 2] = 1

    cnstr_atoms[2, 0, 0] = 1
    cnstr_atoms[2, 0, 1] = 0
    cnstr_atoms[2, 0, 2] = 0
    cnstr_atoms[2, 1, 0] = 1
    cnstr_atoms[2, 1, 1] = 1
    cnstr_atoms[2, 1, 2] = 1

    constraints.add_constraints(constfn10, cnstr_atoms)

    cnstr_atoms[0, 0, 0] = 0
    cnstr_atoms[0, 0, 1] = 1
    cnstr_atoms[0, 0, 2] = 0
    cnstr_atoms[0, 1, 0] = 0
    cnstr_atoms[0, 1, 1] = 2
    cnstr_atoms[0, 1, 2] = 1

    cnstr_atoms[1, 0, 0] = 2
    cnstr_atoms[1, 0, 1] = 0
    cnstr_atoms[1, 0, 2] = 0
    cnstr_atoms[1, 1, 0] = 2
    cnstr_atoms[1, 1, 1] = 3
    cnstr_atoms[1, 1, 2] = 1

    cnstr_atoms[2, 0, 0] = 1
    cnstr_atoms[2, 0, 1] = 0
    cnstr_atoms[2, 0, 2] = 0
    cnstr_atoms[2, 1, 0] = 1
    cnstr_atoms[2, 1, 1] = 2
    cnstr_atoms[2, 1, 2] = 1

    """cnstr_atoms[0, 0, 0] = 0
    cnstr_atoms[0, 0, 1] = 0
    cnstr_atoms[0, 0, 2] = 0
    cnstr_atoms[0, 1, 0] = 1
    cnstr_atoms[0, 1, 1] = 1
    cnstr_atoms[0, 1, 2] = 1

    cnstr_atoms[1, 0, 0] = 1
    cnstr_atoms[1, 0, 1] = 0
    cnstr_atoms[1, 0, 2] = 0
    cnstr_atoms[1, 1, 0] = 2
    cnstr_atoms[1, 1, 1] = 1
    cnstr_atoms[1, 1, 2] = 1

    cnstr_atoms[2, 0, 0] = 0
    cnstr_atoms[2, 0, 1] = 0
    cnstr_atoms[2, 0, 2] = 0
    cnstr_atoms[2, 1, 0] = 1
    cnstr_atoms[2, 1, 1] = 1
    cnstr_atoms[2, 1, 2] = 1"""

    constraints.add_constraints(constfn5, cnstr_atoms)

    print(constraints.constraint_function_inds)
    print(constraints.constraint_atoms)
    print(constraints.constraint_functions)

    pn.constraint_set = constraints

    energy_term = ConstraintEnergyTerm(param_db=default_database, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        energy_term.setup_block_type(bt)
    energy_term.setup_packed_block_types(pn.packed_block_types)
    energy_term.setup_poses(pn)

    pose_scorer = energy_term.render_whole_pose_scoring_module(pn)
    # pose_scorer = get_pose_scorer(pn, default_database, torch_device)

    coords = torch.nn.Parameter(pn.coords.clone())
    # scores = pose_scorer(coords).cpu().detach().numpy()
    scores = pose_scorer(coords, output_block_pair_energies=True).cpu().detach().numpy()

    print(scores)

    # gold_vals = cls.get_test_baseline_data(cls.test_whole_pose_scoring_10.__name__)

    # assert_allclose(gold_vals, scores, atol, rtol)


class TestConstraintEnergyTerm(EnergyTermTestBase):
    energy_term_class = ConstraintEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 10)]
        return super().test_whole_pose_scoring_10(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=True,
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        ubq_pdb,
        default_database,
        torch_device: torch.device,
    ):
        return super().test_whole_pose_scoring_jagged(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 4)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb, default_database, torch_device, resnums=resnums, eps=1e-3
        )

    @classmethod
    def test_block_scoring_matches_whole_pose_scoring(
        cls, ubq_pdb, default_database, torch_device
    ):
        return super().test_block_scoring_matches_whole_pose_scoring(
            ubq_pdb, default_database, torch_device
        )

    @classmethod
    def test_block_scoring(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 5)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=True,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(0, 4)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb, default_database, torch_device, resnums, eps=1e-3
        )
