import attrs
import numpy
import torch
import math
import pytest

from tmol.pose.constraint_set import ConstraintSet
from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm
from tmol.tests.score.common.test_energy_term import EnergyTermTestBase
from tmol import pose_stack_from_pdb, ScoreFunction, ScoreType


def add_test_constraints_to_pose_stack(pose_stack):
    torch_device = pose_stack.device

    constraints = pose_stack.get_constraint_set()
    if constraints is None:
        constraints = ConstraintSet.create_empty(
            device=torch_device, n_poses=pose_stack.n_poses
        )

    # a distance constraint
    cnstr_atoms = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 3)
    res2_type = pose_stack.block_type(0, 4)
    cnstr_atoms[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47
    cnstr_params[0, 1] = 0.1

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    # repeat to test function caching
    cnstr_atoms = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 4)
    res2_type = pose_stack.block_type(0, 5)
    cnstr_atoms[0, 0] = torch.tensor([0, 4, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 5, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47
    cnstr_params[0, 1] = 0.1

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    # double the previous constraints, but this time batch them
    cnstr_atoms = torch.full((2, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((2, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 3)
    res2_type = pose_stack.block_type(0, 4)
    cnstr_atoms[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47
    cnstr_params[0, 1] = 0.1
    res1_type = pose_stack.block_type(0, 4)
    res2_type = pose_stack.block_type(0, 5)
    cnstr_atoms[1, 0] = torch.tensor([0, 4, res1_type.atom_to_idx["C"]])
    cnstr_atoms[1, 1] = torch.tensor([0, 5, res2_type.atom_to_idx["N"]])
    cnstr_params[1, 0] = 1.47
    cnstr_params[1, 1] = 0.1

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    # a circular harmonic constraint
    cnstr_atoms = torch.full((1, 4, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 3), 0, dtype=torch.float32, device=torch_device)

    # get the omega between res1 and res2
    res1_type = pose_stack.block_type(0, 0)
    res2_type = pose_stack.block_type(0, 1)
    cnstr_atoms[0, 0] = torch.tensor([0, 0, res1_type.atom_to_idx["CA"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 0, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 2] = torch.tensor([0, 1, res2_type.atom_to_idx["N"]])
    cnstr_atoms[0, 3] = torch.tensor([0, 1, res2_type.atom_to_idx["CA"]])
    cnstr_params[0, 0] = math.pi
    cnstr_params[0, 1] = 0.1
    cnstr_params[0, 2] = 0.0

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.circularharmonic, cnstr_atoms, cnstr_params
    )
    return attrs.evolve(
        pose_stack,
        constraint_set=constraints,
    )


# This should fail since the atoms are from different poses
def check_fail_add_cross_pose_constraint(pose_stack):
    torch_device = pose_stack.device

    constraints = pose_stack.get_constraint_set()
    if constraints is None:
        constraints = ConstraintSet.create_empty(
            device=torch_device, n_poses=pose_stack.n_poses
        )

    # a distance constraint
    cnstr_atoms = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 0)
    res2_type = pose_stack.block_type(1, 1)
    cnstr_atoms[0, 0] = torch.tensor([0, 0, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([1, 1, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47
    cnstr_params[0, 1] = 0.1

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )


def test_get_torsion_angle(torch_device):
    n_angles = 2
    n_atoms = 4
    tnsor = torch.full(
        (n_angles, n_atoms, 3), 0.0, dtype=torch.float32, device=torch_device
    )
    tnsor[0, 0] = torch.tensor([0.0, 1.0, 1.0])
    tnsor[0, 1] = torch.tensor([0.0, 1.0, 0.0])
    tnsor[0, 2] = torch.tensor([0.0, -1.0, 0.0])
    tnsor[0, 3] = torch.tensor([0.0, -1.0, -1.0])

    tnsor[1, 0] = torch.tensor([0.0, 1.0, 1.0])
    tnsor[1, 1] = torch.tensor([0.0, 1.0, 0.0])
    tnsor[1, 2] = torch.tensor([0.0, -1.0, 0.0])
    tnsor[1, 3] = torch.tensor([0.0, -1.0, 1.0])

    gold_vals = torch.tensor([-3.141593, 0.0])
    angles = ConstraintEnergyTerm.get_torsion_angle_test(tnsor)
    numpy.testing.assert_allclose(gold_vals.cpu(), angles.cpu(), atol=1e-5, rtol=1e-5)


def add_constraints_to_all_poses(pose_stack):
    torch_device = pose_stack.device
    constraints = pose_stack.get_constraint_set()
    if constraints is None:
        constraints = ConstraintSet.create_empty(
            device=torch_device, n_poses=pose_stack.n_poses
        )

    # a distance constraint
    cnstr_atoms = torch.full((1, 2, 2), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 0)
    res2_type = pose_stack.block_type(0, 1)
    cnstr_atoms[0, 0] = torch.tensor([0, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([1, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47
    cnstr_params[0, 1] = 0.1

    constraints = constraints.add_constraints_to_all_poses(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )
    return attrs.evolve(
        pose_stack,
        constraint_set=constraints,
    )


def modify_distances_and_check_constraints(pose_stack):
    torch_device = pose_stack.device
    constraints = pose_stack.get_constraint_set()
    if constraints is None:
        constraints = ConstraintSet.create_empty(
            device=torch_device, n_poses=pose_stack.n_poses
        )

    num_cnstrs = 10

    # a distance constraint
    cnstr_atoms = torch.full(
        (num_cnstrs, 2, 3), 0, dtype=torch.int32, device=torch_device
    )
    cnstr_params = torch.full(
        (num_cnstrs, 4), 0, dtype=torch.float32, device=torch_device
    )

    for res in range(10):
        pose_stack.coords[0, pose_stack.block_coord_offset[0, res]] = torch.tensor(
            [0, 0, res]
        )

        cnstr_atoms[res, 0] = torch.tensor([0, 0, 0])
        cnstr_atoms[res, 1] = torch.tensor([0, res, 0])
        cnstr_params[res, 0] = 1.0  # lb
        cnstr_params[res, 1] = 3.0  # ub
        cnstr_params[res, 2] = 1.0  # sd
        cnstr_params[res, 3] = 1.0  # rswitch

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.bounded, cnstr_atoms, cnstr_params
    )
    return attrs.evolve(
        pose_stack,
        constraint_set=constraints,
    )


class TestConstraintEnergyTerm(EnergyTermTestBase):
    energy_term_class = ConstraintEnergyTerm

    @classmethod
    def test_constraint_distance_range_score(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(0, 10)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            edit_pose_stack_fn=modify_distances_and_check_constraints,
            override_baseline_name=cls.test_constraint_distance_range_score.__name__,
            update_baseline=False,
        )

    @classmethod
    def test_ensure_fail_add_cross_pose_constraint(
        cls, ubq_pdb, default_database, torch_device
    ):
        with pytest.raises(Exception):
            super().test_whole_pose_scoring_10(
                ubq_pdb,
                default_database,
                torch_device,
                edit_pose_stack_fn=check_fail_add_cross_pose_constraint,
                update_baseline=False,
            )

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_10(
            ubq_pdb,
            default_database,
            torch_device,
            edit_pose_stack_fn=add_constraints_to_all_poses,
            update_baseline=False,
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls, ubq_pdb, default_database, torch_device: torch.device
    ):
        return super().test_whole_pose_scoring_jagged(
            ubq_pdb,
            default_database,
            torch_device,
            edit_pose_stack_fn=add_constraints_to_all_poses,
            update_baseline=False,
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 6)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            edit_pose_stack_fn=add_test_constraints_to_pose_stack,
            eps=1e-3,
            atol=5e-4,
            rtol=5e-4,
        )

    @classmethod
    def test_block_scoring_matches_whole_pose_scoring(
        cls, ubq_pdb, default_database, torch_device
    ):
        return super().test_block_scoring_matches_whole_pose_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            edit_pose_stack_fn=add_test_constraints_to_pose_stack,
        )

    @classmethod
    def test_block_scoring(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 6)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            edit_pose_stack_fn=add_test_constraints_to_pose_stack,
            update_baseline=False,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(0, 6)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums,
            edit_pose_stack_fn=add_test_constraints_to_pose_stack,
            eps=1e-3,
            atol=5e-4,
            rtol=5e-4,
        )


def test_create_coordinate_constraints(
    ubq_pdb, kin_minimized_ubq_pdb, default_database, torch_device
):
    ps = pose_stack_from_pdb(ubq_pdb, device=torch_device)
    pbt = ps.packed_block_types
    constraints = ps.get_constraint_set()
    if constraints is None:
        constraints = ConstraintSet.create_empty(
            device=torch_device, n_poses=ps.n_poses
        )

    is_heavy_atom = torch.zeros(
        (ps.n_poses, ps.max_n_blocks, ps.max_n_block_atoms),
        dtype=torch.bool,
        device=torch_device,
    )
    is_real_block = ps.block_type_ind64 != -1

    is_heavy_atom[is_real_block, :] = torch.logical_and(
        pbt.atom_is_real[ps.block_type_ind64[is_real_block]].to(torch.bool),
        ~pbt.atom_is_hydrogen[ps.block_type_ind64[is_real_block]].to(torch.bool),
    )
    heavy_atom_inds = torch.nonzero(is_heavy_atom, as_tuple=True)
    heavy_atom_inds = torch.stack(heavy_atom_inds, dim=1).unsqueeze(1)

    expanded_coords, _ = ps.expand_coords()
    target_coords = expanded_coords[is_heavy_atom]
    n_constrained_atoms = target_coords.size(0)
    cst_params = torch.full(
        (n_constrained_atoms, 5), 0, dtype=torch.float32, device=torch_device
    )
    cst_params[:, 1:4] = target_coords
    cst_params[:, 4] = 1.0

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic_coordinate,
        heavy_atom_inds,
        cst_params,
    )
    ps = attrs.evolve(
        ps,
        constraint_set=constraints,
    )

    cst_sfxn = ScoreFunction(default_database.scoring, torch_device)
    cst_sfxn.set_weight(ScoreType.constraint, 1.0)
    wpsm = cst_sfxn.render_whole_pose_scoring_module(ps)
    s = wpsm(ps.coords)
    torch.testing.assert_close(s, torch.tensor([0.0], device=torch_device))

    ps2 = pose_stack_from_pdb(kin_minimized_ubq_pdb, device=torch_device)
    constraints2 = ps2.get_constraint_set()
    if constraints2 is None:
        constraints2 = ConstraintSet.create_empty(
            device=torch_device, n_poses=ps2.n_poses
        )

    constraints2 = constraints2.add_constraints(
        ConstraintEnergyTerm.harmonic_coordinate,
        heavy_atom_inds,
        cst_params,
    )
    ps2 = attrs.evolve(
        ps2,
        constraint_set=constraints2,
    )

    wpsm = cst_sfxn.render_whole_pose_scoring_module(ps2)
    s2 = wpsm(ps2.coords)
    torch.testing.assert_close(
        s2, torch.tensor([4554.6299], device=torch_device), atol=1e-3, rtol=1e-3
    )
