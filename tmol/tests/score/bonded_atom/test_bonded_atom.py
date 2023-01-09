import torch

from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.bond_dependent_term import BondDependentTerm


def test_bonded_atom_two_iterations(rts_ubq_res, default_database, torch_device):
    compiled = load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["test.cpp", "test.pybind.cpp", "test.cu"])
        ),
    )

    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res[0:1], device=torch_device
    )
    pbt = p1.packed_block_types
    dbt = BondDependentTerm(default_database, torch_device)
    for bt in pbt.active_block_types:
        dbt.setup_block_type(bt)
    dbt.setup_packed_block_types(pbt)
    dbt.setup_poses(p1)

    one_step, two_steps = compiled.two_steps(
        p1.inter_residue_connections,
        p1.block_type_ind,
        pbt.n_atoms,
        pbt.n_all_bonds,
        pbt.all_bonds,
        pbt.atom_all_bond_ranges,
        pbt.conn_atom,
    )

    one_step_gold = pbt.all_bonds[
        0, pbt.atom_all_bond_ranges[0, :, 0].to(dtype=torch.int64), 1
    ]
    two_step_tentative = pbt.all_bonds[
        0,
        pbt.atom_all_bond_ranges[0, one_step_gold.to(dtype=torch.int64), 0].to(
            dtype=torch.int64
        ),
        1,
    ]

    torch.testing.assert_close(one_step_gold, one_step[0, 0, :, 1])

    two_step_wrong = two_step_tentative == torch.arange(
        17, dtype=torch.int32, device=torch_device
    )
    two_step_tentative[two_step_wrong] = pbt.all_bonds[
        0,
        pbt.atom_all_bond_ranges[
            0, one_step_gold[two_step_wrong].to(dtype=torch.int64), 0
        ].to(dtype=torch.int64)
        + 1,
        1,
    ]

    torch.testing.assert_close(two_step_tentative, two_steps[0, 0, :, 1])
