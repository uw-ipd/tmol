import pytest

import numpy
import torch

from scipy.spatial.distance import cdist

from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available
from tmol.tests.benchmark import subfixture, make_subfixture

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.bond_dependent_term import BondDependentTerm

import sparse


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

    print("pbt.atom_all_bond_ranges")
    print(pbt.atom_all_bond_ranges)

    one_step, two_steps = compiled.two_steps(
        p1.inter_residue_connections,
        p1.block_type_ind,
        pbt.n_atoms,
        pbt.n_all_bonds,
        pbt.all_bonds,
        pbt.atom_all_bond_ranges,
        pbt.conn_atom,
    )

    blah = torch.arange(100, device=torch_device)

    print("one_step[0,0]")
    print(one_step[0, 0])
    print("two_steps[0,0]")
    print(two_steps[0, 0])
