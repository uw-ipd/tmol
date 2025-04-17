import numpy

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io import pose_stack_from_pdb
from tmol.io.chain_deduction import chain_inds_for_pose_stack
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form


def test_deduce_chains_for_monomer(ubq_pdb, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    chain_inds = chain_inds_for_pose_stack(p1)
    numpy.testing.assert_equal(
        numpy.zeros(5, dtype=numpy.int32).reshape((1, 5)), chain_inds
    )


def test_deduce_chains_two_monomers(ubq_pdb, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    chain_inds = chain_inds_for_pose_stack(poses)
    gold_chain_inds = numpy.array(
        [[0, 0, 0, 0, 0, -1, -1], [0, 0, 0, 0, 0, 0, 0]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(gold_chain_inds, chain_inds)


def test_deduce_chains_dslf_dimer(pertuzumab_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pertuzumab_pdb, torch_device)

    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

    # note that in this test case, there is a disulfide formed between the two
    # chains and that this chemical bond should not be used to join the two
    # chains into a single chain
    chain_inds = chain_inds_for_pose_stack(pose_stack)

    chain_inds_gold = numpy.zeros(
        (
            1,
            pose_stack.max_n_blocks,
        ),
        dtype=numpy.int32,
    )
    chain_inds_gold[0, 214:] = 1  # there is a second chain

    numpy.testing.assert_equal(chain_inds_gold, chain_inds)
