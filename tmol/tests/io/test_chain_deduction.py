import numpy

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io.chain_deduction import chain_inds_for_pose_stack
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form


def test_deduce_chains_for_monomer(ubq_res, default_restype_set, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_restype_set.chem_db, ubq_res[:5], torch_device
    )
    chain_inds = chain_inds_for_pose_stack(p1)
    numpy.testing.assert_equal(
        numpy.zeros(5, dtype=numpy.int32).reshape((1, 5)), chain_inds
    )


def test_deduce_chains_two_monomers(ubq_res, default_restype_set, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_restype_set.chem_db, ubq_res[:5], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_restype_set.chem_db, ubq_res[:7], torch_device
    )
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
