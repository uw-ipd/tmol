import numpy

# import torch
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io.chain_deduction import chain_inds_for_pose_stack


def test_deduce_chains_for_monomer(ubq_res, default_restype_set, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        ubq_res[:5], torch_device
    )
    chain_inds = chain_inds_for_pose_stack(p1)
    numpy.testing.assert_equal(
        numpy.zeros(5, dtype=numpy.int32).reshape((1, 5)), chain_inds
    )


def test_deduce_chains_two_monomers(ubq_res, default_restype_set, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        ubq_res[:5], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        ubq_res[:7], torch_device
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    chain_inds = chain_inds_for_pose_stack(poses)
    gold_chain_inds = numpy.array(
        [[0, 0, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 0, 0]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(gold_chain_inds, chain_inds)
