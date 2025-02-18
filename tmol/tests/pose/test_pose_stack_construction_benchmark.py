import pytest

from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.tests.torch import zero_padded_counts
from tmol.chemical.restypes import three2one


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30, 100]))
@pytest.mark.benchmark(group="pose_stack_construction_from_seq")
def test_pose_construction_from_sequence(
    benchmark,
    n_poses,
    ubq_pdb,
    fresh_default_packed_block_types,
    torch_device,
):
    n_poses = int(n_poses)

    ubq_pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    pbt = ubq_pose_stack.packed_block_types
    ubq_res = [
        pbt.active_block_types[ubq_pose_stack.block_type_ind64[0, i]]
        for i in range(ubq_pose_stack.max_n_blocks)
    ]

    seq = [three2one(res.name3) for res in ubq_res]
    n_pose_seq = [seq] * n_poses

    @benchmark
    def construct_pass():
        pose_stack_n = PoseStackBuilder.pose_stack_from_monomer_polymer_sequences(
            fresh_default_packed_block_types, n_pose_seq
        )
        return pose_stack_n
