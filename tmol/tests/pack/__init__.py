import pytest

from tmol.pack.rotamer.build_rotamers import build_rotamers

from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler

from tmol.tests.data import no_termini_pose_stack_from_pdb


@pytest.fixture
def ubq_repacking_rotamers(default_database, ubq_pdb, torch_device, dun_sampler):
    n_poses = 2

    # fd TEMP: NO TERM VARIANTS
    p = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=5
    )
    poses = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    restype_set = poses.packed_block_types.restype_set
    for restype in restype_set.residue_types:
        print(restype.name, restype.base_name, restype.name3)

    palette = PackerPalette(restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_chi_sampler(dun_sampler)
    task.add_chi_sampler(fixed_sampler)

    poses, rotamer_set = build_rotamers(poses, task, default_database.chemical)

    return poses, rotamer_set
