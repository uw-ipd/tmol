# import attrs
# import torch
import math

from tmol.pose.pose_stack_builder import PoseStackBuilder

# from tmol.score.score_function import ScoreFunction
from tmol.pack.packer_task import PackerTask, PackerPalette, SetPackerTask
from tmol.pack.rotamer.build_rotamers import build_rotamers
from tmol.pack.rotamer.fixed_aa_chi_sampler import (
    FixedAAChiSampler,
)
from tmol.pack.rotamer.include_current_sampler import IncludeCurrentSampler
from tmol.pack.rotamer.opth_sampler import OptHSampler
from tmol.io import pose_stack_from_pdb


def test_optH_rotamer_sampler(ubq_pdb, torch_device):
    n_poses = 4
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    palette = PackerPalette()
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(OptHSampler())
    task.add_conformer_sampler(FixedAAChiSampler())

    for sampler in task.conformer_samplers:
        assert id(sampler) in task.conformer_sampler_index

    task = SetPackerTask.from_packer_task(task)

    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )

    # NHQ flip rotamers must have chi either matching the input (~0 deg)
    # or flipped by ~180 deg.
    from tmol.numeric.dihedrals import coord_dihedrals as _cd

    for i in range(task.allowed_bt_block_type.shape[0]):
        pose_i = task.allowed_bt_pose[i].item()
        block_i = task.allowed_bt_block[i].item()
        orig_bt = task.per_block_orig_block_type[pose_i, block_i].item()
        orig = pose_stack.packed_block_types.active_block_types[orig_bt]
        if not hasattr(orig, "opth_sampler_cache"):
            continue
        cache = orig.opth_sampler_cache
        if cache.nhq_chi_col < 0:
            continue
        a4 = cache.nhq_chi_4atoms
        off = int(pose_stack.block_coord_offset[pose_i, block_i].item())
        c = pose_stack.coords[pose_i][[off + int(a4[k]) for k in range(4)]].double()
        input_chi = float(_cd(c[0:1], c[1:2], c[2:3], c[3:4])[0])
        n_rots = int(rotamer_set.n_rots_for_block[pose_i, block_i].item())
        rot_off = int(rotamer_set.rot_offset_for_block[pose_i, block_i].item())
        for r in range(n_rots):
            co = int(rotamer_set.coord_offset_for_rot[rot_off + r].item())
            rc4 = rotamer_set.coords[[co + int(a4[k]) for k in range(4)]].double()
            rot_chi = float(_cd(rc4[0:1], rc4[1:2], rc4[2:3], rc4[3:4])[0])
            delta = math.degrees(rot_chi - input_chi)
            delta = (delta + 180.0) % 360.0 - 180.0
            # assert deltas are only 0 or 180
            assert min(abs(delta), abs(abs(delta) - 180.0)) < 1.0, (
                f"res {block_i} ({orig.name3}) rot {r}: "
                f"unexpected NHQ chi delta {delta:.2f} deg"
            )
