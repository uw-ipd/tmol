# import attrs
import math
from types import SimpleNamespace

import biotite.structure
import numpy
import torch
from tmol.pose import PoseStackBuilder

# from tmol.score import ScoreFunction
from tmol.pack import PackerTask, PackerPalette, SetPackerTask
from tmol.pack.rotamer import (
    build_rotamers,
    FixedAAChiSampler,
    IncludeCurrentSampler,
    OptHSampler,
)
from tmol.io import atom_array_from_cif, pose_stack_from_biotite, pose_stack_from_pdb
from tmol.tests.data import data_path


def test_opth_builds_cartesian_product_for_multiple_proton_chis():
    opth_cache = SimpleNamespace(
        has_proton_chi=torch.tensor([True]),
        n_proton_samples=torch.tensor([6], dtype=torch.int32),
        n_samples_per_chi=torch.tensor([[2, 3]], dtype=torch.int32),
        expanded_samples=torch.tensor(
            [[[10.0, 11.0, 0.0], [20.0, 21.0, 22.0]]],
            dtype=torch.float32,
        ),
        chi_defining_atom=torch.tensor([[4, 5]], dtype=torch.int32),
    )
    pose_stack = SimpleNamespace(
        packed_block_types=SimpleNamespace(opth_sample_cache=opth_cache),
        device=torch.device("cpu"),
    )
    task = SimpleNamespace(cons_bt_block_type=torch.tensor([0], dtype=torch.int64))
    gbt_for_rotamer = torch.zeros(6, dtype=torch.int64)
    chi_defining_atom = torch.full((6, 2), -1, dtype=torch.int32)
    chi_values = torch.zeros((6, 2), dtype=torch.float32)

    OptHSampler()._fill_proton_chi_for_all_blocks(
        pose_stack,
        task,
        rot_offset_for_gbt=torch.tensor([0], dtype=torch.int32),
        gbt_for_rotamer=gbt_for_rotamer,
        chi_defining_atom_for_rotamer=chi_defining_atom,
        chi_for_rotamers=chi_values,
    )

    torch.testing.assert_close(
        chi_values,
        torch.tensor(
            [
                [10.0, 20.0],
                [11.0, 20.0],
                [10.0, 21.0],
                [11.0, 21.0],
                [10.0, 22.0],
                [11.0, 22.0],
            ]
        ),
    )
    torch.testing.assert_close(
        chi_defining_atom,
        torch.tensor([[4, 5]] * 6, dtype=torch.int32),
    )


def test_optH_rotamer_sampler_flipNHQ(ubq_pdb, torch_device):
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
    from tmol.numeric import coord_dihedrals as _cd

    for i in range(task.allowed_bt_block_type.shape[0]):
        pose_i = task.allowed_bt_pose[i].item()
        block_i = task.allowed_bt_block[i].item()
        orig_bt = task.per_block_orig_block_type[pose_i, block_i].item()
        orig = pose_stack.packed_block_types.active_block_types[orig_bt]
        assert hasattr(orig, "opth_sampler_cache")
        cache = orig.opth_sampler_cache
        if cache.nhq_chi_col >= 0:
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
        else:
            n_rots = int(rotamer_set.n_rots_for_block[pose_i, block_i].item())
            # n_proton_chi_samples + 1 for include current
            assert cache.n_proton_samples == 0 or n_rots == cache.n_proton_samples + 1


def his_ring_flipped(array, res_id):
    """Histidine `res_id` with its ring atoms turned 180 degrees about CB-CG."""
    his = array.res_id == res_id
    fixed = ["N", "CA", "C", "O", "CB", "CG", "H", "HA", "HB2", "HB3", "1HB", "2HB"]
    ring = his & ~numpy.isin(array.atom_name, fixed)
    cb, cg = (array.coord[his & (array.atom_name == n)][0] for n in ("CB", "CG"))
    axis = (cg - cb) / numpy.linalg.norm(cg - cb)
    offset = array.coord[ring] - cg
    flipped = array.copy()
    flipped.coord[ring] = cg + 2 * numpy.outer(offset @ axis, axis) - offset
    return flipped


def his_ring_after_optH(array, res_id, torch_device):
    """Block type name and ND1/NE2 coordinates of histidine `res_id` after OptH."""
    pose = pose_stack_from_biotite(array, torch_device, no_optH=False)
    index = int(
        numpy.flatnonzero(biotite.structure.get_residues(array)[0] == res_id)[0]
    )
    bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, index])]
    offset = int(pose.block_coord_offset[0, index])
    ring = pose.coords[0, [offset + bt.atom_to_idx[n] for n in ("ND1", "NE2")]]
    return bt.name, ring


def test_optH_flips_a_doubly_protonated_histidine_ring_back(torch_device):
    # 1YG0 His14 is deposited with HD1 and HE2, so it builds as HIS_POS.
    array = atom_array_from_cif(
        data_path("atomworks_regressions/his_pos_ring_1yg0.cif.zst")
    )
    (name, ring), (_, flipped) = (
        his_ring_after_optH(a, 14, torch_device)
        for a in (array, his_ring_flipped(array, 14))
    )
    assert name == "HIS_POS"
    torch.testing.assert_close(ring, flipped, atol=0.1, rtol=0)


def test_optH_rotamer_sampler_no_flipNHQ(ubq_pdb, torch_device):
    n_poses = 4
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    palette = PackerPalette()
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(OptHSampler(flip_NHQ=False))
    task.add_conformer_sampler(FixedAAChiSampler())

    for sampler in task.conformer_samplers:
        assert id(sampler) in task.conformer_sampler_index

    task = SetPackerTask.from_packer_task(task)

    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )

    #  no NHQ flip rotamers, but we do have the alt HIS tautomer
    from tmol.numeric import coord_dihedrals as _cd

    for i in range(task.allowed_bt_block_type.shape[0]):
        pose_i = task.allowed_bt_pose[i].item()
        block_i = task.allowed_bt_block[i].item()
        orig_bt = task.per_block_orig_block_type[pose_i, block_i].item()
        orig = pose_stack.packed_block_types.active_block_types[orig_bt]
        assert hasattr(orig, "opth_sampler_cache")
        cache = orig.opth_sampler_cache
        if cache.nhq_chi_col >= 0:
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
        else:
            n_rots = int(rotamer_set.n_rots_for_block[pose_i, block_i].item())
            assert cache.n_proton_samples == 0 or n_rots == cache.n_proton_samples + 1
