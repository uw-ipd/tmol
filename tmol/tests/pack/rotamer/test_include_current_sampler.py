import torch
import numpy
import cattr

from tmol.chemical.restypes import RefinedResidueType, ResidueTypeSet
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.include_current_sampler import IncludeCurrentSampler

from tmol.tests.data import no_termini_pose_stack_from_pdb


def test_annotate_residue_type_smoke(default_database):
    ala_restype = cattr.structure(
        cattr.unstructure(
            next(res for res in default_database.chemical.residues if res.name == "ALA")
        ),
        RefinedResidueType,
    )

    sampler = IncludeCurrentSampler()
    sampler.annotate_residue_type(ala_restype)
    # this is a no-op


def test_annotate_packed_block_types_smoke(default_database, torch_device):
    desired = set(["ALA", "GLY", "TYR"])

    all_restypes = [
        cattr.structure(cattr.unstructure(res), RefinedResidueType)
        for res in default_database.chemical.residues
        if res.name in desired
    ]
    restype_set = ResidueTypeSet.from_restype_list(
        default_database.chemical, all_restypes
    )

    sampler = IncludeCurrentSampler()
    for restype in all_restypes:
        sampler.annotate_residue_type(restype)

    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, restype_set, all_restypes, torch_device
    )
    sampler.annotate_packed_block_types(pbt)


def test_include_current_sampler_smoke(ubq_pdb, torch_device, default_restype_set):
    torch_device = torch.device("cpu")
    p1 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=5, residue_end=11
    )
    p2 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=8
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    pbt = poses.packed_block_types
    palette = PackerPalette(default_restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    sampler = IncludeCurrentSampler()
    task.add_conformer_sampler(sampler)
    # task.add_conformer_sampler(sampler)

    residues_to_fix = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (1, 5)]
    for pose, res in residues_to_fix:
        task.blts[pose][res].disable_packing()

    for rt in poses.packed_block_types.active_block_types:
        sampler.annotate_residue_type(rt)
    sampler.annotate_packed_block_types(poses.packed_block_types)
    results = sampler.create_samples_for_poses(poses, task)

    assert results[0].shape[0] == 21 * 13
    assert results[1].shape[0] == len(residues_to_fix)

    assert results[0].device == torch_device
    assert results[1].device == torch_device
    assert results[2] == {}
    # assert results[3].device == torch_device

    n_rots_for_rt_gold = numpy.zeros((21 * 13,), dtype=numpy.int32)
    rt_for_rot_gold = numpy.full((6,), -1, dtype=numpy.int32)
    for i, (pose, res) in enumerate(residues_to_fix):
        curr_rt = pbt.active_block_types[poses.block_type_ind[pose, res]]
        curr_rt_in_considered = task.blts[pose][res].considered_block_types.index(
            curr_rt
        )
        i_gbt = (pose * 6 + res) * 21 + curr_rt_in_considered
        n_rots_for_rt_gold[i_gbt] = (
            1  # 6 cause p0 has 6 res and we only have two poses to worry about
        )
        rt_for_rot_gold[i] = i_gbt

    numpy.testing.assert_equal(n_rots_for_rt_gold, results[0].cpu().numpy())
    numpy.testing.assert_equal(rt_for_rot_gold, results[1].cpu().numpy())
