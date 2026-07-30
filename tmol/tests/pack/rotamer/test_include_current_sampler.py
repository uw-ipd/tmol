import torch
import numpy
import cattr

from tmol.chemical.restypes import RefinedResidueType, ResidueTypeSet
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.packer_task import PackerTask, PackerPalette, SetPackerTask
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


def test_include_current_sampler_smoke(ubq_pdb, torch_device):
    torch_device = torch.device("cpu")
    p1 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=5, residue_end=11
    )
    p2 = no_termini_pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_start=1, residue_end=8
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    palette = PackerPalette()
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    sampler = IncludeCurrentSampler()
    task.add_conformer_sampler(sampler)

    disabled_residues = [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (1, 5)]
    pose_ind, block_ind = tuple(
        [torch.tensor([x[i] for x in disabled_residues]) for i in range(2)]
    )
    task.per_block_is_block_type_allowed[pose_ind, block_ind, :] = False

    task = SetPackerTask.from_packer_task(task)

    for rt in poses.packed_block_types.active_block_types:
        sampler.annotate_residue_type(rt)
    sampler.annotate_packed_block_types(poses.packed_block_types)
    results = sampler.create_samples_for_poses(poses, task)

    enabled_residues = [
        (pose, res)
        for pose in range(2)
        for res in range(6 if pose == 0 else 7)
        if (pose, res) not in disabled_residues
    ]

    assert results[0].shape[0] == 21 * 13
    assert results[1].shape[0] == len(enabled_residues)

    assert results[0].device == torch_device
    assert results[1].device == torch_device
    assert results[2] == {}

    n_rots_for_rt_gold = numpy.zeros((21 * 13,), dtype=numpy.int32)
    rt_for_rot_gold = numpy.full((len(enabled_residues),), -1, dtype=numpy.int32)
    for i, (pose, res) in enumerate(enabled_residues):
        curr_rt = poses.block_type_ind64[pose, res]
        curr_rt_in_considered = next(
            i
            for i in range(task.per_block_considered_block_types.shape[2])
            if task.per_block_considered_block_types[pose, res, i] == curr_rt
        )
        i_gbt = (pose * 6 + res) * 21 + curr_rt_in_considered
        n_rots_for_rt_gold[i_gbt] = 1
        rt_for_rot_gold[i] = i_gbt

    numpy.testing.assert_equal(n_rots_for_rt_gold, results[0].cpu().numpy())
    numpy.testing.assert_equal(rt_for_rot_gold, results[1].cpu().numpy())
