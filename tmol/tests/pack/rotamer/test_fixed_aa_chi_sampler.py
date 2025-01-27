import torch
import numpy
import cattr

from tmol.io import pose_stack_from_pdb
from tmol.chemical.restypes import RefinedResidueType, ResidueTypeSet
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.tests.data import no_termini_pose_stack_from_pdb


def test_annotate_residue_type_smoke(default_database):
    ala_restype = cattr.structure(
        cattr.unstructure(
            next(res for res in default_database.chemical.residues if res.name == "ALA")
        ),
        RefinedResidueType,
    )

    sampler = FixedAAChiSampler()
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

    sampler = FixedAAChiSampler()
    for restype in all_restypes:
        sampler.annotate_residue_type(restype)

    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, restype_set, all_restypes, torch_device
    )
    sampler.annotate_packed_block_types(pbt)


def test_chi_sampler_smoke(
    ubq_pdb, torch_device, default_database, default_restype_set
):
    torch_device = torch.device("cpu")
    # p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[5:11], torch_device
    # )
    # p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_database.chemical, ubq_res[:7], torch_device
    # )
    p1 = pose_stack_from_pdb(
        ubq_pdb,
        torch_device,
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    palette = PackerPalette(default_restype_set)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    sampler = FixedAAChiSampler()
    task.add_chi_sampler(sampler)

    for rt in poses.packed_block_types.active_block_types:
        sampler.annotate_residue_type(rt)
    sampler.annotate_packed_block_types(poses.packed_block_types)
    results = sampler.sample_chi_for_poses(poses, task)

    assert results[0].shape[0] == 13
    assert results[1].shape[0] == 1
    assert results[2].shape == (1, 1)
    assert results[3].shape == (1, 1)

    assert results[0].device == torch_device
    assert results[1].device == torch_device
    assert results[2].device == torch_device
    assert results[3].device == torch_device

    n_rots_for_rt_gold = numpy.zeros((13,), dtype=numpy.int32)
    n_rots_for_rt_gold[4] = 1
    numpy.testing.assert_equal(n_rots_for_rt_gold, results[0].cpu().numpy())

    rt_for_rot_gold = numpy.full((1,), 4, dtype=numpy.int32)
    numpy.testing.assert_equal(rt_for_rot_gold, results[1].cpu().numpy())
