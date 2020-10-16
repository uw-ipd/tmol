import numpy
import torch
import attr
import cattr

from tmol.pack.rotamer.build_rotamers import (
    annotate_restype,
    annotate_packed_block_types,
    build_rotamers,
    construct_scans_for_rotamers,
)
from tmol.system.restypes import RefinedResidueType, ResidueTypeSet
from tmol.system.pose import PackedBlockTypes, Pose, Poses
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.chi_sampler import ChiSampler
from tmol.pack.rotamer.dunbrack_chi_sampler import DunbrackChiSampler


def test_build_rotamers_smoke(ubq_res, default_database):
    torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)

    # replace them with residues constructed from the residue types
    # that live in our locally constructed set of refined residue types
    ubq_res = [
        attr.evolve(
            res,
            residue_type=next(
                rt for rt in rts.residue_types if rt.name == res.residue_type.name
            ),
        )
        for res in ubq_res
    ]

    p1 = Pose.from_residues_one_chain(ubq_res[5:11], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:7], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)
    palette = PackerPalette(rts)
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    sampler = DunbrackChiSampler.from_database(param_resolver)
    task.add_chi_sampler(sampler)

    build_rotamers(poses, task)


def test_update_scan_starts(default_database):
    torch_device = torch.device("cpu")

    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt_list = [rts.restype_map["MET"][0]]
    pbt = PackedBlockTypes.from_restype_list(leu_rt_list, device=torch_device)

    annotate_restype(leu_rt_list[0])
    annotate_packed_block_types(pbt)

    rt_block_inds = numpy.zeros(3, dtype=numpy.int32)
    rt_for_rot = torch.zeros(3, dtype=torch.int64)

    print("pbt.kintree_nodes")
    print(pbt.kintree_nodes)
    print("pbt.kintree_scans")
    print(pbt.kintree_scans)
    print("pbt.kintree_gens")
    print(pbt.kintree_gens)

    nodes, scans, gens = construct_scans_for_rotamers(pbt, rt_block_inds, rt_for_rot)
    print("nodes")
    print(nodes)
    print("scans")
    print(scans)
    print("gens")
    print(gens)
