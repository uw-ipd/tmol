import torch
import attr
import cattr

from tmol.pack.rotamer.build_rotamers import build_rotamers
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
