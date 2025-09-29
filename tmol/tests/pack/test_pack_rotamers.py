import torch

from tmol.chemical.patched_chemdb import PatchedChemicalDatabase
from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

from tmol.pack.compiled.compiled import validate_energies, build_interaction_graph
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.build_rotamers import (
    build_rotamers,
    RotamerSet,
)
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import (
    DunbrackChiSampler,
)
from tmol.pack.rotamer.fixed_aa_chi_sampler import (
    FixedAAChiSampler,
)
from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import run_simulated_annealing

from tmol.io import pose_stack_from_pdb


def test_pack_rotamers(default_database, ubq_pdb, dun_sampler, torch_device):
    # torch_device = torch.device("cpu")
    n_poses = 2

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)

    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)

    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()

    n_rots_total = rotamer_set.n_rotamers_total
    energy1b = torch.zeros((n_rots_total), dtype=torch.float32, device=torch_device)

    chunk_size = 16

    (chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
        build_interaction_graph(
            chunk_size,
            rotamer_set.n_rots_for_pose,
            rotamer_set.rot_offset_for_pose,
            rotamer_set.n_rots_for_block,
            rotamer_set.rot_offset_for_block,
            rotamer_set.pose_for_rot,
            rotamer_set.block_type_ind_for_rot,
            rotamer_set.block_ind_for_rot,
            energies.indices().to(torch.int32),
            energies.values(),
        )
    )

    packer_energy_tables = PackerEnergyTables(
        max_n_rotamers_per_pose=rotamer_set.max_n_rots_per_pose,
        pose_n_res=pose_stack.n_res_per_pose,
        pose_n_rotamers=rotamer_set.n_rots_for_pose,
        pose_rotamer_offset=rotamer_set.rot_offset_for_pose,
        nrotamers_for_res=rotamer_set.n_rots_for_block,
        oneb_offsets=rotamer_set.rot_offset_for_block,
        res_for_rot=rotamer_set.block_ind_for_rot,
        chunk_size=chunk_size,
        chunk_offset_offsets=chunk_pair_offset_for_block_pair,
        chunk_offsets=chunk_pair_offset,
        energy1b=energy1b,
        energy2b=energy2b,
    )

    scores, rotamer_assignments = run_simulated_annealing(packer_energy_tables)

    print("scores", scores)
