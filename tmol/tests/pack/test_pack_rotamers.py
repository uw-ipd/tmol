import attrs
import torch
import math

from tmol.pose.constraint_set import ConstraintSet
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

from tmol.pack.compiled.compiled import build_interaction_graph
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.build_rotamers import build_rotamers
from tmol.pack.rotamer.fixed_aa_chi_sampler import (
    FixedAAChiSampler,
)
from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import run_simulated_annealing
from tmol.pack.impose_rotamers import impose_top_rotamer_assignments

from tmol.io import pose_stack_from_pdb
from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb

from tmol.pack.pack_rotamers import pack_rotamers

from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm


def get_packer_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.fa_elec, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.lk_ball_iso, -0.38)
    sfxn.set_weight(ScoreType.lk_ball, 0.92)
    sfxn.set_weight(ScoreType.lk_bridge, -0.33)
    sfxn.set_weight(ScoreType.lk_bridge_uncpl, -0.33)
    sfxn.set_weight(ScoreType.dunbrack_rot, 0.76)
    sfxn.set_weight(ScoreType.dunbrack_rotdev, 0.69)
    sfxn.set_weight(ScoreType.dunbrack_semirot, 0.78)
    sfxn.set_weight(ScoreType.cart_lengths, 0.5)
    sfxn.set_weight(ScoreType.cart_angles, 0.5)
    sfxn.set_weight(ScoreType.cart_torsions, 0.5)
    sfxn.set_weight(ScoreType.cart_impropers, 0.5)
    sfxn.set_weight(ScoreType.cart_hxltorsions, 0.5)
    sfxn.set_weight(ScoreType.omega, 0.48)
    sfxn.set_weight(ScoreType.rama, 0.50)
    sfxn.set_weight(ScoreType.ref, 1.0)
    sfxn.set_weight(ScoreType.disulfide, 1.0)
    sfxn.set_weight(ScoreType.constraint, 1.0)

    return sfxn


def get_constraints_only_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.constraint, 1.0)

    return sfxn


def test_pack_rotamers(default_database, ubq_pdb, dun_sampler, torch_device):
    n_poses = 4

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = get_packer_sfxn(default_database, torch_device)

    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)

    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()

    chunk_size = 16

    (energy1b, chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
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

    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack, rotamer_set, rotamer_assignments
    )
    write_pose_stack_pdb(new_pose_stack, "pack_rotamers_1ubq_ex1ex2.pdb")

    wpsm = sfxn.render_whole_pose_scoring_module(new_pose_stack)
    new_scores = wpsm(new_pose_stack.coords)
    torch.testing.assert_close(scores[:, 0], new_scores, atol=1e-3, rtol=1e-5)


def test_pack_rotamers_w_cst(default_database, ubq_pdb, dun_sampler, torch_device):
    n_poses = 4

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = get_constraints_only_sfxn(default_database, torch_device)

    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)
    constraints = ConstraintSet.create_empty(device=torch_device, n_poses=n_poses)

    # a distance constraint
    cnstr_atoms = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 1), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 3)
    res2_type = pose_stack.block_type(0, 4)
    cnstr_atoms[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    # a distance constraint
    cnstr_atoms = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 1), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 5)
    res2_type = pose_stack.block_type(0, 6)
    cnstr_atoms[0, 0] = torch.tensor([0, 5, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 6, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    # a circular harmonic constraint
    cnstr_atoms = torch.full((1, 4, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 3), 0, dtype=torch.float32, device=torch_device)

    # get the omega between res1 and res2
    res1_type = pose_stack.block_type(0, 0)
    res2_type = pose_stack.block_type(0, 1)
    cnstr_atoms[0, 0] = torch.tensor([0, 0, res1_type.atom_to_idx["CA"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 0, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 2] = torch.tensor([0, 1, res2_type.atom_to_idx["N"]])
    cnstr_atoms[0, 3] = torch.tensor([0, 1, res2_type.atom_to_idx["CA"]])
    cnstr_params[0, 0] = math.pi
    cnstr_params[0, 1] = 0.1
    cnstr_params[0, 2] = 0.0

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.circularharmonic, cnstr_atoms, cnstr_params
    )

    pose_stack = attrs.evolve(pose_stack, constraint_set=constraints)

    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()

    chunk_size = 16

    (energy1b, chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
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

    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack, rotamer_set, rotamer_assignments
    )
    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()

    wpsm = sfxn.render_whole_pose_scoring_module(new_pose_stack)
    new_scores = wpsm(new_pose_stack.coords)
    torch.testing.assert_close(scores[:, 0], new_scores, atol=1e-3, rtol=1e-5)


def test_pack_rotamers_w_empty_interaction_graph(
    default_database, disulfide_pdb, dun_sampler, torch_device
):
    n_poses = 4

    p = pose_stack_from_pdb(disulfide_pdb, torch_device)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = get_constraints_only_sfxn(default_database, torch_device)

    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)
    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()

    chunk_size = 16

    (energy1b, chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
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

    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack, rotamer_set, rotamer_assignments
    )

    wpsm = sfxn.render_whole_pose_scoring_module(new_pose_stack)
    new_scores = wpsm(new_pose_stack.coords)
    torch.testing.assert_close(scores[:, 0], new_scores, atol=1e-3, rtol=1e-5)


def test_pack_rotamers_w_dslf(
    default_database, disulfide_pdb, dun_sampler, torch_device
):
    n_poses = 4

    p = pose_stack_from_pdb(disulfide_pdb, torch_device)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = get_packer_sfxn(default_database, torch_device)

    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)

    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()

    chunk_size = 16

    (energy1b, chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
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

    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack, rotamer_set, rotamer_assignments
    )

    wpsm = sfxn.render_whole_pose_scoring_module(new_pose_stack)
    new_scores = wpsm(new_pose_stack.coords)
    torch.testing.assert_close(scores[:, 0], new_scores, atol=1e-3, rtol=1e-5)


def test_pack_rotamers2(default_database, ubq_pdb, dun_sampler, torch_device):

    if torch_device == torch.device("cpu"):
        return
    n_poses = 100

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)

    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()
    task.or_expand_chi(1)
    task.or_expand_chi(2)

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = get_packer_sfxn(default_database, torch_device)

    pack_rotamers(pose_stack, sfxn, task)


def test_pack_rotamers_irregular_sized_poses(
    default_database, ubq_pdb, dun_sampler, torch_device
):

    if torch_device == torch.device("cpu"):
        return
    n_poses = 20

    pose_stack = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(
                ubq_pdb, torch_device, residue_start=0, residue_end=20 + i
            )
            for i in range(n_poses)
        ],
        torch_device,
    )

    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()
    task.or_expand_chi(1)
    task.or_expand_chi(2)

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = get_packer_sfxn(default_database, torch_device)

    pack_rotamers(pose_stack, sfxn, task)
