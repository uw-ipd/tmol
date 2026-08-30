"""Rotamer generation for nucleic acids."""

import torch
import pytest

import tmol.pack.rotamer._na_chi_sampler as na_chi_sampler_module

from tmol.io import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
    pose_stack_from_canonical_form,
)
from tmol.pack import PackerTask, SetPackerTask, PackerPalette
from tmol.pack.rotamer import (
    build_rotamers,
    NaChiRotamerSampler,
)
from tmol.score.na_torsion import BASE_FOR_NAME3, SYN_MEAN, SYN_RANGE


def _pose_stack(pdb_lines, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pdb_lines, torch_device)
    return pose_stack_from_canonical_form(co, pbt, *canonical_form)


def _set_task(poses, sampler):
    palette = PackerPalette()
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    return SetPackerTask.from_packer_task(task)


def _sample(pdb_lines, default_database, torch_device, **kwargs):
    poses = _pose_stack(pdb_lines, torch_device)
    sampler = NaChiRotamerSampler.from_database(
        default_database, torch_device, **kwargs
    )
    n_rots, gbt_for_rot, _, chi = sampler.sample_chi_for_poses(
        poses, _set_task(poses, sampler)
    )
    return sampler, n_rots, gbt_for_rot, chi


def test_na_sampler_builds_for_nucleotides_only(default_database, torch_device):
    sampler = NaChiRotamerSampler.from_database(
        default_database, torch.device(torch_device.type)
    )
    assert sampler.device == torch_device
    assert sampler.params.chi_means.device == torch_device
    pbt = default_packed_block_types(torch_device)
    for bt in pbt.active_block_types:
        assert sampler.defines_rotamers_for_rt(bt) == (
            bt.name3 in BASE_FOR_NAME3
        ), bt.name


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb"])
def test_na_sampler_chi_sit_at_scored_minima(
    fixture, request, default_database, torch_device
):
    """Every glycosidic chi is a mean of the term that will score it.

    At sample level 0 the expansion is k=0, so each chi must equal a per-base,
    per-pucker anti mean or the fixed syn mean exactly -- a rotamer the energy
    function disagrees with would show up here as an unmatched value.
    """
    sampler, n_rots, gbt_for_rot, chi = _sample(
        request.getfixturevalue(fixture), default_database, torch_device
    )

    assert chi.shape[0] > 0
    assert int(n_rots.sum()) == gbt_for_rot.shape[0]
    assert chi.shape[0] == gbt_for_rot.shape[0]
    assert gbt_for_rot.dtype == torch.int32

    allowed = torch.cat(
        (
            sampler.params.chi_means.reshape(-1).to(chi.device),
            torch.tensor([SYN_MEAN], dtype=chi.dtype, device=chi.device),
        )
    )
    closest = (chi[:, 0].unsqueeze(-1) - allowed.unsqueeze(0)).abs().min(dim=1).values
    assert float(closest.max()) < 1e-3


def test_na_sampler_syn_is_rna_only(dna_pdb, rna_pdb, default_database, torch_device):
    """RNA gets syn rotamers for the bases whose syn well is deep enough; DNA
    gets anti only."""

    def has_syn(pdb_lines):
        _, _, _, chi = _sample(pdb_lines, default_database, torch_device)
        chi1 = chi[:, 0]
        return bool(torch.any((chi1 >= SYN_RANGE[0]) & (chi1 <= SYN_RANGE[1])))

    assert not has_syn(dna_pdb)
    assert has_syn(rna_pdb)


@pytest.mark.parametrize("fixture,n_proton_chi", [("dna_pdb", 1), ("rna_pdb", 2)])
def test_na_sampler_expands_proton_chis(
    fixture, n_proton_chi, request, default_database, torch_device
):
    """The sampler expands proton chis itself rather than composing with OptH.

    A block gets either this sampler or OptHSampler, never both, so every
    hydroxyl the block carries has to appear in this sampler's chi vector: a
    terminal 5'- or 3'-OH for DNA, and the 2'-OH as well for RNA. Column 0 is
    the glycosidic chi; the rest are padded to the widest block type, so a
    proton chi is identified by actually varying.
    """
    _, _, _, chi = _sample(
        request.getfixturevalue(fixture), default_database, torch_device
    )
    sampled = [c for c in range(1, chi.shape[1]) if len(torch.unique(chi[:, c])) > 1]
    assert len(sampled) == n_proton_chi

    for col in sampled:
        vals = torch.unique(chi[:, col])  # full circle in 20-degree steps
        assert len(vals) == 18
        assert float(vals.min()) == 0.0
        assert float(vals.max()) == 340.0


def test_na_sampler_chi_level_widens_the_rotamer_set(
    rna_pdb, default_database, torch_device
):
    """A higher extra-chi level expands each mean in units of the term's sdev."""
    _, n_rots_0, _, _ = _sample(rna_pdb, default_database, torch_device)
    _, n_rots_1, _, _ = _sample(
        rna_pdb, default_database, torch_device, chi_sample_level=1
    )
    assert int(n_rots_1.sum()) == 3 * int(n_rots_0.sum())


def test_na_sampler_measures_only_enabled_na_puckers(
    protein_dna_pdb, default_database, torch_device, monkeypatch
):
    """Protein blocks in a mixed complex must not enter the sugar-pucker kernel."""
    poses = _pose_stack(protein_dna_pdb, torch_device)
    sampler = NaChiRotamerSampler.from_database(default_database, torch_device)
    task = _set_task(poses, sampler)
    sampler.annotate_packed_block_types(poses.packed_block_types)
    sampler_index = task.conformer_sampler_index[id(sampler)]
    allowed = task.per_block_conformer_sampler_allowed[
        task.cons_bt_pose, task.cons_bt_block, sampler_index
    ]
    builds = poses.packed_block_types.na_chi_sampler_cache["builds_bt"][
        task.cons_bt_block_type
    ]
    block_type_allowed = task.per_block_is_block_type_allowed[
        task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
    ]
    active = allowed & builds & block_type_allowed
    active_pose = task.cons_bt_pose[active]
    active_block = task.cons_bt_block[active]
    expected = torch.unique(active_pose * poses.max_n_blocks + active_block).numel()
    assert 0 < expected < task.cons_bt_block_type.shape[0]

    calls = []
    original = na_chi_sampler_module.pucker_weights

    def record_pucker_count(xyz, temperature):
        calls.append(xyz.shape[0])
        return original(xyz, temperature)

    monkeypatch.setattr(na_chi_sampler_module, "pucker_weights", record_pucker_count)
    sampler.sample_chi_for_poses(poses, task)

    assert calls == [expected]


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb"])
def test_na_build_rotamers(fixture, request, default_database, torch_device):
    """Rotamers build to real coordinates through the whole packer path."""
    poses = _pose_stack(request.getfixturevalue(fixture), torch_device)
    sampler = NaChiRotamerSampler.from_database(default_database, torch_device)

    palette = PackerPalette()
    task = PackerTask(poses, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    task = SetPackerTask.from_packer_task(task)

    poses, rotamer_set = build_rotamers(poses, task, default_database.chemical)

    assert rotamer_set is not None
    assert rotamer_set.coords.shape[0] > 0
    assert not bool(torch.any(torch.isnan(rotamer_set.coords)))


def test_na_mutation_rotamers_preserve_sugar_coordinates(
    dna_pdb, default_database, torch_device
):
    """A base substitution is anchored to the input nucleotide sugar."""
    poses = _pose_stack(dna_pdb, torch_device)
    pbt = poses.packed_block_types
    name3 = [bt.name3 for bt in pbt.active_block_types]
    dg_ind = name3.index("DG")
    pose_ind, block_ind = torch.nonzero(poses.block_type_ind == dg_ind, as_tuple=True)
    pose_ind = int(pose_ind[0])
    block_ind = int(block_ind[0])

    mutation_mask = torch.zeros_like(poses.block_type_ind, dtype=torch.bool)
    mutation_mask[pose_ind, block_ind] = True
    task = PackerTask(poses, PackerPalette())
    task.restrict_to_repacking(~mutation_mask)
    task.restrict_absent_name3s({"DA"}, mutation_mask)
    sampler = NaChiRotamerSampler.from_database(default_database, torch_device)
    task.add_conformer_sampler(sampler)
    task = SetPackerTask.from_packer_task(task)

    _, rotamers = build_rotamers(poses, task, default_database.chemical)
    da_inds = torch.tensor(
        [i for i, bt in enumerate(pbt.active_block_types) if bt.name3 == "DA"],
        dtype=torch.int64,
        device=torch_device,
    )
    is_da = torch.isin(rotamers.block_type_ind_for_rot, da_inds)
    selected = torch.nonzero(
        (rotamers.pose_for_rot == pose_ind)
        & (rotamers.block_ind_for_rot == block_ind)
        & is_da
    ).flatten()
    assert selected.numel() > 0

    original_bt = pbt.active_block_types[dg_ind]
    original_offset = int(poses.block_coord_offset[pose_ind, block_ind])
    sugar_atoms = ("P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
    original = poses.coords[
        pose_ind,
        [original_offset + original_bt.atom_to_idx[name] for name in sugar_atoms],
    ]
    for rot_ind in selected.tolist():
        rotamer_bt = pbt.active_block_types[
            int(rotamers.block_type_ind_for_rot[rot_ind])
        ]
        rotamer_offset = int(rotamers.coord_offset_for_rot[rot_ind])
        mutated = rotamers.coords[
            [rotamer_offset + rotamer_bt.atom_to_idx[name] for name in sugar_atoms]
        ]
        torch.testing.assert_close(mutated, original, atol=1e-3, rtol=0)
