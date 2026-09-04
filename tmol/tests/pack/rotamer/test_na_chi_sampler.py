"""Rotamer generation for nucleic acids."""

import torch
import pytest

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
    sampler = NaChiRotamerSampler.from_database(default_database, torch_device)
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

    allowed = torch.cat(
        (
            sampler.params.chi_means.reshape(-1).to(chi.device),
            torch.tensor([SYN_MEAN], dtype=chi.dtype, device=chi.device),
        )
    )
    # the sampler reports chi in radians; the tables that define it are degrees
    chi1 = torch.rad2deg(chi[:, 0])
    closest = (chi1.unsqueeze(-1) - allowed.unsqueeze(0)).abs().min(dim=1).values
    assert float(closest.max()) < 1e-3


def test_na_sampler_syn_is_rna_only(dna_pdb, rna_pdb, default_database, torch_device):
    """RNA gets syn rotamers for the bases whose syn well is deep enough; DNA
    gets anti only."""

    def has_syn(pdb_lines):
        _, _, _, chi = _sample(pdb_lines, default_database, torch_device)
        chi1 = torch.rad2deg(chi[:, 0])
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

    # three staggered samples, each with the expansions either side
    expected = sorted(
        (s + e) % 360.0 for s in (60.0, -60.0, 180.0) for e in (0.0, 20.0, -20.0)
    )
    for col in sampled:
        # a block type without this chi contributes a zero in its column
        vals = sorted(
            float(v) % 360.0
            for v in torch.rad2deg(torch.unique(chi[:, col]))
            if float(v) != 0.0
        )
        assert vals == pytest.approx(expected, abs=1e-3)


def test_na_sampler_chi_level_widens_the_rotamer_set(
    rna_pdb, default_database, torch_device
):
    """A higher extra-chi level expands each mean in units of the term's sdev."""
    _, n_rots_0, _, _ = _sample(rna_pdb, default_database, torch_device)
    _, n_rots_1, _, _ = _sample(
        rna_pdb, default_database, torch_device, chi_sample_level=1
    )
    assert int(n_rots_1.sum()) == 3 * int(n_rots_0.sum())


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
