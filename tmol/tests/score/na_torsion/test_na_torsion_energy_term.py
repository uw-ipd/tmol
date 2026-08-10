import numpy
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.score.na_torsion.na_torsion_energy_term import (
    NaTorsionEnergyTerm,
    na_torsion_subterms,
)
from tmol.score.na_torsion.params import polymer_index
from tmol.score.na_torsion.potentials import wrap_degrees


def _term_and_pose(pdb, torch_device):
    ps = pose_stack_from_pdb(pdb, torch_device)
    term = NaTorsionEnergyTerm(ParameterDatabase.get_default(), torch_device)
    for bt in ps.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(ps.packed_block_types)
    term.setup_poses(ps)
    return term, ps


def test_smoke(default_database, torch_device):
    term = NaTorsionEnergyTerm(param_db=default_database, device=torch_device)
    assert term.device == torch_device


@pytest.mark.parametrize(
    "fixture,names,n_scored",
    [
        ("dna_pdb", ("DA", "DC", "DG", "DT"), 24),  # 1BNA is 2 x 12 nt
        ("rna_pdb", ("RA", "RC", "RG", "RU"), 42),  # 3ZP8 chain A
    ],
)
def test_all_na_block_types_are_scoreable(
    fixture, names, n_scored, request, torch_device
):
    """Every nucleotide type including the terminus variants must score; the 5'
    patch drops P, which removes alpha and beta but must not disable the
    residue."""
    term, ps = _term_and_pose(request.getfixturevalue(fixture), torch_device)
    pbt = ps.packed_block_types
    scoreable = {
        bt.name
        for bt, b in zip(pbt.active_block_types, pbt.na_torsion_base.tolist())
        if b >= 0
    }
    for base in names:
        for suffix in ("", ":na5prime", ":na3prime", ":na5prime:na3prime"):
            assert base + suffix in scoreable

    real = ps.block_type_ind64[0] >= 0
    bases = pbt.na_torsion_base.to(torch.int64)[ps.block_type_ind64[0].clamp_min(0)]
    assert int((bases[real] >= 0).sum()) == n_scored


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb"])
def test_scores_na_and_ignores_protein(fixture, request, ubq_pdb, torch_device):
    term, ps = _term_and_pose(request.getfixturevalue(fixture), torch_device)
    na = term.render_whole_pose_scoring_module(ps)(ps.coords)
    assert na.shape == (2, 1)  # harmonic and well score types
    assert torch.isfinite(na).all()
    assert bool((na > 0).all())

    term, ps = _term_and_pose(ubq_pdb, torch_device)
    protein = term.render_whole_pose_scoring_module(ps)(ps.coords)
    assert float(protein.sum()) == 0.0


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb"])
@pytest.mark.parametrize("n_poses", [1, 3, 7])
def test_stacked_poses_scale_linearly(n_poses, fixture, request, torch_device):
    """Identical poses must give exactly n times the single-pose energy.

    Guards the uaid resolution, whose reshapes have to keep pose and block as
    separate dimensions; collapsing them still broadcasts at a single pose.
    """
    from tmol.pose.pose_stack_builder import PoseStackBuilder

    term, ps1 = _term_and_pose(request.getfixturevalue(fixture), torch_device)
    one = term.render_whole_pose_scoring_module(ps1)(ps1.coords)

    psn = PoseStackBuilder.from_poses([ps1] * n_poses, torch_device)
    term.setup_packed_block_types(psn.packed_block_types)
    term.setup_poses(psn)
    scores = term.render_whole_pose_scoring_module(psn)(psn.coords)

    assert scores.shape == (2, n_poses)
    numpy.testing.assert_allclose(
        scores.detach().cpu().numpy(),
        one.detach().cpu().numpy().repeat(n_poses, axis=1),
        rtol=1e-5,
    )


def _subterms(term, ps):
    return na_torsion_subterms(
        ps.coords.flatten(0, -2), ps.block_type_ind, *term.subterm_attributes(ps)
    )


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb"])
def test_subterms_sum_to_the_totals(fixture, request, torch_device):
    term, ps = _term_and_pose(request.getfixturevalue(fixture), torch_device)
    bb, chi, sugar, well, mask, base = _subterms(term, ps)
    p = term.params
    poly = polymer_index(base)
    harmonic = (
        p.weight_bb[poly] * bb + p.weight_chi[poly] * chi + p.weight_sugar[poly] * sugar
    )
    total = term.render_whole_pose_scoring_module(ps)(ps.coords)

    for row, combined in enumerate((harmonic, well)):
        numpy.testing.assert_allclose(
            float(torch.where(mask, combined, torch.zeros_like(combined)).sum()),
            float(total[row]),
            rtol=1e-5,
        )
    for sub in (bb, chi, sugar, well):
        assert bool((sub > 0).any())


def test_rotamer_energies_match_the_pose_energies(protein_dna_pdb, torch_device):
    """The include-current rotamer of each DNA block must reproduce that
    block's pose energy, including the backbone torsions that reach into the
    neighboring nucleotide's background coordinates."""
    from tmol.pack.packer_task import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer.build_rotamers import build_rotamers
    from tmol.pack.rotamer.include_current_sampler import IncludeCurrentSampler
    from tmol.score.score_function import ScoreFunction
    from tmol.score.score_types import ScoreType

    ps = pose_stack_from_pdb(protein_dna_pdb, torch_device)
    sfxn = ScoreFunction(ParameterDatabase.get_default(), torch_device)
    sfxn.set_weight(ScoreType.na_torsion, 1.0)
    sfxn.set_weight(ScoreType.na_torsion_well, 1.0)

    task = PackerTask(ps, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    ps, rotamer_set = build_rotamers(
        ps, SetPackerTask.from_packer_task(task), ps.packed_block_types.chem_db
    )

    per_rot = (
        sfxn.render_rotamer_scoring_module(ps, rotamer_set)(rotamer_set.coords)
        .coalesce()
        .to_dense()
    )
    per_block = sfxn.render_block_pair_scoring_module(ps)(ps.coords)

    pose = rotamer_set.pose_for_rot.to(torch.int64)
    block = rotamer_set.block_ind_for_rot.to(torch.int64)
    rot = torch.arange(pose.shape[0], device=torch_device)
    dna = per_block[pose, block, block] > 0
    assert int(dna.sum()) > 0

    numpy.testing.assert_allclose(
        per_rot[pose, rot, rot][dna].detach().cpu().numpy(),
        per_block[pose, block, block][dna].detach().cpu().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb"])
def test_gradcheck(fixture, request, torch_device):
    term, ps = _term_and_pose(request.getfixturevalue(fixture), torch_device)
    module = term.render_whole_pose_scoring_module(ps)
    # a handful of nucleotides is enough and keeps the numeric jacobian cheap
    coords = ps.coords.clone().to(torch.float64)
    n = 40
    head = torch.nn.Parameter(coords[:, :n])

    def f(x):
        full = torch.cat([x, coords[:, n:]], dim=1)
        return module(full).sum()

    # eps must be small for this input structure (very large curvature)
    torch.autograd.gradcheck(f, (head,), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("delta", [-180.0, -179.9, 0.0, 179.9, 180.0, 360.0])
def test_wrap_degrees_range(delta):
    w = float(wrap_degrees(torch.tensor(delta)))
    assert -180.0 <= w < 180.0
