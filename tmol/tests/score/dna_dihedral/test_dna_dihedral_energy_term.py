import numpy
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.score.dna_dihedral.dna_dihedral_energy_term import (
    DnaDihedralEnergyTerm,
    dna_dihedral_subterms,
)
from tmol.score.dna_dihedral.potentials import pucker_weights, wrap_degrees


def _term_and_pose(pdb, torch_device):
    ps = pose_stack_from_pdb(pdb, torch_device)
    term = DnaDihedralEnergyTerm(ParameterDatabase.get_default(), torch_device)
    for bt in ps.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(ps.packed_block_types)
    term.setup_poses(ps)
    return term, ps


def test_smoke(default_database, torch_device):
    term = DnaDihedralEnergyTerm(param_db=default_database, device=torch_device)
    assert term.device == torch_device


def test_all_dna_block_types_are_scoreable(dna_pdb, torch_device):
    """Every DNA type including the terminus variants must score; the 5' patch
    drops P, which removes alpha and beta but must not disable the residue."""
    term, ps = _term_and_pose(dna_pdb, torch_device)
    pbt = ps.packed_block_types
    scoreable = {
        bt.name
        for bt, b in zip(pbt.active_block_types, pbt.dna_dihedral_base.tolist())
        if b >= 0
    }
    for base in ("DA", "DC", "DG", "DT"):
        for suffix in ("", ":dna5prime", ":dna3prime", ":dna5prime:dna3prime"):
            assert base + suffix in scoreable

    real = ps.block_type_ind64[0] >= 0
    bases = pbt.dna_dihedral_base.to(torch.int64)[ps.block_type_ind64[0].clamp_min(0)]
    assert int((bases[real] >= 0).sum()) == 24  # 1BNA is 2 x 12 nt


def test_scores_dna_and_ignores_protein(dna_pdb, ubq_pdb, torch_device):
    term, ps = _term_and_pose(dna_pdb, torch_device)
    dna = term.render_whole_pose_scoring_module(ps)(ps.coords)
    assert torch.isfinite(dna).all()
    assert float(dna) > 0

    term, ps = _term_and_pose(ubq_pdb, torch_device)
    protein = term.render_whole_pose_scoring_module(ps)(ps.coords)
    assert float(protein) == 0.0


@pytest.mark.parametrize("n_poses", [1, 3, 7])
def test_stacked_poses_scale_linearly(n_poses, dna_pdb, torch_device):
    """Identical poses must give exactly n times the single-pose energy.

    Guards the uaid resolution, whose reshapes have to keep pose and block as
    separate dimensions; collapsing them still broadcasts at a single pose.
    """
    from tmol.pose.pose_stack_builder import PoseStackBuilder

    term, ps1 = _term_and_pose(dna_pdb, torch_device)
    one = float(term.render_whole_pose_scoring_module(ps1)(ps1.coords))

    psn = PoseStackBuilder.from_poses([ps1] * n_poses, torch_device)
    term.setup_packed_block_types(psn.packed_block_types)
    term.setup_poses(psn)
    scores = term.render_whole_pose_scoring_module(psn)(psn.coords)

    assert scores.shape[-1] == n_poses
    numpy.testing.assert_allclose(
        scores.detach().cpu().numpy().reshape(-1),
        numpy.full(n_poses, one),
        rtol=1e-5,
    )


def test_subterms_sum_to_the_total(dna_pdb, torch_device):
    term, ps = _term_and_pose(dna_pdb, torch_device)
    _has_dna, *a = term.get_score_term_attributes(ps)
    bb, chi, sugar, mask = dna_dihedral_subterms(
        ps.coords.flatten(0, -2),
        ps.block_type_ind,
        *a[:11],
        a[11],
        a[12],
        a[16],
        a[17],
    )
    p = term.params
    combined = p.weight_bb * bb + p.weight_chi * chi + p.weight_sugar * sugar
    total = term.render_whole_pose_scoring_module(ps)(ps.coords)
    numpy.testing.assert_allclose(
        float(torch.where(mask, combined, torch.zeros_like(combined)).sum()),
        float(total),
        rtol=1e-5,
    )
    assert bool((bb > 0).any()) and bool((chi > 0).any()) and bool((sugar > 0).any())


def test_gradients_are_finite_in_float32(protein_dna_pdb, torch_device):
    """The pucker softmax overflows float32 if written as a ratio of exps."""
    term, ps = _term_and_pose(protein_dna_pdb, torch_device)
    coords = torch.nn.Parameter(ps.coords.clone())
    e = term.render_whole_pose_scoring_module(ps)(coords)
    assert torch.isfinite(e).all()
    e.sum().backward()
    assert torch.isfinite(coords.grad).all()
    assert int((coords.grad.abs().sum(-1) > 0).sum()) > 0


def test_gradcheck(dna_pdb, torch_device):
    term, ps = _term_and_pose(dna_pdb, torch_device)
    module = term.render_whole_pose_scoring_module(ps)
    # a handful of nucleotides is enough and keeps the numeric jacobian cheap
    coords = ps.coords.clone().to(torch.float64)
    n = 40
    head = torch.nn.Parameter(coords[:, :n])

    def f(x):
        full = torch.cat([x, coords[:, n:]], dim=1)
        return module(full).sum()

    torch.autograd.gradcheck(f, (head,), eps=1e-4, atol=1e-4, rtol=1e-3)


def test_pucker_weights_are_normalized_and_sharp(torch_device):
    torch.manual_seed(0)
    ring = torch.randn((32, 5, 3), dtype=torch.float64, device=torch_device)
    w = pucker_weights(ring, 0.05)
    numpy.testing.assert_allclose(w.sum(-1).cpu().numpy(), numpy.ones(32), atol=1e-10)
    assert bool((w >= 0).all())


def test_pucker_softmax_survives_small_temperature(torch_device):
    """T below ~0.0113 overflows a naive exp ratio in float32; the sigmoid and
    max-subtracted softmax used here must stay finite anyway."""
    torch.manual_seed(0)
    ring = torch.randn((16, 5, 3), dtype=torch.float32, device=torch_device)
    for temperature in (0.05, 0.01, 0.002):
        w = pucker_weights(ring, temperature)
        assert torch.isfinite(w).all(), temperature
        numpy.testing.assert_allclose(
            w.sum(-1).cpu().numpy(), numpy.ones(16), atol=1e-5
        )


@pytest.mark.parametrize("delta", [-180.0, -179.9, 0.0, 179.9, 180.0, 360.0])
def test_wrap_degrees_range(delta):
    w = float(wrap_degrees(torch.tensor(delta)))
    assert -180.0 <= w < 180.0
