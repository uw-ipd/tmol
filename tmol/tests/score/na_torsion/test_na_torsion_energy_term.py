import numpy
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.score.na_torsion import (
    NaTorsionEnergyTerm,
    na_torsion_subterms,
    polymer_index,
    wrap_degrees,
)
from tmol.score.na_torsion._na_torsion_energy_term import _pose_subterms
from tmol.score.common import ZeroTermPoseScoringModule


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


def test_interaction_only_omits_diagonal_na_torsions(dna_pdb, torch_device):
    term, pose_stack = _term_and_pose(dna_pdb, torch_device)

    scorer = term.render_block_pair_scoring_module(pose_stack, interaction_only=True)

    assert isinstance(scorer, ZeroTermPoseScoringModule)
    assert scorer(pose_stack.coords).shape == (
        2,
        pose_stack.n_poses,
        pose_stack.max_n_blocks,
        pose_stack.max_n_blocks,
    )


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
    from tmol.pose import PoseStackBuilder

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


def _native_pose_scores(term, ps, coords):
    from tmol.score.na_torsion.potentials import na_torsion_pose_score

    p = term.params
    return na_torsion_pose_score(
        coords.flatten(0, -2),
        *ps.na_torsion_pose_params,
        p.backbone_means,
        p.backbone_sdev,
        p.sugar_means,
        p.chi_means,
        p.sdev_sugar,
        p.sdev_chi,
        p.well_pucker,
        p.well_alpha_gamma,
        p.well_bibii_pucker,
        p.well_alphanext_bibii,
        p.well_chi_syn,
        p.is_north,
        p.weight_bb,
        p.weight_chi,
        p.weight_sugar,
        p.pucker_temperature,
        p.bin_blend_sdev,
    )[0]


def _reference_pose_scores(term, ps, coords):
    p = term.params
    bb, chi, sugar, well = _pose_subterms(
        coords.flatten(0, -2),
        *ps.na_torsion_pose_params,
        p.backbone_means,
        p.backbone_sdev,
        p.sugar_means,
        p.chi_means,
        p.sdev_sugar,
        p.sdev_chi,
        p.well_pucker,
        p.well_alpha_gamma,
        p.well_bibii_pucker,
        p.well_alphanext_bibii,
        p.well_chi_syn,
        p.is_north,
        p.pucker_temperature,
        p.bin_blend_sdev,
    )
    poly = polymer_index(ps.na_torsion_pose_params[0])
    harmonic = p.weight_bb[poly] * bb + p.weight_chi[poly] * chi
    harmonic = harmonic + p.weight_sugar[poly] * sugar
    mask = ps.na_torsion_pose_params[1]
    zero = torch.zeros_like(harmonic)
    return torch.stack(
        [
            torch.where(mask, harmonic, zero).sum(dim=1),
            torch.where(mask, well, zero).sum(dim=1),
        ]
    )


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb", "protein_dna_pdb"])
@pytest.mark.parametrize("n_poses", [1, 3])
def test_native_cuda_pose_scores_match_reference(
    n_poses, fixture, request, torch_device
):
    if torch_device.type != "cuda":
        pytest.skip("native NA torsion scoring is CUDA-only")

    from tmol.pose import PoseStackBuilder

    term, ps1 = _term_and_pose(request.getfixturevalue(fixture), torch_device)
    ps = PoseStackBuilder.from_poses([ps1] * n_poses, torch_device)
    term.setup_packed_block_types(ps.packed_block_types)
    term.setup_poses(ps)

    expected = _reference_pose_scores(term, ps, ps.coords)
    actual = _native_pose_scores(term, ps, ps.coords)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-3)


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb", "protein_dna_pdb"])
@pytest.mark.parametrize("n_poses", [1, 3])
def test_native_cuda_pose_gradients_match_reference(
    n_poses, fixture, request, torch_device
):
    if torch_device.type != "cuda":
        pytest.skip("native NA torsion scoring is CUDA-only")

    from tmol.pose import PoseStackBuilder

    term, ps1 = _term_and_pose(request.getfixturevalue(fixture), torch_device)
    ps = PoseStackBuilder.from_poses([ps1] * n_poses, torch_device)
    term.setup_packed_block_types(ps.packed_block_types)
    term.setup_poses(ps)
    reference_coords = ps.coords.detach().clone().requires_grad_(True)
    native_coords = ps.coords.detach().clone().requires_grad_(True)
    reference = _reference_pose_scores(term, ps, reference_coords)
    native = _native_pose_scores(term, ps, native_coords)
    output_weights = torch.linspace(
        -1.2,
        0.7,
        2 * n_poses,
        dtype=ps.coords.dtype,
        device=torch_device,
    ).reshape(2, n_poses)
    (reference_grad,) = torch.autograd.grad(
        torch.sum(reference * output_weights), reference_coords
    )
    (native_grad,) = torch.autograd.grad(
        torch.sum(native * output_weights), native_coords
    )

    torch.testing.assert_close(native, reference, rtol=2e-5, atol=2e-3)
    torch.testing.assert_close(native_grad, reference_grad, rtol=5e-4, atol=5e-2)


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb", "protein_dna_pdb"])
def test_native_cuda_pose_scoring_is_graph_capture_safe(
    fixture, request, default_database, torch_device
):
    if torch_device.type != "cuda":
        pytest.skip("CUDA graph capture requires CUDA")

    from tmol.pose import PoseStackBuilder
    from tmol.score import ScoreFunction

    ps1 = pose_stack_from_pdb(request.getfixturevalue(fixture), torch_device)
    ps = PoseStackBuilder.from_poses([ps1] * 3, torch_device)
    sfxn = ScoreFunction(default_database, torch_device)
    for score_type in NaTorsionEnergyTerm.score_types():
        sfxn.set_weight(score_type, 1.0)

    eager = sfxn.render_whole_pose_scoring_module(ps)
    graphed = sfxn.render_whole_pose_scoring_module(ps, cuda_graph="both")
    eager_coords = ps.coords.detach().clone().requires_grad_(True)
    graph_coords = ps.coords.detach().clone().requires_grad_(True)
    eager_score = eager(eager_coords)
    graph_score = graphed(graph_coords)
    (eager_grad,) = torch.autograd.grad(eager_score.sum(), eager_coords)
    (graph_grad,) = torch.autograd.grad(graph_score.sum(), graph_coords)

    torch.testing.assert_close(graph_score, eager_score)
    torch.testing.assert_close(graph_grad, eager_grad)


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
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import build_rotamers
    from tmol.pack.rotamer import IncludeCurrentSampler
    from tmol.score import ScoreFunction
    from tmol.score import ScoreType

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

    rotamer_scorer = sfxn.render_rotamer_scoring_module(ps, rotamer_set)
    per_rot_sparse = rotamer_scorer(rotamer_set.coords).coalesce()
    if torch_device.type == "cuda":
        # Requiring coordinate gradients selects the eager reference path;
        # ordinary search/packing inference uses the native CUDA row kernel.
        eager = rotamer_scorer(
            rotamer_set.coords.detach().clone().requires_grad_(True)
        ).coalesce()
        torch.testing.assert_close(per_rot_sparse.indices(), eager.indices())
        torch.testing.assert_close(
            per_rot_sparse.values(), eager.values(), rtol=2e-5, atol=2e-3
        )
    per_rot = per_rot_sparse.to_dense()
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
