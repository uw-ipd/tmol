import torch
import numpy
import os
import threading
import pytest

from tmol.score import _score_function as score_function_module
from tmol.score import (
    _non_memoized_beta2016,
    ScoreFunction,
    ScoreType,
)
from tmol.score._score_function import (
    BlockPairScoringModule,
    RotamerScoringModule,
    WholePoseScoringModule,
)
from tmol.score.common import ZeroTermPoseScoringModule
from tmol.pose import (
    DEFAULT_ATOM_B_FACTOR,
    DEFAULT_ATOM_OCCUPANCY,
    PoseStackBuilder,
)
from tmol import (
    pose_stack_from_pdb,
    beta2016_score_function,
    canonical_form_from_pdb,
    default_canonical_ordering,
    default_packed_block_types,
    pose_stack_from_canonical_form,
)


def test_pose_score_smoke(ubq_pdb, default_database, torch_device):
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    pose_stack100 = PoseStackBuilder.from_poses([pose_stack1] * 100, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    scorer = sfxn.render_whole_pose_scoring_module(pose_stack100)

    scores = scorer(pose_stack100.coords)

    assert scores is not None


def test_score_function_resolves_unindexed_cuda(default_database, torch_device):
    if torch_device.type != "cuda":
        pytest.skip("Requires CUDA")

    sfxn = ScoreFunction(default_database, torch.device("cuda"))

    assert sfxn._device == torch_device
    assert sfxn._weights.device == torch_device


def test_whole_pose_gradients_respect_per_pose_upstream_weights(
    ubq_pdb, default_database, torch_device
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    stack = PoseStackBuilder.from_poses([pose] * 3, torch_device)
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    single_scorer = sfxn.render_whole_pose_scoring_module(pose)
    single_coords = pose.coords.detach().clone().requires_grad_(True)
    single_grad = torch.autograd.grad(
        single_scorer(single_coords).sum(), single_coords
    )[0]

    stack_scorer = sfxn.render_whole_pose_scoring_module(stack)
    stack_coords = stack.coords.detach().clone().requires_grad_(True)
    upstream = torch.tensor([0.25, -1.5, 2.0], device=torch_device)
    stack_grad = torch.autograd.grad(
        (stack_scorer(stack_coords) * upstream).sum(), stack_coords
    )[0]

    torch.testing.assert_close(
        stack_grad,
        single_grad.expand_as(stack_grad) * upstream[:, None, None],
        atol=1e-5,
        rtol=1e-5,
    )


def test_zero_weight_does_not_construct_energy_term(default_database, torch_device):
    sfxn = ScoreFunction(default_database, torch_device)

    sfxn.set_weight(ScoreType.constraint, 0.0)

    assert sfxn.all_terms() == []
    assert sfxn.get_weight(ScoreType.constraint) == 0

    sfxn.set_weight(ScoreType.constraint, 1.0)
    assert len(sfxn.all_terms()) == 1

    sfxn.set_weight(ScoreType.constraint, 0.0)
    assert sfxn.all_terms() == []


def test_zero_weight_removes_term_only_after_last_subterm(
    default_database, torch_device
):
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 1.0)

    sfxn.set_weight(ScoreType.fa_ljrep, 0.0)
    assert len(sfxn.all_terms()) == 1

    sfxn.set_weight(ScoreType.fa_ljatr, 0.0)
    assert sfxn.all_terms() == []


def test_weight_tensor_refresh_preserves_score_type_order(
    default_database, torch_device
):
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 2.0)
    sfxn.set_weight(ScoreType.fa_lk, 3.0)

    torch.testing.assert_close(
        sfxn.weights_tensor(),
        torch.tensor([1.0, 2.0, 3.0], device=torch_device),
    )

    indices = sfxn._weight_indices_tensor
    sfxn.set_weight(ScoreType.fa_ljrep, 4.0)
    torch.testing.assert_close(
        sfxn.weights_tensor(),
        torch.tensor([1.0, 4.0, 3.0], device=torch_device),
    )
    assert sfxn._weight_indices_tensor is indices


def test_packed_block_setup_is_reused_and_cross_score_function_safe(
    ubq_pdb, default_database, torch_device, monkeypatch
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.ref, 1.0)
    term = sfxn.all_terms()[0]
    original = term.setup_block_type
    calls = 0

    def counted(block_type):
        nonlocal calls
        calls += 1
        return original(block_type)

    monkeypatch.setattr(term, "setup_block_type", counted)
    sfxn.pre_work_initialization(pose)
    first_calls = calls
    assert first_calls == len(pose.packed_block_types.active_block_types)

    sfxn.pre_work_initialization(pose)
    assert calls == first_calls

    # Packed block types are shared. Another score function may replace
    # option-dependent annotations, so its setup must invalidate this cache.
    other = ScoreFunction(default_database, torch_device)
    other.set_weight(ScoreType.ref, 1.0)
    other.pre_work_initialization(pose)
    sfxn.pre_work_initialization(pose)
    assert calls == 2 * first_calls


def test_no_grad_scoring_detaches_coordinates():
    class RecordingTerm(torch.nn.Module):
        def forward(self, coords):
            self.input_requires_grad = coords.requires_grad
            return coords.sum().reshape(1, 1)

    term = RecordingTerm()
    scorer = WholePoseScoringModule(torch.ones(1), [term])
    coords = torch.ones(1, requires_grad=True)

    scorer(coords)
    assert term.input_requires_grad

    with torch.no_grad():
        scorer(coords)
    assert not term.input_requires_grad


def test_cpu_whole_pose_terms_run_concurrently_and_preserve_modes(monkeypatch):
    monkeypatch.setattr(
        score_function_module,
        "_CPU_PARALLEL_SCORE_BACKWARD_MIN_COORD_ELEMENTS",
        0,
    )

    class RecordingTerm(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.calls = []

        def forward(self, coords):
            self.calls.append(
                (
                    threading.get_ident(),
                    torch.is_grad_enabled(),
                    torch.is_inference_mode_enabled(),
                )
            )
            return (self.scale * coords.square().sum()).reshape(1, 1)

    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    terms = [RecordingTerm(scale) for scale in (1.0, 2.0, 3.0, 4.0)]
    scorer = WholePoseScoringModule(torch.ones(4), terms)
    coords = torch.arange(3.0, requires_grad=True)

    score = scorer(coords)
    score.backward(retain_graph=True)

    torch.testing.assert_close(score, torch.tensor([50.0]))
    torch.testing.assert_close(coords.grad, 20 * coords.detach())
    coords.grad = None
    score.backward()
    torch.testing.assert_close(coords.grad, 20 * coords.detach())
    assert all(
        call[0] != threading.get_ident() for term in terms for call in term.calls
    )
    assert all(call[1:] == (True, False) for term in terms for call in term.calls)

    for term in terms:
        term.calls.clear()
    with torch.inference_mode():
        scorer(coords)
    assert all(call[1:] == (False, True) for term in terms for call in term.calls)

    for term in terms:
        term.calls.clear()
    scorer._cpu_term_workers = 0
    scorer(coords)
    assert all(
        call[0] == threading.get_ident() for term in terms for call in term.calls
    )


def test_cpu_parallel_whole_pose_gradient_matches_serial_order(
    ubq_pdb, default_database, torch_device, monkeypatch
):
    if torch_device.type != "cpu":
        pytest.skip("CPU term-parallel scoring test")

    monkeypatch.setattr(
        score_function_module,
        "_CPU_PARALLEL_SCORE_BACKWARD_MIN_COORD_ELEMENTS",
        0,
    )

    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)
    scorer = beta2016_score_function(
        torch_device, default_database
    ).render_whole_pose_scoring_module(pose)

    def score_and_gradient():
        coords = pose.coords.detach().clone().requires_grad_(True)
        score = scorer(coords)
        (gradient,) = torch.autograd.grad(score.sum(), coords)
        return score, gradient

    parallel_score, parallel_gradient = score_and_gradient()
    scorer._cpu_term_workers = 0
    serial_score, serial_gradient = score_and_gradient()

    assert torch.equal(parallel_score, serial_score)
    assert torch.equal(parallel_gradient, serial_gradient)


def test_rotamer_scorer_combines_identical_sparse_layouts(
    torch_device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        score_function_module, "_CUDA_ROTAMER_LAYOUT_DEDUP_MIN_BYTES", 0
    )

    class SparseTerm(torch.nn.Module):
        def __init__(self, scores: torch.Tensor, indices: torch.Tensor) -> None:
            super().__init__()
            self.scores = scores
            self.indices = indices
            self.n_poses = 1
            self.n_rots = 3

        def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.scores * coords, self.indices

    shared = torch.tensor(
        [[0, 0], [0, 1], [1, 0]], dtype=torch.int32, device=torch_device
    )
    distinct = torch.tensor(
        [[0, 0], [0, 2], [2, 1]], dtype=torch.int32, device=torch_device
    )
    terms = [
        SparseTerm(torch.tensor([[1.0, 2.0]], device=torch_device), shared),
        SparseTerm(torch.tensor([[3.0, 4.0]], device=torch_device), shared.clone()),
        SparseTerm(torch.tensor([[5.0, 6.0]], device=torch_device), distinct),
    ]
    scorer = RotamerScoringModule(
        torch.tensor([1.0, 2.0, 4.0], device=torch_device), terms
    )
    coords = torch.ones((), device=torch_device, requires_grad=True)

    scores = scorer(coords)
    dense_scores = scores.to_dense()

    assert scores._nnz() == 4
    torch.testing.assert_close(
        dense_scores,
        torch.tensor(
            [[[0.0, 7.0, 20.0], [10.0, 0.0, 0.0], [0.0, 24.0, 0.0]]],
            device=torch_device,
        ),
    )
    dense_scores.sum().backward()
    torch.testing.assert_close(coords.grad, torch.tensor(61.0, device=torch_device))


def test_cpu_rotamer_scorer_coalesces_subset_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(score_function_module, "_CPU_ROTAMER_SORTED_LAYOUT_MIN_NNZ", 0)

    class SparseTerm(torch.nn.Module):
        def __init__(self, scores: torch.Tensor, indices: torch.Tensor) -> None:
            super().__init__()
            self.scores = scores
            self.indices = indices
            self.n_poses = 1
            self.n_rots = 3

        def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.scores * coords, self.indices

    complete = torch.tensor([[0, 0, 0], [2, 0, 1], [2, 0, 1]], dtype=torch.int32)
    subset = complete[:, [0, 2]]
    scorer = RotamerScoringModule(
        torch.tensor([1.0, 2.0]),
        [
            SparseTerm(torch.tensor([[1.0, 2.0, 3.0]]), complete),
            SparseTerm(torch.tensor([[4.0, 5.0]]), subset),
        ],
    )
    coords = torch.ones((), requires_grad=True)

    scores = scorer(coords)

    assert scores.is_coalesced()
    torch.testing.assert_close(
        scores.to_dense(),
        torch.tensor([[[2.0, 0.0, 0.0], [0.0, 13.0, 0.0], [0.0, 0.0, 9.0]]]),
    )
    scores.values().sum().backward()
    torch.testing.assert_close(coords.grad, torch.tensor(24.0))


def test_cpu_rotamer_terms_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    barrier = threading.Barrier(2, timeout=2)

    class SparseTerm(torch.nn.Module):
        n_poses = 1
        n_rots = 1

        def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            barrier.wait()
            scores = coords.reshape(1, 1)
            indices = torch.zeros((3, 1), dtype=torch.int32)
            return scores, indices

    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    scorer = RotamerScoringModule(torch.ones(2), [SparseTerm(), SparseTerm()])

    scores = scorer(torch.ones(()))

    torch.testing.assert_close(scores.to_dense(), torch.tensor([[[2.0]]]))


def test_cuda_graphed_protein_score_matches_eager(ubq_pdb, torch_device):
    if torch_device.type != "cuda":
        pytest.skip("CUDA graph test")

    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)
    sfxn = beta2016_score_function(torch_device)
    eager = sfxn.render_whole_pose_scoring_module(pose_stack)

    eager_coords = pose_stack.coords.detach().clone().requires_grad_(True)
    expected_score = eager(eager_coords)
    (expected_grad,) = torch.autograd.grad(expected_score.sum(), eager_coords)
    expected_terms = eager(pose_stack.coords, sum_terms=False, apply_weights=False)

    graphed = sfxn.render_whole_pose_scoring_module(pose_stack, cuda_graph=True)
    graph_coords = pose_stack.coords.detach().clone().requires_grad_(True)
    graph_score = graphed(graph_coords)
    (graph_grad,) = torch.autograd.grad(graph_score.sum(), graph_coords)

    torch.testing.assert_close(graph_score, expected_score, rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(graph_grad, expected_grad, rtol=1e-4, atol=1e-3)

    # A later replay must not mutate a score retained by an optimizer.
    retained_graph_score = graph_score.detach().clone()
    replay_coords = pose_stack.coords.detach().clone().requires_grad_(True)
    graphed(replay_coords)
    torch.testing.assert_close(graph_score, retained_graph_score)
    # Non-default reductions intentionally retain their normal eager semantics.
    torch.testing.assert_close(
        graphed(pose_stack.coords, sum_terms=False, apply_weights=False),
        expected_terms,
    )

    changed_coords = pose_stack.coords.detach().clone()
    changed_coords[0, 0, 0] += 0.1
    torch.testing.assert_close(
        graphed(changed_coords), eager(changed_coords), rtol=1e-5, atol=1e-3
    )


def test_block_pair_scoring_matches_whole_pose(ubq_pdb, default_database, torch_device):
    # passing the database bypasses the memoized score function, which the
    # set_weight below would otherwise mutate for the rest of the session
    sfxn = beta2016_score_function(torch_device, default_database)

    # set a weight to ensure weights are being handled properly
    sfxn.set_weight(ScoreType.fa_ljrep, 3)

    # test multiple poses to ensure score is being attributed the same
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)
    pose_stack2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=20)
    pose_stack3 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=30)
    pose_stack = PoseStackBuilder.from_poses(
        [pose_stack1, pose_stack2, pose_stack3], torch_device
    )

    full_scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    block_scorer = sfxn.render_block_pair_scoring_module(pose_stack)

    # check individual terms
    full_score = full_scorer(pose_stack.coords, sum_terms=False)
    block_score = block_scorer(pose_stack.coords, sum_terms=False)
    torch.testing.assert_close(
        full_score, torch.sum(block_score, dim=(2, 3)), atol=1e-3, rtol=1e-3
    )

    # check total pose values
    full_score = full_scorer(pose_stack.coords, sum_terms=True)
    block_score = block_scorer(pose_stack.coords, sum_terms=True)
    torch.testing.assert_close(
        full_score, torch.sum(block_score, dim=(1, 2)), atol=1e-3, rtol=1e-3
    )


def test_block_pair_gradients_respect_sparse_upstream_weights(
    ubq_pdb, default_database, torch_device
):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)
    scorer = beta2016_score_function(
        torch_device, default_database
    ).render_block_pair_scoring_module(pose_stack)

    n_blocks = pose_stack.max_n_blocks
    split = n_blocks // 2
    cross_mask = torch.zeros(
        (1, n_blocks, n_blocks), device=torch_device, dtype=pose_stack.coords.dtype
    )
    cross_mask[:, :split, split:] = 1
    cross_mask[:, split:, :split] = 1
    all_pairs = torch.ones_like(cross_mask)

    def gradient(upstream: torch.Tensor) -> torch.Tensor:
        coords = pose_stack.coords.detach().clone().requires_grad_(True)
        score = (scorer(coords) * upstream).sum()
        return torch.autograd.grad(score, coords)[0]

    sparse_gradient = gradient(cross_mask)
    reference_gradient = gradient(all_pairs) - gradient(all_pairs - cross_mask)

    torch.testing.assert_close(
        sparse_gradient, reference_gradient, atol=2e-3, rtol=2e-3
    )


def test_block_pair_scoring_supports_only_invariant_zero_terms(torch_device):
    scorer = BlockPairScoringModule(
        torch.ones(2, device=torch_device),
        [ZeroTermPoseScoringModule("zero", 2, 3, torch_device, 4)],
    )
    coords = torch.zeros((3, 1, 3), device=torch_device, requires_grad=True)

    scores = scorer(coords)

    assert scores.shape == (3, 4, 4)
    assert torch.count_nonzero(scores) == 0
    scores.sum().backward()


def test_virtual_residue_scoring(ubq_pdb, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)

    def pose_stack_of_nres(nres, add_vrt):
        def xyz(x, y, z):
            return torch.tensor((x, y, z), dtype=torch.float32, device=torch_device)

        canonical_form = canonical_form_from_pdb(
            co, ubq_pdb, torch_device, residue_start=0, residue_end=nres
        )
        if add_vrt:
            vrt_co_ind = co.restype_io_equiv_classes.index("VRT")
            # print("vrt_co_ind", vrt_co_ind)
            orig_coords = canonical_form.coords
            ocs = orig_coords.shape
            new_coords = torch.full(
                (ocs[0], ocs[1] + 1, ocs[2], ocs[3]),
                numpy.nan,
                dtype=torch.float32,
                device=torch_device,
            )
            new_coords[0, :-1, :, :] = orig_coords
            # Let's put the VRT right in the center of res "2", ILE 3
            new_coords[0, -1, 0, :] = xyz(26.849, 29.656, 6.217)
            new_coords[0, -1, 1, :] = xyz(26.849, 29.656, 6.217) + xyz(1.0, 0.0, 0.0)
            new_coords[0, -1, 2, :] = xyz(26.849, 29.656, 6.217) + xyz(0.0, 1.0, 0.0)
            orig_chain_id = canonical_form.chain_id

            ocis = orig_chain_id.shape
            new_chain_id = torch.zeros(
                (ocis[0], ocis[1] + 1), dtype=torch.int32, device=torch_device
            )
            new_chain_id[0, :-1] = orig_chain_id
            new_chain_id[0, -1] = (
                orig_chain_id[0, -1] + 1
            )  # give the vrt res a new chain id

            orig_restypes = canonical_form.res_types
            ors = orig_restypes.shape
            new_restypes = torch.full(
                (ors[0], ors[1] + 1), -1, dtype=torch.int32, device=torch_device
            )
            new_restypes[0, :-1] = orig_restypes
            new_restypes[0, -1] = vrt_co_ind

            orig_chain_labels = canonical_form.chain_labels
            ocls = orig_chain_labels.shape
            new_chain_labels = numpy.full((ocls[0], ocls[1] + 1), "", dtype=object)
            new_chain_labels[0, :-1] = orig_chain_labels
            new_chain_labels[0, -1] = "V"

            new_res_labels = numpy.full((ocls[0], ocls[1] + 1), 0, dtype=int)
            new_res_labels[0, :-1] = canonical_form.res_labels
            new_res_labels[0, -1] = nres + 1
            new_res_ins_codes = numpy.full((ocls[0], ocls[1] + 1), "", dtype=object)
            new_res_ins_codes[0, :-1] = canonical_form.residue_insertion_codes
            new_res_ins_codes[0, -1] = ""

            orig_occupancy = canonical_form.atom_occupancy
            new_atom_occupancy = numpy.full(
                (
                    orig_occupancy.shape[0],
                    orig_occupancy.shape[1] + 1,
                    orig_occupancy.shape[2],
                ),
                DEFAULT_ATOM_OCCUPANCY,
                dtype=numpy.float32,
            )
            new_atom_b_factor = numpy.full(
                (
                    orig_occupancy.shape[0],
                    orig_occupancy.shape[1] + 1,
                    orig_occupancy.shape[2],
                ),
                DEFAULT_ATOM_B_FACTOR,
                dtype=numpy.float32,
            )
            new_atom_occupancy[0, :-1, :] = canonical_form.atom_occupancy
            new_atom_b_factor[0, :-1, :] = canonical_form.atom_b_factor

            canonical_form.coords = new_coords
            canonical_form.chain_id = new_chain_id
            canonical_form.res_types = new_restypes
            canonical_form.chain_labels = new_chain_labels
            canonical_form.res_labels = new_res_labels
            canonical_form.residue_insertion_codes = new_res_ins_codes
            canonical_form.atom_occupancy = new_atom_occupancy
            canonical_form.atom_b_factor = new_atom_b_factor

        return pose_stack_from_canonical_form(co, pbt, *canonical_form)

    ps_wo_vrt = PoseStackBuilder.from_poses(
        [pose_stack_of_nres(x, False) for x in [4, 6, 5]], torch_device
    )
    ps_w_vrt = PoseStackBuilder.from_poses(
        [pose_stack_of_nres(x, True) for x in [4, 6, 5]], torch_device
    )

    sfxn = beta2016_score_function(torch_device)
    scorer_wo_vrt = sfxn.render_whole_pose_scoring_module(ps_wo_vrt)
    scores_wo_vrt = scorer_wo_vrt(ps_wo_vrt.coords)

    scorer_w_vrt = sfxn.render_whole_pose_scoring_module(ps_w_vrt)
    scores_w_vrt = scorer_w_vrt(ps_w_vrt.coords)

    unweighted_scores_wo_vrt = scorer_wo_vrt.unweighted_scores(ps_wo_vrt.coords)
    unweighted_scores_w_vrt = scorer_w_vrt.unweighted_scores(ps_w_vrt.coords)

    torch.testing.assert_close(scores_wo_vrt, scores_w_vrt)
    torch.testing.assert_close(unweighted_scores_wo_vrt, unweighted_scores_w_vrt)


def _assert_matches_gold(score_map, gold_map, score_types, rtol, atol):
    """Compare every score type, reporting all mismatches as pastable source."""
    bad = [
        st
        for st in score_types
        if not numpy.allclose(score_map[st], gold_map[st], rtol=rtol, atol=atol)
    ]
    if bad:
        lines = "\n".join(
            f"        ScoreType.{st.name}: n([{float(score_map[st][0]):.6f}]),"
            for st in score_types
        )
        raise AssertionError(
            f"{len(bad)} score type(s) differ: {[st.name for st in bad]}\n"
            f"observed map:\n{lines}"
        )


def test_soft_score_function_all_score_types(ubq_pdb, default_database, torch_device):
    ps = pose_stack_from_pdb(ubq_pdb, torch_device)

    _weights_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "database",
        "score_functions",
        "beta_soft.sfxn",
    )
    sfxn = ScoreFunction.from_sfxn_file(_weights_path, default_database, torch_device)

    wpsm = sfxn.render_whole_pose_scoring_module(ps)
    term_scores = wpsm(ps.coords, sum_terms=False, apply_weights=False)
    score_types = sfxn.all_score_types()
    unweighted_score_map = {
        st: term_scores[i, :].detach().cpu().numpy() for i, st in enumerate(score_types)
    }

    def n(x):
        return numpy.array(x)

    # fa_ljatr/fa_elec match the beta2016 baselines: soft_rep only softens the
    # LJ repulsive shoulder, leaving the attractive and electrostatic terms
    # unchanged. gen_torsions is enabled in beta_soft.yaml on this branch.
    gold_score_map = {
        ScoreType.cart_lengths: n([38.056973]),
        ScoreType.cart_angles: n([183.9738]),
        ScoreType.cart_torsions: n([46.02357]),
        ScoreType.cart_impropers: n([9.430529]),
        ScoreType.cart_hxltorsions: n([47.41971]),
        ScoreType.disulfide: n([0.0]),
        ScoreType.fa_ljatr: n([-417.02362]),
        ScoreType.fa_ljrep: n([39.92654]),
        ScoreType.fa_lk: n([301.93347]),
        ScoreType.fa_elec: n([-134.03497]),
        ScoreType.hbond: n([-55.675613]),
        ScoreType.lk_ball_iso: n([422.03955]),
        ScoreType.lk_ball: n([172.19647]),
        ScoreType.lk_bridge: n([1.5817453]),
        ScoreType.lk_bridge_uncpl: n([11.031567]),
        ScoreType.rama: n([-12.743372]),
        ScoreType.omega: n([4.100171]),
        ScoreType.ref: n([18.7695]),
        ScoreType.dunbrack_rot: n([70.64968]),
        ScoreType.dunbrack_rotdev: n([240.31009]),
        ScoreType.dunbrack_semirot: n([99.660904]),
        ScoreType.gen_torsions: n([0.0]),
        ScoreType.na_torsion: n([0.0]),
        ScoreType.na_torsion_well: n([0.0]),
    }
    # This test runs on both cpu and cuda; summed full-pose energies drift at the
    # ~1e-3 level in float32 across devices (e.g. omega), so the tolerance is
    # looser than the cpu-only beta2016 golden test below.
    _assert_matches_gold(
        unweighted_score_map, gold_score_map, score_types, rtol=1e-3, atol=1e-3
    )


def test_score_function_all_score_types(ubq_pdb):
    device = torch.device("cpu")
    ps = pose_stack_from_pdb(ubq_pdb, device)
    sfxn = beta2016_score_function(device)

    wpsm = sfxn.render_whole_pose_scoring_module(ps)
    unweighted_scores = wpsm.unweighted_scores(ps.coords)
    score_types = sfxn.all_score_types()
    unweighted_score_map = {
        st: unweighted_scores[i, :].detach().cpu().numpy()
        for i, st in enumerate(score_types)
    }

    def n(x):
        return numpy.array(x)

    gold_score_map = {
        ScoreType.cart_lengths: n([38.056973]),
        ScoreType.cart_angles: n([183.9738]),
        ScoreType.cart_torsions: n([46.02357]),
        ScoreType.cart_impropers: n([9.430529]),
        ScoreType.cart_hxltorsions: n([47.41971]),
        ScoreType.disulfide: n([0.0]),
        ScoreType.fa_ljatr: n([-417.02362]),
        ScoreType.fa_ljrep: n([240.7147]),
        ScoreType.fa_lk: n([301.93347]),
        ScoreType.fa_elec: n([-134.03497]),
        ScoreType.hbond: n([-55.675613]),
        ScoreType.lk_ball_iso: n([422.03955]),
        ScoreType.lk_ball: n([172.19647]),
        ScoreType.lk_bridge: n([1.5817453]),
        ScoreType.lk_bridge_uncpl: n([11.031567]),
        ScoreType.rama: n([-12.743372]),
        ScoreType.omega: n([4.100171]),
        ScoreType.ref: n([-41.275]),
        ScoreType.dunbrack_rot: n([70.64968]),
        ScoreType.dunbrack_rotdev: n([240.31009]),
        ScoreType.dunbrack_semirot: n([99.660904]),
        ScoreType.gen_torsions: n([0.0]),
        ScoreType.na_torsion: n([0.0]),
        ScoreType.na_torsion_well: n([0.0]),
    }
    _assert_matches_gold(
        unweighted_score_map, gold_score_map, score_types, rtol=1e-4, atol=1e-4
    )


def test_score_function_all_score_types_protein_dna(protein_dna_pdb):
    """Golden values for a protein-DNA complex."""
    device = torch.device("cpu")
    ps = pose_stack_from_pdb(protein_dna_pdb, device)
    sfxn = beta2016_score_function(device)

    wpsm = sfxn.render_whole_pose_scoring_module(ps)
    unweighted_scores = wpsm.unweighted_scores(ps.coords)
    score_types = sfxn.all_score_types()
    unweighted_score_map = {
        st: unweighted_scores[i, :].detach().cpu().numpy()
        for i, st in enumerate(score_types)
    }

    def n(x):
        return numpy.array(x)

    gold_score_map = {
        ScoreType.fa_ljatr: n([-1210.854248]),
        ScoreType.fa_ljrep: n([805.994080]),
        ScoreType.fa_lk: n([808.497986]),
        ScoreType.fa_elec: n([-337.304199]),
        ScoreType.hbond: n([-173.997391]),
        ScoreType.cart_lengths: n([157.134415]),
        ScoreType.cart_angles: n([963.831909]),
        ScoreType.cart_torsions: n([134.518875]),
        ScoreType.cart_impropers: n([11.086108]),
        ScoreType.cart_hxltorsions: n([26.123291]),
        ScoreType.disulfide: n([0.0]),
        ScoreType.rama: n([91.406868]),
        ScoreType.omega: n([133.222443]),
        ScoreType.dunbrack_rot: n([198.613831]),
        ScoreType.dunbrack_rotdev: n([553.206238]),
        ScoreType.dunbrack_semirot: n([175.140625]),
        ScoreType.lk_ball_iso: n([1092.919800]),
        ScoreType.lk_ball: n([401.814789]),
        ScoreType.lk_bridge: n([2.487589]),
        ScoreType.lk_bridge_uncpl: n([23.090977]),
        ScoreType.ref: n([-55.255344]),
        ScoreType.gen_torsions: n([0.0]),
        ScoreType.na_torsion: n([367.353271]),
        ScoreType.na_torsion_well: n([57.982567]),
    }
    _assert_matches_gold(
        unweighted_score_map, gold_score_map, score_types, rtol=1e-4, atol=1e-4
    )


def test_score_function_one_body_terms_getter():
    from tmol.score.dunbrack import DunbrackEnergyTerm
    from tmol.score.ref import RefEnergyTerm

    device = torch.device("cpu")
    sfxn = _non_memoized_beta2016(device)
    assert sfxn._one_body_terms_out_of_date

    terms_1b = sfxn.one_body_terms()
    assert not sfxn._one_body_terms_out_of_date

    valid_one_body_terms = [DunbrackEnergyTerm, RefEnergyTerm]
    for term in terms_1b:
        found = False
        for valid_option in valid_one_body_terms:
            if isinstance(term, valid_option):
                found = True
                break
        assert found


def test_score_function_two_body_terms_getter():
    from tmol.score.backbone_torsion import (
        BackboneTorsionEnergyTerm,
    )
    from tmol.score.cartbonded import CartBondedEnergyTerm
    from tmol.score.disulfide import DisulfideEnergyTerm
    from tmol.score.na_torsion import (
        NaTorsionEnergyTerm,
    )
    from tmol.score.elec import ElecEnergyTerm
    from tmol.score.genbonded import GenBondedEnergyTerm
    from tmol.score.hbond import HBondEnergyTerm
    from tmol.score.ljlk import LJLKEnergyTerm
    from tmol.score.lk_ball import LKBallEnergyTerm

    device = torch.device("cpu")
    sfxn = _non_memoized_beta2016(device)
    assert sfxn._two_body_terms_out_of_date

    terms_2b = sfxn.two_body_terms()
    assert not sfxn._two_body_terms_out_of_date

    valid_two_body_terms = [
        BackboneTorsionEnergyTerm,
        CartBondedEnergyTerm,
        DisulfideEnergyTerm,
        NaTorsionEnergyTerm,
        ElecEnergyTerm,
        GenBondedEnergyTerm,
        HBondEnergyTerm,
        LJLKEnergyTerm,
        LKBallEnergyTerm,
    ]
    for term in terms_2b:
        found = False
        for valid_option in valid_two_body_terms:
            if isinstance(term, valid_option):
                found = True
                break
        assert found


def test_score_function_all_terms_getter():
    from tmol.score.backbone_torsion import (
        BackboneTorsionEnergyTerm,
    )
    from tmol.score.cartbonded import CartBondedEnergyTerm
    from tmol.score.disulfide import DisulfideEnergyTerm
    from tmol.score.na_torsion import (
        NaTorsionEnergyTerm,
    )
    from tmol.score.dunbrack import DunbrackEnergyTerm
    from tmol.score.elec import ElecEnergyTerm
    from tmol.score.genbonded import GenBondedEnergyTerm
    from tmol.score.hbond import HBondEnergyTerm
    from tmol.score.ljlk import LJLKEnergyTerm
    from tmol.score.lk_ball import LKBallEnergyTerm
    from tmol.score.ref import RefEnergyTerm

    device = torch.device("cpu")
    sfxn = _non_memoized_beta2016(device)
    assert sfxn._all_terms_out_of_date

    all_terms = sfxn.all_terms()
    assert not sfxn._all_terms_out_of_date

    valid_terms = [
        DunbrackEnergyTerm,
        RefEnergyTerm,
        BackboneTorsionEnergyTerm,
        CartBondedEnergyTerm,
        DisulfideEnergyTerm,
        NaTorsionEnergyTerm,
        ElecEnergyTerm,
        GenBondedEnergyTerm,
        HBondEnergyTerm,
        LJLKEnergyTerm,
        LKBallEnergyTerm,
    ]
    for term in all_terms:
        found = False
        for valid_option in valid_terms:
            if isinstance(term, valid_option):
                found = True
                break
        assert found
