import pytest
from tmol.tests.torch import zero_padded_counts
from tmol.io.canonical_ordering import canonical_form_from_pdb_lines
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.score import beta2016_score_function


@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form")
def test_build_pose_stack_from_canonical_form_ubq_benchmark(
    benchmark, torch_device, ubq_pdb
):
    canonical_form = canonical_form_from_pdb_lines(ubq_pdb, torch_device)

    # warmup
    p = pose_stack_from_canonical_form(*canonical_form)
    assert p.coords.shape[0] == 1

    @benchmark
    def create_pose_stack():
        pose_stack = pose_stack_from_canonical_form(*canonical_form)
        return pose_stack


@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form")
def test_build_pose_stack_from_canonical_form_pert_benchmark(
    benchmark, torch_device, pertuzumab_pdb
):
    canonical_form = canonical_form_from_pdb_lines(pertuzumab_pdb, torch_device)

    # warmup
    p = pose_stack_from_canonical_form(*canonical_form)
    assert p.coords.shape[0] == 1

    @benchmark
    def create_pose_stack():
        pose_stack = pose_stack_from_canonical_form(*canonical_form)
        return pose_stack


@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form_and_score")
def test_build_and_score_ubq_benchmark(benchmark, torch_device, ubq_pdb):
    canonical_form = canonical_form_from_pdb_lines(ubq_pdb, torch_device)

    # warmup
    ps = pose_stack_from_canonical_form(*canonical_form)
    sfxn = beta2016_score_function(torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(ps)
    sc = scorer(ps.coords)
    assert len(sc.shape) == 1

    @benchmark
    def create_and_score_pose_stack():
        pose_stack = pose_stack_from_canonical_form(*canonical_form)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        score = scorer(pose_stack.coords)

        return score


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30]))
@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form")
def test_build_pose_stack_from_canonical_form_pertuzumab_benchmark(
    benchmark,
    pertuzumab_pdb,
    n_poses,
    torch_device,
):
    n_poses = int(n_poses)
    ch_id, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(
        pertuzumab_pdb, torch_device
    )

    ch_id = ch_id.expand(n_poses, -1)
    can_rts = can_rts.expand(n_poses, -1)
    coords = coords.expand(n_poses, -1, -1, -1)
    at_is_pres = at_is_pres.expand(n_poses, -1, -1)

    # warmup
    p = pose_stack_from_canonical_form(ch_id, can_rts, coords, at_is_pres)
    assert p.coords.shape[0] == n_poses

    @benchmark
    def create_pose_stack():
        pose_stack = pose_stack_from_canonical_form(ch_id, can_rts, coords, at_is_pres)
        return pose_stack


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30]))
@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form_and_score")
def test_build_and_score_pertuzumab_benchmark(
    benchmark, pertuzumab_pdb, n_poses, torch_device
):
    n_poses = int(n_poses)
    ch_id, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(
        pertuzumab_pdb, torch_device
    )

    ch_id = ch_id.expand(n_poses, -1)
    can_rts = can_rts.expand(n_poses, -1)
    coords = coords.expand(n_poses, -1, -1, -1)
    at_is_pres = at_is_pres.expand(n_poses, -1, -1)

    # warmup
    ps = pose_stack_from_canonical_form(ch_id, can_rts, coords, at_is_pres)
    sfxn = beta2016_score_function(torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(ps)
    sc = scorer(ps.coords)
    assert len(sc.shape) == 1

    @benchmark
    def create_and_score_pose_stack():
        pose_stack = pose_stack_from_canonical_form(ch_id, can_rts, coords, at_is_pres)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        score = scorer(pose_stack.coords)

        return score
