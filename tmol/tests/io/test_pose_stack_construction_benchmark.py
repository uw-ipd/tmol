import numpy
import pytest

from tmol.io.canonical_ordering import (
    canonical_form_from_pdb,
    default_canonical_ordering,
    default_packed_block_types,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.score import beta2016_score_function
from tmol.tests.torch import zero_padded_counts


@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form")
def test_build_pose_stack_from_canonical_form_ubq_benchmark(benchmark, torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, ubq_pdb, torch_device)

    # warmup
    p = pose_stack_from_canonical_form(co, pbt, *canonical_form)
    assert p.coords.shape[0] == 1

    @benchmark
    def create_pose_stack():
        pose_stack = pose_stack_from_canonical_form(co, pbt, *canonical_form)
        return pose_stack


@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form")
def test_build_pose_stack_from_canonical_form_pert_benchmark(benchmark, torch_device, pertuzumab_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pertuzumab_pdb, torch_device)

    # warmup
    p = pose_stack_from_canonical_form(co, pbt, *canonical_form)
    assert p.coords.shape[0] == 1

    @benchmark
    def create_pose_stack():
        pose_stack = pose_stack_from_canonical_form(co, pbt, *canonical_form)
        return pose_stack


@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form_and_score")
def test_build_and_score_ubq_benchmark(benchmark, torch_device, ubq_pdb):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, ubq_pdb, torch_device)

    # warmup
    ps = pose_stack_from_canonical_form(co, pbt, *canonical_form)
    sfxn = beta2016_score_function(torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(ps)
    sc = scorer(ps.coords)
    assert len(sc.shape) == 1

    @benchmark
    def create_and_score_pose_stack():
        pose_stack = pose_stack_from_canonical_form(co, pbt, *canonical_form)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        score = scorer(pose_stack.coords)

        return score


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10]))
@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form")
def test_build_pose_stack_from_canonical_form_pertuzumab_benchmark(
    benchmark,
    pertuzumab_pdb,
    n_poses,
    torch_device,
):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    n_poses = int(n_poses)
    # ch_id, can_rts, coords
    canonical_form = canonical_form_from_pdb(co, pertuzumab_pdb, torch_device)

    canonical_form.chain_id = canonical_form.chain_id.expand(n_poses, -1)
    canonical_form.res_types = canonical_form.res_types.expand(n_poses, -1)
    canonical_form.coords = canonical_form.coords.expand(n_poses, -1, -1, -1)
    canonical_form.chain_labels = numpy.broadcast_to(
        canonical_form.chain_labels,
        (n_poses, canonical_form.chain_labels.shape[1]),
    )
    canonical_form.res_labels = numpy.broadcast_to(
        canonical_form.res_labels,
        (n_poses, canonical_form.res_labels.shape[1]),
    )
    canonical_form.residue_insertion_codes = numpy.broadcast_to(
        canonical_form.residue_insertion_codes,
        (n_poses, canonical_form.residue_insertion_codes.shape[1]),
    )
    canonical_form.atom_occupancy = numpy.broadcast_to(
        canonical_form.atom_occupancy,
        (
            n_poses,
            canonical_form.atom_occupancy.shape[1],
            canonical_form.atom_occupancy.shape[2],
        ),
    )
    canonical_form.atom_b_factor = numpy.broadcast_to(
        canonical_form.atom_b_factor,
        (
            n_poses,
            canonical_form.atom_b_factor.shape[1],
            canonical_form.atom_b_factor.shape[2],
        ),
    )

    # warmup
    p = pose_stack_from_canonical_form(co, pbt, *canonical_form)
    assert p.coords.shape[0] == n_poses

    @benchmark
    def create_pose_stack():
        pose_stack = pose_stack_from_canonical_form(co, pbt, *canonical_form)
        return pose_stack


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30]))
@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form_and_score")
def test_build_and_score_pertuzumab_benchmark(benchmark, pertuzumab_pdb, n_poses, torch_device):
    n_poses = int(n_poses)
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pertuzumab_pdb, torch_device)

    canonical_form.chain_id = canonical_form.chain_id.expand(n_poses, -1)
    canonical_form.res_types = canonical_form.res_types.expand(n_poses, -1)
    canonical_form.coords = canonical_form.coords.expand(n_poses, -1, -1, -1)
    canonical_form.chain_labels = numpy.broadcast_to(
        canonical_form.chain_labels,
        (n_poses, canonical_form.chain_labels.shape[1]),
    )
    canonical_form.res_labels = numpy.broadcast_to(
        canonical_form.res_labels,
        (n_poses, canonical_form.res_labels.shape[1]),
    )
    canonical_form.residue_insertion_codes = numpy.broadcast_to(
        canonical_form.residue_insertion_codes,
        (n_poses, canonical_form.residue_insertion_codes.shape[1]),
    )
    canonical_form.atom_occupancy = numpy.broadcast_to(
        canonical_form.atom_occupancy,
        (
            n_poses,
            canonical_form.atom_occupancy.shape[1],
            canonical_form.atom_occupancy.shape[2],
        ),
    )
    canonical_form.atom_b_factor = numpy.broadcast_to(
        canonical_form.atom_b_factor,
        (
            n_poses,
            canonical_form.atom_b_factor.shape[1],
            canonical_form.atom_b_factor.shape[2],
        ),
    )

    # warmup
    ps = pose_stack_from_canonical_form(co, pbt, *canonical_form)
    sfxn = beta2016_score_function(torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(ps)
    sc = scorer(ps.coords)
    assert len(sc.shape) == 1

    @benchmark
    def create_and_score_pose_stack():
        pose_stack = pose_stack_from_canonical_form(co, pbt, *canonical_form)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        score = scorer(pose_stack.coords)

        return score
