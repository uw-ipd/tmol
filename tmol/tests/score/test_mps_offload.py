"""A CPU ScoreFunction offloading its Metal-capable terms matches the host."""

import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import (
    FixedAAChiSampler,
    IncludeCurrentSampler,
    build_rotamers,
)
from tmol.pose import PoseStackBuilder
from tmol.score import ScoreType, beta2016_score_function
from tmol.score._mps_offload import (
    OffloadedRotamerScoringModule,
    OffloadedWholePoseScoringModule,
    mps_offload_enabled,
)

CPU = torch.device("cpu")


@pytest.fixture
def offload(monkeypatch):
    monkeypatch.setenv("TMOL_MPS_OFFLOAD", "1")
    if not mps_offload_enabled():
        pytest.skip("needs an Apple GPU and a Metal-enabled build")


def _host_and_offloaded(render):
    """Render twice, with the offload off and then on."""
    mp = pytest.MonkeyPatch()
    mp.setenv("TMOL_MPS_OFFLOAD", "0")
    host = render()
    mp.setenv("TMOL_MPS_OFFLOAD", "1")
    offloaded = render()
    mp.undo()
    return host, offloaded


@pytest.fixture
def ubq_stack(ubq_pdb):
    one = pose_stack_from_pdb(ubq_pdb, CPU)
    return PoseStackBuilder.from_poses([one] * 3, CPU)


def test_whole_pose_scores_gradients_and_lanes_match_host(
    offload, default_database, ubq_stack
):
    sfxn = beta2016_score_function(CPU, default_database)
    host, offloaded = _host_and_offloaded(
        lambda: sfxn.render_whole_pose_scoring_module(ubq_stack)
    )
    assert isinstance(offloaded, OffloadedWholePoseScoringModule)

    results = []
    for scorer in (host, offloaded):
        coords = ubq_stack.coords.clone().requires_grad_(True)
        energy = scorer(coords)
        energy.sum().backward()
        lanes = scorer(ubq_stack.coords, sum_terms=False, apply_weights=False)
        results.append((energy.detach(), coords.grad, lanes))

    (e_host, g_host, l_host), (e_off, g_off, l_off) = results
    torch.testing.assert_close(e_off, e_host, rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(g_off, g_host, rtol=1e-4, atol=1e-2)
    torch.testing.assert_close(l_off, l_host, rtol=1e-5, atol=1e-3)


def test_in_place_weight_refresh_reaches_both_devices(
    offload, default_database, ubq_stack
):
    sfxn = beta2016_score_function(CPU, default_database)
    host, offloaded = _host_and_offloaded(
        lambda: sfxn.render_whole_pose_scoring_module(ubq_stack)
    )
    # What the minimizers do between relax stages; both halves must see it.
    halved = sfxn.weights_tensor().unsqueeze(1) * 0.5
    with torch.no_grad():
        host.weights.copy_(halved)
        offloaded.weights.copy_(halved)
    torch.testing.assert_close(
        offloaded(ubq_stack.coords),
        host(ubq_stack.coords),
        rtol=1e-5,
        atol=1e-3,
    )


def test_float64_coordinates_score_entirely_on_host(
    offload, default_database, ubq_stack
):
    # Metal has no float64; such calls must not reach the GPU.
    sfxn = beta2016_score_function(CPU, default_database)
    host, offloaded = _host_and_offloaded(
        lambda: sfxn.render_whole_pose_scoring_module(ubq_stack)
    )
    coords = ubq_stack.coords.to(torch.float64)
    torch.testing.assert_close(offloaded(coords), host(coords))


def test_rotamer_pair_energies_match_host(
    offload, default_database, ubq_pdb, dun_sampler
):
    one = pose_stack_from_pdb(ubq_pdb, CPU, residue_start=0, residue_end=12)
    stack = PoseStackBuilder.from_poses([one, one], CPU)
    task = PackerTask(stack, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(FixedAAChiSampler())
    task = SetPackerTask.from_packer_task(task)
    stack, rotamer_set = build_rotamers(stack, task, default_database.chemical)

    sfxn = beta2016_score_function(CPU, default_database)
    host, offloaded = _host_and_offloaded(
        lambda: sfxn.render_rotamer_scoring_module(stack, rotamer_set)
    )
    assert isinstance(offloaded, OffloadedRotamerScoringModule)
    with torch.no_grad():
        expected = host(rotamer_set.coords).to_dense()
        actual = offloaded(rotamer_set.coords).coalesce().to_dense()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)


def test_offload_can_be_switched_off(default_database, ubq_stack, monkeypatch):
    monkeypatch.setenv("TMOL_MPS_OFFLOAD", "0")
    sfxn = beta2016_score_function(CPU, default_database)
    scorer = sfxn.render_whole_pose_scoring_module(ubq_stack)
    assert not isinstance(scorer, OffloadedWholePoseScoringModule)


def test_host_only_score_functions_are_not_split(offload, default_database, ubq_stack):
    sfxn = beta2016_score_function(CPU, default_database)
    for st in (ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk):
        sfxn.set_weight(st, 0.0)
    sfxn.set_weight(ScoreType.fa_elec, 0.0)
    scorer = sfxn.render_whole_pose_scoring_module(ubq_stack)
    assert not isinstance(scorer, OffloadedWholePoseScoringModule)
