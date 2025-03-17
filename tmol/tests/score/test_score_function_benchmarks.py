import pytest
import torch
from tmol.tests.torch import zero_padded_counts

from tmol.score.score_function import ScoreFunction

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io import pose_stack_from_pdb

from tmol.score.cartbonded.cartbonded_energy_term import CartBondedEnergyTerm
from tmol.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
from tmol.score.dunbrack.dunbrack_energy_term import DunbrackEnergyTerm
from tmol.score.elec.elec_energy_term import ElecEnergyTerm
from tmol.score.hbond.hbond_energy_term import HBondEnergyTerm
from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm
from tmol.score.lk_ball.lk_ball_energy_term import LKBallEnergyTerm
from tmol.score.backbone_torsion.bb_torsion_energy_term import BackboneTorsionEnergyTerm
from tmol.score.ref.ref_energy_term import RefEnergyTerm
from tmol.score import beta2016_score_function

from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form


@pytest.mark.parametrize("energy_term", [LJLKEnergyTerm], ids=["ljlk"])
@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30, 100]))
@pytest.mark.benchmark(group="setup_res_centric_scoring")
def dont_test_res_centric_score_benchmark_setup(
    benchmark, energy_term, n_poses, ubq_pdb, default_database, torch_device
):
    n_poses = int(n_poses)
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device)

    pose_stack_n = PoseStackBuilder.from_poses([pose_stack1] * n_poses, torch_device)
    sfxn = ScoreFunction(default_database, torch_device)

    for st in energy_term.score_types():
        sfxn.set_weight(st, 1.0)

    @benchmark
    def render_whole_pose_scoring_module():
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack_n)
        return scorer

    render_whole_pose_scoring_module


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30, 100]))
@pytest.mark.parametrize("benchmark_pass", ["forward", "full", "backward"])
@pytest.mark.parametrize(
    "energy_term",
    [
        CartBondedEnergyTerm,
        DisulfideEnergyTerm,
        DunbrackEnergyTerm,
        ElecEnergyTerm,
        HBondEnergyTerm,
        LJLKEnergyTerm,
        LKBallEnergyTerm,
        BackboneTorsionEnergyTerm,
        RefEnergyTerm,
    ],
    ids=[
        "cartbonded",
        "disulfide",
        "dunbrack",
        "elec",
        "hbond",
        "ljlk",
        "lk_ball",
        "backbone_torsion",
        "ref",
    ],
)
@pytest.mark.benchmark(group="res_centric_score_components")
def test_res_centric_score_benchmark(
    benchmark,
    benchmark_pass,
    energy_term,
    n_poses,
    ubq_pdb,
    default_database,
    torch_device,
):
    n_poses = int(n_poses)
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pose_stack_n = PoseStackBuilder.from_poses([pose_stack1] * n_poses, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)

    for st in energy_term.score_types():
        sfxn.set_weight(st, 1.0)

    scorer = sfxn.render_whole_pose_scoring_module(pose_stack_n)

    if benchmark_pass == "full":
        pose_stack_n.coords.requires_grad_(True)

        @benchmark
        def score_pass():
            scores = torch.sum(scorer(pose_stack_n.coords))
            scores.backward(retain_graph=True)
            return scores.cpu()

    elif benchmark_pass == "forward":

        @benchmark
        def score_pass():
            scores = torch.sum(scorer(pose_stack_n.coords))
            scores.cpu()
            return scores

    elif benchmark_pass == "backward":
        pose_stack_n.coords.requires_grad_(True)
        scores = torch.sum(scorer(pose_stack_n.coords))

        @benchmark
        def score_pass():
            scores.backward(retain_graph=True)
            return scores.cpu()

    else:
        raise NotImplementedError


@pytest.mark.parametrize("n_poses", zero_padded_counts([1, 3, 10, 30, 100]))
@pytest.mark.parametrize("benchmark_pass", ["forward", "full", "backward"])
@pytest.mark.parametrize(
    "energy_terms",
    [
        [
            CartBondedEnergyTerm,
            DisulfideEnergyTerm,
            DunbrackEnergyTerm,
            ElecEnergyTerm,
            HBondEnergyTerm,
            LJLKEnergyTerm,
            LKBallEnergyTerm,
            BackboneTorsionEnergyTerm,
            RefEnergyTerm,
        ]
    ],
    ids=["cartbonded_disulfide_dunbrack_elec_hbond_ljlk_lkb_bbtorsion_ref"],
)
@pytest.mark.benchmark(group="res_centric_combined_score_components")
def test_combined_res_centric_score_benchmark(
    benchmark,
    benchmark_pass,
    energy_terms,
    n_poses,
    ubq_pdb,
    default_database,
    torch_device,
):
    n_poses = int(n_poses)
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pose_stack_n = PoseStackBuilder.from_poses([pose_stack1] * n_poses, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)

    for energy_term in energy_terms:
        for st in energy_term.score_types():
            sfxn.set_weight(st, 1.0)

    scorer = sfxn.render_whole_pose_scoring_module(pose_stack_n)

    if benchmark_pass == "full":
        pose_stack_n.coords.requires_grad_(True)

        @benchmark
        def score_pass():
            scores = torch.sum(scorer(pose_stack_n.coords))
            scores.backward(retain_graph=True)
            return scores.cpu()

    elif benchmark_pass == "forward":

        @benchmark
        def score_pass():
            scores = torch.sum(scorer(pose_stack_n.coords))
            scores.cpu()
            return scores

    elif benchmark_pass == "backward":
        pose_stack_n.coords.requires_grad_(True)
        scores = torch.sum(scorer(pose_stack_n.coords))

        @benchmark
        def score_pass():
            scores.backward(retain_graph=True)
            return scores.cpu()

    else:
        raise NotImplementedError


@pytest.mark.benchmark(group="res_centric_build_posestack")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300, 600])
def test_build_posestack(
    benchmark, systems_bysize, system_size, default_database, torch_device
):
    @benchmark
    def setup():
        co = default_canonical_ordering()
        pbt = default_packed_block_types(torch_device)
        canonical_form = canonical_form_from_pdb(
            co, systems_bysize[system_size], torch_device
        )
        _ = pose_stack_from_canonical_form(co, pbt, **canonical_form)

    setup


@pytest.mark.benchmark(group="res_centric_render_module")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300, 600])
def test_render_module(
    benchmark, systems_bysize, system_size, default_database, torch_device
):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, systems_bysize[system_size], torch_device
    )
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

    @benchmark
    def setup():
        sfxn = beta2016_score_function(torch_device)
        _ = sfxn.render_whole_pose_scoring_module(pose_stack)

    setup


@pytest.mark.benchmark(group="total_score_onepass")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300, 600])
def test_full(benchmark, systems_bysize, system_size, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, systems_bysize[system_size], torch_device
    )
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)
    pose_stack.coords.requires_grad_(True)

    sfxn = beta2016_score_function(torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

    @benchmark
    def forward_backward():
        scores = torch.sum(scorer(pose_stack.coords))
        scores.backward(retain_graph=True)
        return scores.cpu()

    forward_backward
