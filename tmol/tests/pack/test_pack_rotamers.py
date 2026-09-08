import attrs
import pytest
import torch
import math
from types import SimpleNamespace

from tmol.pose import (
    ConstraintSet,
    PoseStackBuilder,
)
from tmol.score import (
    ScoreFunction,
    ScoreType,
)
from tmol.pack.compiled import build_interaction_graph
from tmol.pack import (
    PackerTask,
    PackerPalette,
    SetPackerTask,
    PackerEnergyTables,
    run_simulated_annealing,
    impose_top_rotamer_assignments,
    pack_rotamers,
)
from tmol.pack.rotamer import (
    build_rotamers,
    FixedAAChiSampler,
    IncludeCurrentSampler,
    OptHSampler,
)
from tmol.io import pose_stack_from_pdb

from tmol.score.constraint import ConstraintEnergyTerm
from tmol.pack._pack_rotamers import (
    _PACKER_TASK_POSE_TENSORS,
    _interaction_graph_chunk_size,
    _max_poses_per_packing_chunk,
    _slice_packer_task,
    _slice_pose_stack_for_packing,
)


def test_interaction_graph_chunk_size_is_backend_specific(torch_device):
    expected = 32 if torch_device.type == "cuda" else 16
    assert _interaction_graph_chunk_size(torch_device) == expected


def setup_pose_stack_and_task(poses, torch_device, dun_sampler):
    pose_stack = PoseStackBuilder.from_poses(poses, torch_device)
    palette = PackerPalette()
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)
    return pose_stack, task


def build_packer_energy_tables(
    pose_stack, rotamer_set, sfxn, chunk_size=16, raw_entries=False
):
    pbt = pose_stack.packed_block_types
    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)
    if raw_entries:
        energy_indices, energy_values = rotamer_scoring_module.forward_sparse_entries(
            rotamer_set.coords
        )
    else:
        energies = rotamer_scoring_module(rotamer_set.coords).coalesce()
        energy_indices, energy_values = (
            energies.indices().to(torch.int32),
            energies.values(),
        )

    (
        max_n_bump_checked_rotamers_per_pose_tensor,
        n_molten_blocks_per_pose,
        n_bc_rots_per_pose,
        bc_rot_offset_for_pose,
        n_bc_rots_for_molten_block,
        bc_rot_offset_for_molten_block,
        molten_block_ind_for_bc_rot,
        rotamer_for_nonmolten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
        energy1b,
        chunk_pair_offset_for_block_pair,
        chunk_pair_offset,
        energy2b,
    ) = build_interaction_graph(
        True,
        chunk_size,
        pbt.n_types,
        rotamer_set.n_rots_for_pose,
        rotamer_set.rot_offset_for_pose,
        rotamer_set.n_rots_for_block,
        rotamer_set.rot_offset_for_block,
        rotamer_set.pose_for_rot,
        rotamer_set.block_type_ind_for_rot,
        rotamer_set.block_ind_for_rot,
        energy_indices,
        energy_values,
        False,
    )

    # what else??!
    return (
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
        PackerEnergyTables(
            max_n_rotamers_per_pose=max_n_bump_checked_rotamers_per_pose_tensor.item(),
            pose_n_res=n_molten_blocks_per_pose,
            pose_n_rotamers=n_bc_rots_per_pose,
            pose_rotamer_offset=bc_rot_offset_for_pose,
            nrotamers_for_res=n_bc_rots_for_molten_block,
            oneb_offsets=bc_rot_offset_for_molten_block,
            res_for_rot=molten_block_ind_for_bc_rot,
            chunk_size=chunk_size,
            chunk_offset_offsets=chunk_pair_offset_for_block_pair,
            chunk_offsets=chunk_pair_offset,
            energy1b=energy1b,
            energy2b=energy2b,
        ),
    )

    # energy1b, chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b = (
    #     build_interaction_graph(
    #         chunk_size,
    #         rotamer_set.n_rots_for_pose,
    #         rotamer_set.rot_offset_for_pose,
    #         rotamer_set.n_rots_for_block,
    #         rotamer_set.rot_offset_for_block,
    #         rotamer_set.pose_for_rot,
    #         rotamer_set.block_type_ind_for_rot,
    #         rotamer_set.block_ind_for_rot,
    #         energies.indices().to(torch.int32),
    #         energies.values(),
    #     )
    # )
    # return PackerEnergyTables(
    #     max_n_rotamers_per_pose=rotamer_set.max_n_rots_per_pose,
    #     pose_n_res=pose_stack.n_res_per_pose,
    #     pose_n_rotamers=rotamer_set.n_rots_for_pose,
    #     pose_rotamer_offset=rotamer_set.rot_offset_for_pose,
    #     nrotamers_for_res=rotamer_set.n_rots_for_block,
    #     oneb_offsets=rotamer_set.rot_offset_for_block,
    #     res_for_rot=rotamer_set.block_ind_for_rot,
    #     chunk_size=chunk_size,
    #     chunk_offset_offsets=chunk_pair_offset_for_block_pair,
    #     chunk_offsets=chunk_pair_offset,
    #     energy1b=energy1b,
    #     energy2b=energy2b,
    # )


def run_pack_and_assert_scores(
    pose_stack,
    rotamer_set,
    packer_energy_tables,
    sfxn,
    rotamer_for_nonmolten_block,
    n_molten_blocks_per_pose,
    bc_rot_offset_for_molten_block,
    bc_rot_to_orig_rot,
    bg_bg_energies,
):
    scores, rotamer_assignments = run_simulated_annealing(packer_energy_tables)

    # correct for some residues being ignored as part of the background
    scores = scores + bg_bg_energies.unsqueeze(1)

    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack,
        rotamer_set,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        rotamer_assignments,
    )
    wpsm = sfxn.render_whole_pose_scoring_module(new_pose_stack)
    new_scores = wpsm(new_pose_stack.coords)
    torch.testing.assert_close(scores[:, 0], new_scores, atol=1e-3, rtol=1e-5)
    return new_pose_stack, scores


def get_packer_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.fa_elec, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.lk_ball_iso, -0.38)
    sfxn.set_weight(ScoreType.lk_ball, 0.92)
    sfxn.set_weight(ScoreType.lk_bridge, -0.33)
    sfxn.set_weight(ScoreType.lk_bridge_uncpl, -0.33)
    sfxn.set_weight(ScoreType.dunbrack_rot, 0.76)
    sfxn.set_weight(ScoreType.dunbrack_rotdev, 0.69)
    sfxn.set_weight(ScoreType.dunbrack_semirot, 0.78)
    sfxn.set_weight(ScoreType.cart_lengths, 0.5)
    sfxn.set_weight(ScoreType.cart_angles, 0.5)
    sfxn.set_weight(ScoreType.cart_torsions, 0.5)
    sfxn.set_weight(ScoreType.cart_impropers, 0.5)
    sfxn.set_weight(ScoreType.cart_hxltorsions, 0.5)
    sfxn.set_weight(ScoreType.omega, 0.48)
    sfxn.set_weight(ScoreType.rama, 0.50)
    sfxn.set_weight(ScoreType.ref, 1.0)
    sfxn.set_weight(ScoreType.disulfide, 1.0)
    sfxn.set_weight(ScoreType.constraint, 1.0)

    return sfxn


def get_constraints_only_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.constraint, 1.0)

    return sfxn


@pytest.mark.parametrize("chunk_size", [8, 16])
def test_pack_rotamers(
    default_database, ubq_pdb, dun_sampler, torch_device, chunk_size
):
    n_poses = 4
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    pose_stack, task = setup_pose_stack_and_task(
        [p] * n_poses, torch_device, dun_sampler
    )
    task = SetPackerTask.from_packer_task(task)

    sfxn = get_packer_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    (
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
        packer_energy_tables,
    ) = build_packer_energy_tables(pose_stack, rotamer_set, sfxn, chunk_size=chunk_size)

    _, _ = run_pack_and_assert_scores(
        pose_stack,
        rotamer_set,
        packer_energy_tables,
        sfxn,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
    )


def test_raw_rotamer_entries_match_coalesced_interaction_graph(
    default_database, ubq_pdb, dun_sampler, torch_device
):
    if torch_device.type != "cuda":
        pytest.skip("raw duplicate accumulation requires CUDA atomics")

    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=20)
    pose_stack, task = setup_pose_stack_and_task([pose] * 2, torch_device, dun_sampler)
    task = SetPackerTask.from_packer_task(task)
    sfxn = get_packer_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    coalesced = build_packer_energy_tables(pose_stack, rotamer_set, sfxn)
    raw = build_packer_energy_tables(pose_stack, rotamer_set, sfxn, raw_entries=True)

    for coalesced_tensor, raw_tensor in zip(coalesced[:-1], raw[:-1]):
        torch.testing.assert_close(coalesced_tensor, raw_tensor)
    coalesced_tables, raw_tables = coalesced[-1], raw[-1]
    assert (
        coalesced_tables.max_n_rotamers_per_pose == raw_tables.max_n_rotamers_per_pose
    )
    assert coalesced_tables.chunk_size == raw_tables.chunk_size
    for attribute in (
        "pose_n_res",
        "pose_n_rotamers",
        "pose_rotamer_offset",
        "nrotamers_for_res",
        "oneb_offsets",
        "res_for_rot",
        "chunk_offset_offsets",
        "chunk_offsets",
        "energy1b",
        "energy2b",
    ):
        torch.testing.assert_close(
            getattr(coalesced_tables, attribute),
            getattr(raw_tables, attribute),
            atol=1e-3,
            rtol=1e-5,
        )


def test_pack_rotamers_pose_chunks_preserve_pose_order_and_task(
    default_database, ubq_pdb, dun_sampler, torch_device, monkeypatch
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=20)
    pose_stack, task = setup_pose_stack_and_task([pose] * 4, torch_device, dun_sampler)
    sliced_pose_stack = _slice_pose_stack_for_packing(pose_stack, 1, 3)
    assert sliced_pose_stack.n_poses == 2
    assert torch.equal(sliced_pose_stack.coords, pose_stack.coords[1:3])
    assert (
        sliced_pose_stack.coords.untyped_storage().data_ptr()
        == pose_stack.coords.untyped_storage().data_ptr()
    )
    sliced_task = _slice_packer_task(task, 1, 3)
    for attribute in _PACKER_TASK_POSE_TENSORS:
        assert torch.equal(
            getattr(sliced_task, attribute), getattr(task, attribute)[1:3]
        )
    sfxn = get_packer_sfxn(default_database, torch_device)

    torch.manual_seed(11723)
    monkeypatch.setenv("TMOL_PACK_MAX_POSES_PER_CHUNK", "100")
    unchunked = pack_rotamers(pose_stack, sfxn, task)

    torch.manual_seed(11723)
    monkeypatch.setenv("TMOL_PACK_MAX_POSES_PER_CHUNK", "2")
    chunked = pack_rotamers(pose_stack, sfxn, task)

    # Annealing intentionally consumes a different RNG stream when the batch
    # shape changes, so exact rotamer coordinates need not match. Residue
    # identities, pose order, shapes, and validity must be preserved.
    assert torch.equal(unchunked.block_type_ind, chunked.block_type_ind)
    assert unchunked.coords.shape == chunked.coords.shape
    assert torch.isfinite(chunked.coords).all()


@pytest.mark.parametrize("n_blocks,expected", [(256, 25), (257, 10), (1025, 10)])
def test_default_packing_chunk_size_tiers(n_blocks, expected, monkeypatch):
    monkeypatch.delenv("TMOL_PACK_MAX_POSES_PER_CHUNK", raising=False)
    pose_stack = SimpleNamespace(max_n_blocks=n_blocks, device=torch.device("cpu"))
    assert _max_poses_per_packing_chunk(pose_stack) == expected


@pytest.mark.parametrize("configured", ["0", "-1", "not-an-int"])
def test_packing_chunk_size_rejects_invalid_override(configured, monkeypatch):
    monkeypatch.setenv("TMOL_PACK_MAX_POSES_PER_CHUNK", configured)
    pose_stack = SimpleNamespace(max_n_blocks=20, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="must be a positive integer"):
        _max_poses_per_packing_chunk(pose_stack)


def test_shared_rotamer_dispatch_matches_independent_lk_ball_layout(
    default_database, ubq_pdb, dun_sampler, torch_device
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=10)
    pose_stack, task = setup_pose_stack_and_task([pose], torch_device, dun_sampler)
    task = SetPackerTask.from_packer_task(task)
    sfxn = get_packer_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    scorer = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    shared_coords = rotamer_set.coords.detach().clone().requires_grad_(True)
    shared = scorer(shared_coords).coalesce()
    (shared_grad,) = torch.autograd.grad(shared.values().sum(), shared_coords)

    lk_ball = next(term for term in scorer.term_modules if term.classname == "LKBall")
    lk_ball.block_neighbor_cutoff += 0.5
    fallback_coords = rotamer_set.coords.detach().clone().requires_grad_(True)
    fallback = scorer(fallback_coords).coalesce()
    (fallback_grad,) = torch.autograd.grad(fallback.values().sum(), fallback_coords)

    torch.testing.assert_close(shared.to_dense(), fallback.to_dense())
    if torch_device.type == "cuda":
        # Rotamer gradients use atomics, so even two independent evaluations
        # need not be elementwise deterministic. Require the shared-layout
        # error to remain inside the measured independent-repeat envelope.
        repeat_coords = rotamer_set.coords.detach().clone().requires_grad_(True)
        repeat = scorer(repeat_coords).coalesce()
        (repeat_grad,) = torch.autograd.grad(repeat.values().sum(), repeat_coords)
        torch.testing.assert_close(fallback.to_dense(), repeat.to_dense())
        repeat_error = torch.max(torch.abs(fallback_grad - repeat_grad))
        shared_error = torch.max(torch.abs(shared_grad - fallback_grad))
        assert shared_error <= 2 * repeat_error + 1e-6
    else:
        torch.testing.assert_close(shared_grad, fallback_grad)


def test_shared_rotamer_dispatch_matches_independent_hbond_layout(
    default_database, ubq_pdb, dun_sampler, torch_device
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=10)
    pose_stack, task = setup_pose_stack_and_task([pose], torch_device, dun_sampler)
    sfxn = get_packer_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack,
        SetPackerTask.from_packer_task(task),
        pose_stack.packed_block_types.chem_db,
    )
    scorer = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)
    ljlk = next(term for term in scorer.term_modules if term.classname == "LJLK")
    hbond = next(term for term in scorer.term_modules if term.classname == "HBond")

    def dense_hbond(coords, shared_indices=None):
        if shared_indices is None:
            scores, indices = hbond.forward(coords)
        else:
            scores, indices = hbond.forward(coords, shared_indices)
            assert indices.is_set_to(shared_indices)
        sparse = torch.sparse_coo_tensor(
            indices.to(torch.int64),
            scores[0],
            (hbond.n_poses, hbond.n_rots, hbond.n_rots),
            check_invariants=False,
        )
        return sparse.coalesce().to_dense()

    shared_coords = rotamer_set.coords.detach().clone().requires_grad_(True)
    _, shared_indices = ljlk.forward(shared_coords)
    shared = dense_hbond(shared_coords, shared_indices)
    (shared_grad,) = torch.autograd.grad(shared.sum(), shared_coords)

    independent_coords = rotamer_set.coords.detach().clone().requires_grad_(True)
    independent = dense_hbond(independent_coords)
    (independent_grad,) = torch.autograd.grad(independent.sum(), independent_coords)

    torch.testing.assert_close(shared, independent)
    if torch_device.type == "cuda":
        repeat_coords = rotamer_set.coords.detach().clone().requires_grad_(True)
        repeat = dense_hbond(repeat_coords)
        (repeat_grad,) = torch.autograd.grad(repeat.sum(), repeat_coords)
        torch.testing.assert_close(independent, repeat)
        repeat_error = torch.max(torch.abs(independent_grad - repeat_grad))
        shared_error = torch.max(torch.abs(shared_grad - independent_grad))
        assert shared_error <= 2 * repeat_error + 1e-6
    else:
        torch.testing.assert_close(shared_grad, independent_grad)

    if torch_device.type == "cuda":
        dispatch_key = hbond.rotamer_dispatch_key
        with torch.no_grad():
            hbond.rotamer_dispatch_key = None
            independent_indices, *_ = scorer._weighted_entries_by_layout(
                rotamer_set.coords
            )
            hbond.rotamer_dispatch_key = dispatch_key
            shared_indices, *_ = scorer._weighted_entries_by_layout(rotamer_set.coords)
        assert sum(layout.shape[1] for layout in shared_indices) < sum(
            layout.shape[1] for layout in independent_indices
        )


def test_weighted_fused_ljlk_elec_rotamer_scores_match_fallback(
    default_database, ubq_pdb, dun_sampler, torch_device
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=10)
    pose_stack, task = setup_pose_stack_and_task([pose], torch_device, dun_sampler)
    task = SetPackerTask.from_packer_task(task)
    sfxn = get_packer_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    scorer = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)
    assert scorer._fused_ljlk_elec is not None
    if torch_device.type == "cpu":
        scorer._cpu_term_workers = 1

    with torch.no_grad():
        fused = scorer(rotamer_set.coords).coalesce()
        fused_group = scorer._fused_ljlk_elec
        scorer._fused_ljlk_elec = None
        separate = scorer(rotamer_set.coords).coalesce()
        scorer._fused_ljlk_elec = fused_group

    assert torch.equal(fused.indices(), separate.indices())
    torch.testing.assert_close(fused.values(), separate.values(), atol=2e-3, rtol=2e-5)

    # Weights are read at every call instead of being specialized into the
    # rendered module.
    weight_begin = scorer._fused_ljlk_elec_weight_offset
    with torch.no_grad():
        scorer.weights[weight_begin : weight_begin + 4, 0, 0, 0] *= torch.tensor(
            [0.5, 1.25, 0.75, 0.0], device=torch_device
        )
        reweighted_fused = scorer(rotamer_set.coords).coalesce()
        scorer._fused_ljlk_elec = None
        reweighted_separate = scorer(rotamer_set.coords).coalesce()
        scorer._fused_ljlk_elec = fused_group
    assert torch.equal(reweighted_fused.indices(), reweighted_separate.indices())
    torch.testing.assert_close(
        reweighted_fused.values(),
        reweighted_separate.values(),
        atol=2e-3,
        rtol=2e-5,
    )

    # Fusion remains the measured fast path when the term pool has four CPU
    # workers; each native operator can still use ATen's inner CPU parallelism.
    if torch_device.type == "cpu":
        original_forward = fused_group.forward
        fused_calls = []

        def record_fused_call(*args):
            fused_calls.append(None)
            return original_forward(*args)

        fused_group.forward = record_fused_call
        scorer._cpu_term_workers = 4
        with torch.no_grad():
            parallel_scores = scorer(rotamer_set.coords).coalesce()
        assert len(fused_calls) == 1
        assert torch.equal(parallel_scores.indices(), reweighted_fused.indices())
        torch.testing.assert_close(
            parallel_scores.values(), reweighted_fused.values(), atol=2e-3, rtol=2e-5
        )
        scorer._cpu_term_workers = 1
        fused_group.forward = original_forward

    # A caller that changes either dispatch cutoff must use the canonical
    # operators, since the fused traversal assumes the default compatible pair.
    original_cutoff = fused_group.ljlk_module.block_neighbor_cutoff
    fused_group.ljlk_module.block_neighbor_cutoff = original_cutoff + 0.25
    fused_group.forward = lambda *_: pytest.fail(
        "fusion used after changing a live dispatch cutoff"
    )
    with torch.no_grad():
        changed_cutoff = scorer(rotamer_set.coords).coalesce()
    assert changed_cutoff._nnz() != 0
    fused_group.ljlk_module.block_neighbor_cutoff = original_cutoff

    # Differentiable callers must retain the canonical independent operators.
    fused_group.forward = lambda *_: pytest.fail(
        "packing-only fusion used for a differentiable call"
    )
    coords = rotamer_set.coords.detach().clone().requires_grad_(True)
    differentiable = scorer(coords).coalesce()
    differentiable.values().sum().backward()
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()

    scorer.weights.requires_grad_(True)
    weight_differentiable = scorer(rotamer_set.coords.detach()).coalesce()
    weight_differentiable.values().sum().backward()
    assert scorer.weights.grad is not None
    assert torch.isfinite(scorer.weights.grad).all()
    assert torch.count_nonzero(scorer.weights.grad[:4]) != 0


def test_pack_rotamers_optH(default_database, ubq_pdb, torch_device):
    n_poses = 4
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    palette = PackerPalette()
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(OptHSampler())
    task.add_conformer_sampler(FixedAAChiSampler())
    task = SetPackerTask.from_packer_task(task)

    sfxn = get_packer_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )

    # NHQ flip rotamers must have chi either matching the input (~0 deg)
    # or flipped by ~180 deg.
    from tmol.numeric import coord_dihedrals as _cd

    for i in range(task.allowed_bt_block_type.shape[0]):
        pose_i = task.allowed_bt_pose[i].item()
        block_i = task.allowed_bt_block[i].item()
        orig_bt = task.per_block_orig_block_type[pose_i, block_i].item()
        orig = pose_stack.packed_block_types.active_block_types[orig_bt]
        assert hasattr(orig, "opth_sampler_cache")
        cache = orig.opth_sampler_cache
        if cache.nhq_chi_col >= 0:
            a4 = cache.nhq_chi_4atoms
            off = int(pose_stack.block_coord_offset[pose_i, block_i].item())
            c = pose_stack.coords[pose_i][[off + int(a4[k]) for k in range(4)]].double()
            input_chi = float(_cd(c[0:1], c[1:2], c[2:3], c[3:4])[0])
            n_rots = int(rotamer_set.n_rots_for_block[pose_i, block_i].item())
            rot_off = int(rotamer_set.rot_offset_for_block[pose_i, block_i].item())
            for r in range(n_rots):
                co = int(rotamer_set.coord_offset_for_rot[rot_off + r].item())
                rc4 = rotamer_set.coords[[co + int(a4[k]) for k in range(4)]].double()
                rot_chi = float(_cd(rc4[0:1], rc4[1:2], rc4[2:3], rc4[3:4])[0])
                delta = math.degrees(rot_chi - input_chi)
                delta = (delta + 180.0) % 360.0 - 180.0
                # assert deltas are only 0 or 180
                assert min(abs(delta), abs(abs(delta) - 180.0)) < 1.0, (
                    f"res {block_i} ({orig.name3}) rot {r}: "
                    f"unexpected NHQ chi delta {delta:.2f} deg"
                )
        else:
            n_rots = int(rotamer_set.n_rots_for_block[pose_i, block_i].item())
            assert cache.n_proton_samples == 0 or n_rots == cache.n_proton_samples + 1

    (
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
        packer_energy_tables,
    ) = build_packer_energy_tables(pose_stack, rotamer_set, sfxn)
    # bg_bg_energies, packer_energy_tables = build_packer_energy_tables(
    #     pose_stack, rotamer_set, sfxn
    # )
    # _, _ = run_pack_and_assert_scores(
    #     pose_stack, rotamer_set, packer_energy_tables, sfxn, bg_bg_energies
    # )
    _, _ = run_pack_and_assert_scores(
        pose_stack,
        rotamer_set,
        packer_energy_tables,
        sfxn,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
    )


def test_pack_rotamers_w_cst(
    default_database, ubq_pdb, dun_sampler, torch_device, monkeypatch
):
    n_poses = 4
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    pose_stack, task = setup_pose_stack_and_task(
        [p] * n_poses, torch_device, dun_sampler
    )
    task = SetPackerTask.from_packer_task(task)

    sfxn = get_constraints_only_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    constraints = ConstraintSet.create_empty(device=torch_device, n_poses=n_poses)

    # a distance constraint
    cnstr_atoms = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 3)
    res2_type = pose_stack.block_type(0, 4)
    cnstr_atoms[0, 0] = torch.tensor([0, 3, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 4, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47
    cnstr_params[0, 1] = 0.1

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    # a distance constraint
    cnstr_atoms = torch.full((1, 2, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 2), 0, dtype=torch.float32, device=torch_device)

    res1_type = pose_stack.block_type(0, 5)
    res2_type = pose_stack.block_type(0, 6)
    cnstr_atoms[0, 0] = torch.tensor([0, 5, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 6, res2_type.atom_to_idx["N"]])
    cnstr_params[0, 0] = 1.47
    cnstr_params[0, 1] = 0.1

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic, cnstr_atoms, cnstr_params
    )

    # a circular harmonic constraint
    cnstr_atoms = torch.full((1, 4, 3), 0, dtype=torch.int32, device=torch_device)
    cnstr_params = torch.full((1, 3), 0, dtype=torch.float32, device=torch_device)

    # get the omega between res1 and res2
    res1_type = pose_stack.block_type(0, 0)
    res2_type = pose_stack.block_type(0, 1)
    cnstr_atoms[0, 0] = torch.tensor([0, 0, res1_type.atom_to_idx["CA"]])
    cnstr_atoms[0, 1] = torch.tensor([0, 0, res1_type.atom_to_idx["C"]])
    cnstr_atoms[0, 2] = torch.tensor([0, 1, res2_type.atom_to_idx["N"]])
    cnstr_atoms[0, 3] = torch.tensor([0, 1, res2_type.atom_to_idx["CA"]])
    cnstr_params[0, 0] = math.pi
    cnstr_params[0, 1] = 0.1
    cnstr_params[0, 2] = 0.0

    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.circularharmonic, cnstr_atoms, cnstr_params
    )

    pose_stack = attrs.evolve(pose_stack, constraint_set=constraints)
    # Exercise constraint slicing and remapping: constraint pose indices must
    # be remapped independently for every packing chunk.
    monkeypatch.setenv("TMOL_PACK_MAX_POSES_PER_CHUNK", "2")
    packed = pack_rotamers(pose_stack, sfxn, task)
    assert packed.n_poses == n_poses
    assert packed.constraint_set is not None
    assert (
        packed.constraint_set.constraint_functions == constraints.constraint_functions
    )
    for attribute in (
        "constraint_function_inds",
        "constraint_atoms",
        "constraint_params",
        "constraint_num_unique_blocks",
        "constraint_unique_blocks",
    ):
        torch.testing.assert_close(
            getattr(packed.constraint_set, attribute), getattr(constraints, attribute)
        )
    assert torch.isfinite(packed.coords).all()
    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()


def test_pack_rotamers_w_empty_interaction_graph(
    default_database, disulfide_pdb, dun_sampler, torch_device
):
    n_poses = 4
    p = pose_stack_from_pdb(disulfide_pdb, torch_device)
    pose_stack, task = setup_pose_stack_and_task(
        [p] * n_poses, torch_device, dun_sampler
    )
    task = SetPackerTask.from_packer_task(task)
    sfxn = get_constraints_only_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    (
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
        packer_energy_tables,
    ) = build_packer_energy_tables(pose_stack, rotamer_set, sfxn)
    # bg_bg_energies, packer_energy_tables = build_packer_energy_tables(
    #     pose_stack, rotamer_set, sfxn
    # )
    _, _ = run_pack_and_assert_scores(
        pose_stack,
        rotamer_set,
        packer_energy_tables,
        sfxn,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
    )
    # run_pack_and_assert_scores(
    #     pose_stack, rotamer_set, packer_energy_tables, sfxn, bg_bg_energies
    # )


def test_pack_rotamers_w_dslf(
    default_database, disulfide_pdb, dun_sampler, torch_device
):
    n_poses = 4
    p = pose_stack_from_pdb(disulfide_pdb, torch_device)
    pose_stack, task = setup_pose_stack_and_task(
        [p] * n_poses, torch_device, dun_sampler
    )
    task = SetPackerTask.from_packer_task(task)
    sfxn = get_packer_sfxn(default_database, torch_device)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    (
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
        packer_energy_tables,
    ) = build_packer_energy_tables(pose_stack, rotamer_set, sfxn)
    # bg_bg_energies, packer_energy_tables = build_packer_energy_tables(
    #     pose_stack, rotamer_set, sfxn
    # )
    # run_pack_and_assert_scores(
    #     pose_stack, rotamer_set, packer_energy_tables, sfxn, bg_bg_energies
    # )
    _, _ = run_pack_and_assert_scores(
        pose_stack,
        rotamer_set,
        packer_energy_tables,
        sfxn,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
    )


def test_pack_rotamers2(default_database, ubq_pdb, dun_sampler, torch_device):
    if torch_device == torch.device("cpu"):
        return
    n_poses = 10
    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)
    pose_stack, task = setup_pose_stack_and_task(
        [p] * n_poses, torch_device, dun_sampler
    )
    task.or_expand_chi(1)
    sfxn = get_packer_sfxn(default_database, torch_device)
    pack_rotamers(pose_stack, sfxn, task)


def test_pack_rotamers_irregular_sized_poses(
    default_database, ubq_pdb, dun_sampler, torch_device
):
    if torch_device == torch.device("cpu"):
        return
    n_poses = 20
    poses = [
        pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=20 + i)
        for i in range(n_poses)
    ]
    pose_stack, task = setup_pose_stack_and_task(poses, torch_device, dun_sampler)
    task.or_expand_chi(1)
    sfxn = get_packer_sfxn(default_database, torch_device)
    pack_rotamers(pose_stack, sfxn, task)
