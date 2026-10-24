"""M13a-2 gates for flexible-sidechain grid search and restoration."""

from __future__ import annotations

import time

import pytest
import torch

from tmol.dock import (
    GASchedule,
    GridBackend,
    LigandGeneLayout,
    build_ligand_docking_dofs,
    run_docking,
    run_ga_docking,
    select_pocket_residue_blocks,
)
from tmol.dock.ga import (
)
from tmol.dock.ligand_dofs import (
    _block_kinforest_nodes,
)
from tmol.dock.scoring_backend import _LightweightGridObjective
from tmol.score import beta2016_score_function
from tmol.score.gridscore import (
    build_live_ligand_intra_data,
    score_ligand_grid_batched,
    score_ligand_hbond,
    score_ligand_intra,
    score_ligand_intra_with_gradients,
)
from tmol.score.gridscore.live_ligand_intra import (
    _weighted_movable_block_pairs,
)
from tmol.score import ScoreType
from tmol.tests.dock.test_dock_loop import _load_cdk2_pose_on_device
from tmol.tests.dock.test_flexible_sidechain_dofs import (
    _connected_chi_blocks,
)
from tmol.tests.dock.test_grid_backend import cpu_grid_case
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
    cdk2_pose_and_db,
)
from tmol.score.gridscore._movable import (
    movable_block_mapping,
    movable_pose_stack,
    sidechain_context_blocks,
)


def _cdk2_sidechain_blocks(reference, ligand_block):
    selected = select_pocket_residue_blocks(reference, ligand_block, cutoff=6.0)
    return _connected_chi_blocks(reference, ligand_block, selected)[:2]


def _movable_layout(reference, ligand_block, sidechain_blocks):
    movable = movable_pose_stack(reference, ligand_block, sidechain_blocks)
    _, compact_sidechains, _ = movable_block_mapping(
        reference, ligand_block, sidechain_blocks
    )
    docking_dofs = build_ligand_docking_dofs(
        movable, 0, sidechain_blocks=compact_sidechains
    )
    return (
        movable,
        docking_dofs,
        LigandGeneLayout.from_pose_stack(movable, docking_dofs),
    )


def _exact_movable_oracle(movable, param_db, context_blocks=()):
    sfxn = beta2016_score_function(torch.device("cpu"), param_db=param_db)
    scorer = sfxn.render_block_pair_scoring_module(movable)
    pairs = scorer(movable.coords)
    if context_blocks:
        context = torch.tensor(context_blocks, dtype=torch.int64)
        pairs[:, context, context] = 0
    return pairs.sum(dim=(1, 2))


def test_movable_whole_pose_score_matches_exact_beta2016_fp64(
    cdk2_pose_and_db,
):
    reference, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(reference)
    sidechain_blocks = _cdk2_sidechain_blocks(reference, ligand_block)
    movable, docking_dofs, layout = _movable_layout(
        reference, ligand_block, sidechain_blocks
    )
    genes = layout.genes_from_pose_stack(movable)
    genes[:, 0] += 0.08
    genes[:, 6] += 0.11
    genes[:, 6 + len(docking_dofs.torsions)] -= 0.17
    moved = layout.pose_stack_from_genes(genes)

    data = build_live_ligand_intra_data(
        reference,
        ligand_block,
        param_db,
        weight_profile="full",
        sidechain_blocks=sidechain_blocks,
    )
    actual = score_ligand_intra(data, moved.coords)
    expected = _exact_movable_oracle(moved, param_db, data.context_block_indices)
    error = torch.max(torch.abs(actual - expected))
    print("movable full-beta score max error:", float(error))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-4)


@pytest.mark.slow
def test_movable_block_pairs_match_full_connected_complex_fp64(
    cdk2_pose_and_db,
):
    reference, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(reference)
    sidechain_blocks = _cdk2_sidechain_blocks(reference, ligand_block)
    movable, _, _ = _movable_layout(reference, ligand_block, sidechain_blocks)
    source_blocks, compact_sidechains, compact_context = movable_block_mapping(
        reference, ligand_block, sidechain_blocks
    )

    sfxn = beta2016_score_function(torch.device("cpu"), param_db=param_db)
    compact_pairs = sfxn.render_block_pair_scoring_module(movable)(
        movable.coords, sum_terms=False
    )
    full_pairs = sfxn.render_block_pair_scoring_module(reference)(
        reference.coords, sum_terms=False
    )
    source_index = torch.tensor(source_blocks, dtype=torch.int64)
    expected_pairs = full_pairs.index_select(2, source_index).index_select(
        3, source_index
    )

    # The block-pair module stores each unordered pair's energy in exactly one
    # triangle, keyed by (min block index, max block index); the transpose slot
    # is zero. Compacting moves the ligand from the highest source block index to
    # block 0, which flips which triangle holds every ligand<->receptor pair, so a
    # positional comparison must first fold both tensors to a triangle-agnostic
    # form. The total (sum over both block dims) is unaffected, which is why the
    # whole-pose score already matches exactly.
    compact_sym = compact_pairs + compact_pairs.transpose(2, 3)
    expected_sym = expected_pairs + expected_pairs.transpose(2, 3)

    # Immediate context is present to make selected-residue one-body terms and
    # every retained pair exact. A context residue's own diagonal may depend on
    # its next omitted peptide neighbor and is a fixed search constant.
    compare_mask = torch.ones(
        (len(source_blocks), len(source_blocks)), dtype=torch.bool
    )
    for compact_block in compact_context:
        compare_mask[compact_block, compact_block] = False

    # Backbone-dihedral energies (rama/omega/p_aa_pp) are functions of phi/psi/omega
    # only. Flexible docking moves the ligand rigid body, ligand torsions, and
    # sidechain chi; the protein BACKBONE is fixed, so these terms are constant
    # across every pose and are outside the movable module's optimized score set
    # (which is fa_ljatr/ljrep/lk/elec + gen_torsions + cart_* + live hbond/lk_ball,
    # plus chi-dependent fa_dun). At the truncated context boundary (a context
    # residue whose far peptide neighbor is omitted) they take a fixed value that
    # differs from the full complex by a per-pose constant, which cannot bias search
    # or dG ranking. Restrict the strict per-pair equality to the score types the
    # flexible search actually optimizes over.
    score_types = tuple(sfxn.all_score_types())
    backbone_constant = {
        name for name in ("rama", "omega", "p_aa_pp") if hasattr(ScoreType, name)
    }
    term_keep = torch.tensor(
        [st.name not in backbone_constant for st in score_types],
        dtype=torch.bool,
    )
    actual_selected = compact_sym[term_keep][:, :, compare_mask]
    expected_selected = expected_sym[term_keep][:, :, compare_mask]
    error = torch.max(torch.abs(actual_selected - expected_selected))
    print("connected compact block-pair max error:", float(error))
    torch.testing.assert_close(
        actual_selected, expected_selected, rtol=0.0, atol=1.0e-4
    )

    # Explicit backbone-dependent Dunbrack (HIGH-2): the total rotamer energy
    # attributed to each flexible residue must land in the correct phi/psi bin,
    # which the included peptide context provides. Sum the three Dunbrack score
    # types over each selected block's full symmetrized row so the check is robust
    # to whether the term is stored on the diagonal or the (i, i-1) edge.
    dun_types = (
        ScoreType.dunbrack_rot,
        ScoreType.dunbrack_rotdev,
        ScoreType.dunbrack_semirot,
    )
    dun_index = torch.tensor(
        [score_types.index(st) for st in dun_types], dtype=torch.int64
    )
    selected_compact = torch.tensor(compact_sidechains, dtype=torch.int64)
    compact_dun = compact_sym[dun_index][:, :, selected_compact, :].sum(dim=(0, -1))
    expected_dun = expected_sym[dun_index][:, :, selected_compact, :].sum(dim=(0, -1))
    torch.testing.assert_close(compact_dun, expected_dun, rtol=0.0, atol=1.0e-4)


def test_search_repulsion_is_soft_only_on_block_diagonal(cdk2_pose_and_db):
    reference, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(reference)
    sidechain_blocks = _cdk2_sidechain_blocks(reference, ligand_block)
    movable, _, _ = _movable_layout(reference, ligand_block, sidechain_blocks)
    data = build_live_ligand_intra_data(
        reference,
        ligand_block,
        param_db,
        weight_profile="search",
        sidechain_blocks=sidechain_blocks,
    )
    weighted = _weighted_movable_block_pairs(data, movable.coords)
    scorer = data.score_function.render_block_pair_scoring_module(movable)
    unweighted = scorer(movable.coords, sum_terms=False, apply_weights=False)
    score_types = tuple(data.score_function.all_score_types())
    rep_index = score_types.index(ScoreType.fa_ljrep)
    full_rep = data.score_function.get_weight(ScoreType.fa_ljrep).to(
        dtype=movable.coords.dtype
    )
    n_blocks = movable.max_n_blocks
    eye = torch.eye(n_blocks, dtype=torch.bool)
    active_diagonal = eye.clone()
    context = torch.tensor(data.context_block_indices, dtype=torch.int64)
    active_diagonal[context, context] = False
    torch.testing.assert_close(
        weighted[rep_index, :, active_diagonal],
        0.01 * unweighted[rep_index, :, active_diagonal],
        rtol=0.0,
        atol=1.0e-10,
    )
    assert torch.count_nonzero(weighted[rep_index, :, eye & ~active_diagonal]) == 0
    torch.testing.assert_close(
        weighted[rep_index, :, ~eye],
        full_rep * unweighted[rep_index, :, ~eye],
        rtol=0.0,
        atol=1.0e-10,
    )


def test_movable_whole_pose_gradient_matches_finite_difference_fp64(
    cdk2_pose_and_db,
):
    reference, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(reference)
    sidechain_blocks = _cdk2_sidechain_blocks(reference, ligand_block)[:1]
    movable, docking_dofs, layout = _movable_layout(
        reference, ligand_block, sidechain_blocks
    )
    genes = layout.genes_from_pose_stack(movable)
    genes[:, 6 + len(docking_dofs.torsions)] += 0.13
    coords = layout.pose_stack_from_genes(genes).coords.contiguous()

    data = build_live_ligand_intra_data(
        reference,
        ligand_block,
        param_db,
        weight_profile="full",
        sidechain_blocks=sidechain_blocks,
    )
    _, gradient = score_ligand_intra_with_gradients(data, coords)
    columns = torch.arange(coords.numel(), dtype=coords.dtype).reshape_as(coords)
    direction = torch.sin(0.37 * columns + 0.19)
    direction = direction / torch.linalg.vector_norm(direction)
    step = 1.0e-6
    finite = (
        score_ligand_intra(data, coords + step * direction)
        - score_ligand_intra(data, coords - step * direction)
    ) / (2.0 * step)
    analytic = torch.sum(gradient * direction, dim=(1, 2))
    error = torch.max(torch.abs(analytic - finite))
    print("movable full-beta directional-gradient max error:", float(error))
    torch.testing.assert_close(analytic, finite, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.slow
def test_flexible_grid_atom_partition_and_score_decomposition_fp64(
    cdk2_pose_and_db,
):
    reference, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(reference)
    sidechain_blocks = _cdk2_sidechain_blocks(reference, ligand_block)
    backend = GridBackend.from_pose_stack(
        reference,
        ligand_block,
        param_db,
        kin_dtype=torch.float64,
        sidechain_blocks=sidechain_blocks,
    )
    movable, docking_dofs, layout = _movable_layout(
        reference, ligand_block, sidechain_blocks
    )
    ligand_n_atoms = movable.block_type(0, 0).n_atoms
    expected_grid_atoms = torch.cat(
        (
            torch.arange(ligand_n_atoms, dtype=torch.int64),
            layout.sidechain_atom_indices[0],
        )
    )
    assert torch.equal(backend.grid_atom_indices, expected_grid_atoms)
    assert backend.live_hbond.n_ligand_atoms == movable.max_n_pose_atoms
    assert backend.live_lk_ball.n_ligand_atoms == movable.max_n_pose_atoms
    assert backend.live_ligand_intra.n_ligand_atoms == movable.max_n_pose_atoms

    expected_receptor_heavy = []
    pbt = reference.packed_block_types
    context_blocks = sidechain_context_blocks(
        reference, ligand_block, sidechain_blocks
    )
    assert backend.context_blocks == context_blocks
    excluded_blocks = {
        ligand_block,
        *sidechain_blocks,
        *context_blocks,
    }
    for block in range(reference.max_n_blocks):
        block_type_index = int(reference.block_type_ind64[0, block])
        if block_type_index < 0 or block in excluded_blocks:
            continue
        block_type = pbt.active_block_types[block_type_index]
        offset = int(reference.block_coord_offset64[0, block])
        for atom in range(block_type.n_atoms):
            if int(pbt.hbpbt_params.is_hydrogen[block_type_index, atom]) == 0:
                expected_receptor_heavy.append(reference.coords[0, offset + atom])
    expected_receptor_heavy = torch.stack(expected_receptor_heavy)
    torch.testing.assert_close(
        backend.live_lk_ball.receptor_coords,
        expected_receptor_heavy,
        rtol=0.0,
        atol=0.0,
    )
    assert backend.live_ligand_intra.ligand_pose_stack.max_n_blocks == (
        1 + len(sidechain_blocks) + len(context_blocks)
    )
    full = backend.for_ligand_block(0)._score_ligand_coordinates(movable.coords)
    common = dict(
        grids=backend._grid_for(torch.device("cpu"), torch.float64),
        ligand_coords=movable.coords,
        ligand_atom_types=backend.ligand_atom_types,
        ligand_charges=backend._charges_for(torch.device("cpu"), torch.float64),
        score_function=backend._score_function(torch.device("cpu")),
        grid_atom_indices=backend._grid_atom_indices_for(torch.device("cpu")),
    )
    grid_only = score_ligand_grid_batched(**common)
    with_hbond = score_ligand_grid_batched(
        **common,
        live_hbond=backend._hbond_for(torch.device("cpu"), torch.float64),
    )
    with_lk_ball = score_ligand_grid_batched(
        **common,
        live_lk_ball=backend._lk_ball_for(torch.device("cpu"), torch.float64),
        lk_ball_grid=backend.lk_ball_grid,
    )
    hbond = with_hbond - grid_only
    lk_ball = with_lk_ball - grid_only
    direct_hbond = score_ligand_hbond(
        backend._hbond_for(torch.device("cpu"), torch.float64), movable.coords
    )
    hbond_weight = backend._score_function(torch.device("cpu")).get_weight(
        ScoreType.hbond
    )
    torch.testing.assert_close(
        hbond,
        hbond_weight * direct_hbond,
        rtol=0.0,
        atol=1.0e-10,
    )
    interface = score_ligand_grid_batched(
        backend._grid_for(torch.device("cpu"), torch.float64),
        movable.coords,
        backend.ligand_atom_types,
        backend._charges_for(torch.device("cpu"), torch.float64),
        score_function=backend._score_function(torch.device("cpu")),
        live_hbond=backend._hbond_for(torch.device("cpu"), torch.float64),
        live_lk_ball=backend._lk_ball_for(torch.device("cpu"), torch.float64),
        lk_ball_grid=backend.lk_ball_grid,
        grid_atom_indices=backend._grid_atom_indices_for(torch.device("cpu")),
    )
    intra = score_ligand_intra(
        backend._intra_for(torch.device("cpu"), torch.float64), movable.coords
    )
    torch.testing.assert_close(
        interface, grid_only + hbond + lk_ball, rtol=0.0, atol=1.0e-10
    )
    torch.testing.assert_close(
        full, grid_only + hbond + lk_ball + intra, rtol=0.0, atol=1.0e-10
    )

    objective = _LightweightGridObjective(
        backend.for_ligand_block(0),
        movable,
        docking_dofs,
        torch.float64,
        None,
    )
    masked = objective.initial_masked_dofs
    _, dof_gradient = objective.energy_and_gradient(masked)
    mapping = docking_dofs.sidechain_torsions[0]
    block_nodes = _block_kinforest_nodes(
        movable,
        mapping.block,
        movable.block_type(0, mapping.block).n_atoms,
        objective.forest.id,
    )
    raw_node = block_nodes[0, mapping.raw_atom]
    flat_key = raw_node * 9 + int(mapping.dof_type)
    sidechain_column = torch.nonzero(
        objective.dof_flat_idx == flat_key, as_tuple=False
    ).flatten()
    assert sidechain_column.numel() == 1
    sidechain_column = int(sidechain_column[0])
    step = 1.0e-6
    plus = masked.clone()
    minus = masked.clone()
    plus[sidechain_column] += step
    minus[sidechain_column] -= step
    finite = (objective.energy(plus)[0] - objective.energy(minus)[0]) / (2.0 * step)
    error = torch.abs(dof_gradient[sidechain_column] - finite)
    print(
        "flexible grid sidechain-DOF analytic/finite/error:",
        float(dof_gradient[sidechain_column]),
        float(finite),
        float(error),
    )
    assert torch.abs(dof_gradient[sidechain_column]) > 1.0e-8
    torch.testing.assert_close(
        dof_gradient[sidechain_column], finite, rtol=2.0e-4, atol=2.0e-4
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_flexible_grid_search_moves_sidechain_and_restores_full_complex(
    cdk2_pose_and_db,
):
    cpu_reference, param_db = cdk2_pose_and_db
    ligand_block = _ligand_block(cpu_reference)
    sidechain_blocks = _cdk2_sidechain_blocks(cpu_reference, ligand_block)
    backend = GridBackend.from_pose_stack(
        cpu_reference,
        ligand_block,
        param_db,
        kin_dtype=torch.float32,
        sidechain_blocks=sidechain_blocks,
    )
    cpu_movable, _, _ = _movable_layout(cpu_reference, ligand_block, sidechain_blocks)
    cpu_energy, cpu_gradient = backend.for_ligand_block(
        0
    )._score_ligand_coordinates_with_gradients(cpu_movable.coords)
    cuda_movable_coords = cpu_movable.coords.to(device="cuda")
    cuda_energy, cuda_gradient = backend.for_ligand_block(
        0
    )._score_ligand_coordinates_with_gradients(cuda_movable_coords)
    energy_error = torch.max(torch.abs(cuda_energy.cpu() - cpu_energy))
    gradient_error = torch.max(torch.abs(cuda_gradient.cpu() - cpu_gradient))
    print(
        "flexible fp64 CPU/CUDA energy and gradient max errors:",
        float(energy_error),
        float(gradient_error),
    )
    torch.testing.assert_close(cuda_energy.cpu(), cpu_energy, rtol=1.0e-10, atol=1.0e-8)
    # CUDA vs CPU fp64 gradients differ at the reduction floor (~1e-8); the
    # gradient accumulates more reduction error than the energy, so use a
    # slightly looser bound than the energy parity above.
    torch.testing.assert_close(
        cuda_gradient.cpu(), cpu_gradient, rtol=1.0e-6, atol=2.0e-7
    )

    reference, _ = _load_cdk2_pose_on_device(torch.device("cuda"))
    movable, _, layout = _movable_layout(reference, ligand_block, sidechain_blocks)
    native_score = backend.for_ligand_block(0).score(movable)[0]
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = run_docking(
        reference,
        ligand_block,
        backend,
        schedule=GASchedule(
            generations=1,
            npool=8,
            rmsdthreshold=0.0,
            pmut=0.2,
            maxiter=10,
        ),
        seed=20260825,
        sidechain_blocks=sidechain_blocks,
    )
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    assert torch.isfinite(result.best_energies).all()
    assert result.best_energies[0] <= native_score + 1.0e-5

    moved_count = 0
    bad_bonds = 0
    checked_blocks = (ligand_block,) + sidechain_blocks
    for source_block in checked_blocks:
        block_type = reference.block_type(0, source_block)
        bond_indices = torch.tensor(
            block_type.bond_indices[::2],
            dtype=torch.int64,
            device=reference.device,
        )
        source_offset = reference.block_coord_offset64[0, source_block]
        before = torch.linalg.vector_norm(
            reference.coords[0, source_offset + bond_indices[:, 0]]
            - reference.coords[0, source_offset + bond_indices[:, 1]],
            dim=1,
        )
        after = torch.linalg.vector_norm(
            result.best_pose_stack.coords[0, source_offset + bond_indices[:, 0]]
            - result.best_pose_stack.coords[0, source_offset + bond_indices[:, 1]],
            dim=1,
        )
        bad_bonds += int(torch.sum(torch.abs(after - before) > 2.0e-4))

    _, compact_sidechains, _ = movable_block_mapping(
        reference, ligand_block, sidechain_blocks
    )
    for compact_block, source_block in zip(compact_sidechains, sidechain_blocks):
        compact_start = int(movable.block_coord_offset64[0, compact_block])
        block_type = reference.block_type(0, source_block)
        compact_atoms = layout.sidechain_atom_indices[0]
        local_atoms = (
            compact_atoms[
                (compact_atoms >= compact_start)
                & (compact_atoms < compact_start + block_type.n_atoms)
            ]
            - compact_start
        )
        source_atoms = reference.block_coord_offset64[0, source_block] + local_atoms
        displacement = torch.linalg.vector_norm(
            result.best_pose_stack.coords[0, source_atoms]
            - reference.coords[0, source_atoms],
            dim=1,
        )
        moved_count += int(torch.sum(displacement > 1.0e-5))

        backbone_local = torch.tensor(
            [
                atom_index
                for atom_index, atom in enumerate(block_type.atoms)
                if atom.name in {"N", "CA", "C", "O", "OXT"}
            ],
            dtype=torch.int64,
            device=reference.device,
        )
        backbone_atoms = (
            reference.block_coord_offset64[0, source_block] + backbone_local
        )
        torch.testing.assert_close(
            result.best_pose_stack.coords[0, backbone_atoms],
            reference.coords[0, backbone_atoms],
            rtol=0.0,
            atol=0.0,
        )

    print(
        "flexible cdk2 native/best, moved atoms, bad bonds, wall_s:",
        float(native_score),
        float(result.best_energies[0]),
        moved_count,
        bad_bonds,
        wall_seconds,
    )
    assert moved_count > 0
    assert bad_bonds == 0


def test_rigid_none_argument_is_byte_identical(cpu_grid_case):
    case = cpu_grid_case
    schedule = GASchedule(generations=1, npool=2, maxiter=2)
    historical = run_ga_docking(
        case.reference,
        case.ligand_block,
        case.backend,
        schedule=schedule,
        seed=20260825,
    )
    explicit_none = run_ga_docking(
        case.reference,
        case.ligand_block,
        case.backend,
        schedule=schedule,
        seed=20260825,
        sidechain_blocks=None,
    )
    assert torch.equal(historical.pool.genes, explicit_none.pool.genes)
    assert torch.equal(historical.pool.energies, explicit_none.pool.energies)
    assert torch.equal(
        historical.best_pose_stack.coords, explicit_none.best_pose_stack.coords
    )
