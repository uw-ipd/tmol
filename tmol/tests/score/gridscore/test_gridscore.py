from __future__ import annotations

import math

import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
from tmol.score.elec.params import ElecParamResolver
from tmol.score.gridscore import (
    GridGeometry,
    GridSet,
    build_grid,
    compute_cubic_bspline_coefficients,
    compute_grid_geometry,
    interpolate_grid,
    ixform_rep,
    probe_atom_types_from_name3,
    score_ligand_grid,
    xform_rep,
)
from tmol.score.gridscore.grid import _smooth_fa_rep_grid
from tmol.score.ljlk.params import LJLKParamResolver
from tmol.tests.data import data_path
from tmol.tests.ligand.test_protein_ligand_ddg import (
    LIGAND_RES_NAME,
    _load_complex_cif,
)

PLI_DATA_DIR = data_path("protein_ligand_test")


def test_grid_geometry_index_conventions():
    ligand_coords = torch.tensor(
        [[-8.0, 1.0, 2.0], [8.0, 1.0, 2.0]], dtype=torch.float64
    )
    geometry = compute_grid_geometry(ligand_coords)

    assert geometry.max_radius == 8.0
    assert geometry.dims == (96, 96, 96)
    torch.testing.assert_close(
        geometry.ligand_com, torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        geometry.origin, torch.tensor([-12.0, -11.0, -10.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        geometry.cartesian_from_zero_based_index((3, 4, 5)),
        torch.tensor([-11.25, -10.0, -8.75], dtype=torch.float64),
    )
    torch.testing.assert_close(
        geometry.cartesian_from_one_based_index((4, 5, 6)),
        geometry.cartesian_from_zero_based_index((3, 4, 5)),
    )
    torch.testing.assert_close(
        geometry.zero_based_index_from_cartesian(
            geometry.cartesian_from_zero_based_index((3, 4, 5))
        ),
        torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64),
    )

    floored = compute_grid_geometry(torch.zeros((2, 3), dtype=torch.float64))
    assert floored.max_radius == 6.0
    assert floored.dims == (80, 80, 80)
    torch.testing.assert_close(
        floored.origin, torch.full((3,), -10.0, dtype=torch.float64)
    )


def test_rep_transform_round_trip():
    raw = torch.tensor([0.0, 1.0e-8, 1.0, 100.0, 1.0e8], dtype=torch.float64)
    torch.testing.assert_close(ixform_rep(xform_rep(raw)), raw, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("smoothing", (0.0, 0.125, 0.25, 0.375, 1.25))
def test_fa_rep_smoothing_matches_spherical_maximum(dtype, smoothing):
    generator = torch.Generator().manual_seed(537396662)
    fa_rep = xform_rep(100 * torch.rand((2, 3, 4, 5), dtype=dtype, generator=generator))
    original = fa_rep.clone()
    actual = _smooth_fa_rep_grid(fa_rep, smoothing, 0.25)

    # Independent all-pairs oracle: clip the sphere to the grid at boundaries
    # and keep atom-type channels separate. No wrapping or repeated dilation.
    indices = torch.cartesian_prod(torch.arange(3), torch.arange(4), torch.arange(5))
    positions = indices.to(torch.float64) * 0.25
    distances_squared = ((positions[:, None] - positions[None, :]) ** 2).sum(-1)
    expected = torch.stack(
        [
            fa_rep.flatten(1)[:, neighbors].max(dim=1).values
            for neighbors in distances_squared <= smoothing**2
        ],
        dim=1,
    ).reshape_as(fa_rep)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(fa_rep, original, rtol=0, atol=0)
    assert torch.all(actual >= fa_rep)
    if smoothing == 0:
        assert actual is fa_rep


def test_build_grid_fa_rep_smoothing():
    pose_stack, param_db = _load_cdk2_pose()
    probe_names = probe_atom_types_from_name3(pose_stack, LIGAND_RES_NAME)
    kwargs = dict(ligand_name3=LIGAND_RES_NAME, voxel_spacing=0.5)
    default = build_grid(pose_stack, probe_names, param_db, **kwargs)
    zero = build_grid(pose_stack, probe_names, param_db, fa_rep_smoothing=0.0, **kwargs)
    smoothed = build_grid(
        pose_stack, probe_names, param_db, fa_rep_smoothing=0.75, **kwargs
    )
    for name in (
        "fa_atr",
        "fa_rep",
        "fa_sol",
        "fa_elec",
        "fa_atr_coeffs",
        "fa_rep_coeffs",
        "fa_sol_coeffs",
        "fa_elec_coeffs",
    ):
        torch.testing.assert_close(
            getattr(zero, name), getattr(default, name), rtol=0, atol=0
        )
        if not name.startswith("fa_rep"):
            torch.testing.assert_close(
                getattr(smoothed, name), getattr(default, name), rtol=0, atol=0
            )
    assert torch.all(smoothed.fa_rep >= default.fa_rep)
    assert torch.any(smoothed.fa_rep > default.fa_rep)
    torch.testing.assert_close(
        smoothed.fa_rep,
        _smooth_fa_rep_grid(default.fa_rep, 0.75, default.geometry.voxel_spacing),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        smoothed.fa_rep_coeffs,
        compute_cubic_bspline_coefficients(smoothed.fa_rep),
        rtol=0,
        atol=0,
    )

    for invalid in (-0.1, float("nan"), float("inf"), -float("inf")):
        with pytest.raises(
            ValueError, match="fa_rep_smoothing must be finite and nonnegative"
        ):
            build_grid(
                pose_stack, probe_names, param_db, fa_rep_smoothing=invalid, **kwargs
            )
        with pytest.raises(
            ValueError, match="fa_rep_smoothing must be finite and nonnegative"
        ):
            _smooth_fa_rep_grid(default.fa_rep, invalid, default.geometry.voxel_spacing)


def test_grid_serialization_round_trip(tmp_path):
    geometry = GridGeometry(
        origin=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        dims=(2, 3, 4),
        voxel_spacing=0.5,
        ligand_com=torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64),
        max_radius=7.5,
    )
    typed = torch.arange(48, dtype=torch.float64).reshape(2, 2, 3, 4)
    shared = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    grids = GridSet(
        geometry=geometry,
        atom_type_names=("CH1", "OOC"),
        atom_type_indices=torch.tensor([1, 2], dtype=torch.int64),
        fa_atr=typed,
        fa_rep=typed + 1,
        fa_sol=typed + 2,
        fa_elec=shared,
        lk_ball=typed + 3,
        lk_ball_water=typed + 4,
        lk_bridge=typed + 5,
        lk_ball_enabled=True,
        lk_bridge_enabled=True,
    )

    path = tmp_path / "receptor_grids.pt"
    grids.save(path)
    loaded = GridSet.load(path)

    assert loaded.atom_type_names == grids.atom_type_names
    assert loaded.atom_type_to_grid == {"CH1": 0, "OOC": 1}
    assert loaded.geometry.dims == grids.geometry.dims
    assert loaded.lk_ball_enabled
    assert loaded.lk_bridge_enabled
    torch.testing.assert_close(loaded.geometry.origin, grids.geometry.origin)
    torch.testing.assert_close(loaded.atom_type_indices, grids.atom_type_indices)
    torch.testing.assert_close(loaded.fa_atr, grids.fa_atr)
    torch.testing.assert_close(loaded.fa_rep, grids.fa_rep)
    torch.testing.assert_close(loaded.fa_lk, grids.fa_sol)
    torch.testing.assert_close(loaded.fa_elec, grids.fa_elec)
    torch.testing.assert_close(loaded.lk_ball, grids.lk_ball)
    torch.testing.assert_close(loaded.lk_ball_water, grids.lk_ball_water)
    torch.testing.assert_close(loaded.lk_bridge, grids.lk_bridge)
    torch.testing.assert_close(loaded.fa_atr_coeffs, grids.fa_atr_coeffs)
    torch.testing.assert_close(loaded.fa_rep_coeffs, grids.fa_rep_coeffs)
    torch.testing.assert_close(loaded.fa_sol_coeffs, grids.fa_sol_coeffs)
    torch.testing.assert_close(loaded.fa_elec_coeffs, grids.fa_elec_coeffs)
    torch.testing.assert_close(loaded.lk_ball_coeffs, grids.lk_ball_coeffs)
    torch.testing.assert_close(loaded.lk_ball_water_coeffs, grids.lk_ball_water_coeffs)
    torch.testing.assert_close(loaded.lk_bridge_coeffs, grids.lk_bridge_coeffs)

    legacy_state = torch.load(path, weights_only=True)
    legacy_state["version"] = 1
    for name in (
        "fa_atr_coeffs",
        "fa_rep_coeffs",
        "fa_sol_coeffs",
        "fa_elec_coeffs",
        "lk_ball_coeffs",
        "lk_ball_water_coeffs",
        "lk_bridge_coeffs",
    ):
        legacy_state.pop(name)
    legacy_path = tmp_path / "m2b_receptor_grids.pt"
    torch.save(legacy_state, legacy_path)
    migrated = GridSet.load(legacy_path)
    torch.testing.assert_close(migrated.fa_atr_coeffs, grids.fa_atr_coeffs)
    torch.testing.assert_close(migrated.fa_rep_coeffs, grids.fa_rep_coeffs)
    torch.testing.assert_close(migrated.fa_sol_coeffs, grids.fa_sol_coeffs)
    torch.testing.assert_close(migrated.fa_elec_coeffs, grids.fa_elec_coeffs)
    torch.testing.assert_close(migrated.lk_ball_coeffs, grids.lk_ball_coeffs)
    torch.testing.assert_close(
        migrated.lk_ball_water_coeffs, grids.lk_ball_water_coeffs
    )
    torch.testing.assert_close(migrated.lk_bridge_coeffs, grids.lk_bridge_coeffs)


def test_cubic_bspline_round_trip_at_every_node():
    generator = torch.Generator().manual_seed(1977)
    values = torch.randn((2, 5, 6, 7), dtype=torch.float64, generator=generator)
    coefficients = compute_cubic_bspline_coefficients(values)

    node_indices = torch.cartesian_prod(
        torch.arange(5, dtype=torch.float64),
        torch.arange(6, dtype=torch.float64),
        torch.arange(7, dtype=torch.float64),
    )
    all_indices = node_indices.repeat(2, 1)
    grid_indices = torch.arange(2, dtype=torch.int64).repeat_interleave(
        node_indices.shape[0]
    )
    interpolated = interpolate_grid(coefficients, all_indices, grid_indices)
    error = torch.max(torch.abs(interpolated - values.reshape(-1)))

    print(f"cubic B-spline all-node round-trip max-abs error: {float(error):.3e}")
    torch.testing.assert_close(interpolated, values.reshape(-1), rtol=0.0, atol=1.0e-9)
    # The Rosetta origin sentinel is intentionally absent: this value is data.
    torch.testing.assert_close(interpolated[0], values[0, 0, 0, 0])


def test_cubic_bspline_prefilter_uses_mirror_boundary_coefficients():
    values = torch.tensor([[[1.0, 3.0]]], dtype=torch.float64)
    coefficients = compute_cubic_bspline_coefficients(values)
    # With mirrored neighbors, samples obey f0=(4*c0+2*c1)/6 and
    # f1=(2*c0+4*c1)/6, whose solution here is exactly (-1, 5).
    torch.testing.assert_close(
        coefficients,
        torch.tensor([[[-1.0, 5.0]]], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_cubic_bspline_cpu_cuda_agreement(torch_device):
    generator = torch.Generator().manual_seed(2214)
    values = torch.randn((3, 6, 7, 8), dtype=torch.float64, generator=generator)
    coefficients = compute_cubic_bspline_coefficients(values).transpose(2, 3)
    contiguous_indices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [5.0, 6.0, 7.0],
            [0.125, 2.75, 6.5],
            [-0.25, 6.25, 8.5],
        ],
        dtype=torch.float64,
    )
    index_storage = torch.zeros((4, 6), dtype=torch.float64)
    index_storage[:, ::2] = contiguous_indices
    indices = index_storage[:, ::2]
    grid_index_storage = torch.zeros(8, dtype=torch.int64)
    grid_index_storage[::2] = torch.tensor([0, 1, 2, 1], dtype=torch.int64)
    grid_indices = grid_index_storage[::2]
    assert not coefficients.is_contiguous()
    assert not indices.is_contiguous()
    assert not grid_indices.is_contiguous()
    expected = interpolate_grid(coefficients, indices, grid_indices)
    actual = interpolate_grid(
        coefficients.to(torch_device),
        indices.to(torch_device),
        grid_indices.to(torch_device),
    ).cpu()
    error = torch.max(torch.abs(actual - expected))

    print(f"cubic B-spline CPU/device max-abs error: {float(error):.3e}")
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-10)


def test_weighted_ligand_grid_scorer_uses_rep_transform_and_charge():
    geometry = GridGeometry(
        origin=torch.zeros(3, dtype=torch.float64),
        dims=(4, 4, 4),
        voxel_spacing=0.5,
        ligand_com=torch.zeros(3, dtype=torch.float64),
        max_radius=7.5,
    )
    shape = (2, *geometry.dims)
    fa_atr = torch.empty(shape, dtype=torch.float64)
    fa_atr[0].fill_(-2.0)
    fa_atr[1].fill_(-4.0)
    raw_rep = torch.empty(shape, dtype=torch.float64)
    raw_rep[0].fill_(3.0)
    raw_rep[1].fill_(5.0)
    fa_sol = torch.empty(shape, dtype=torch.float64)
    fa_sol[0].fill_(0.5)
    fa_sol[1].fill_(1.5)
    grids = GridSet(
        geometry=geometry,
        atom_type_names=("A", "B"),
        atom_type_indices=torch.tensor([4, 9], dtype=torch.int64),
        fa_atr=fa_atr,
        fa_rep=xform_rep(raw_rep),
        fa_sol=fa_sol,
        fa_elec=torch.full(geometry.dims, 2.0, dtype=torch.float64),
    )
    actual = score_ligand_grid(
        grids,
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64),
        ("A", "B"),
        torch.tensor([0.25, -0.5], dtype=torch.float64),
    )
    expected = torch.tensor([-6.0, 0.55 * 8.0, 2.0, -0.5], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1.0e-7, atol=1.0e-7)


def _load_target_pose(target):
    param_db = ParameterDatabase.get_default()
    pose_stack, context = pose_stack_from_biotite(
        _load_complex_cif(target),
        torch.device("cpu"),
        prepare_ligands=True,
        ligand_params_files=[str(PLI_DATA_DIR / f"{target}.xtal-lig.mmff94.tmol")],
        no_optH=True,
        param_db=param_db,
        return_context=True,
    )
    pose_stack.coords = pose_stack.coords.to(dtype=torch.float64)
    return pose_stack, context.parameter_database


def _load_cdk2_pose():
    return _load_target_pose("cdk2")


def _independent_receptor_extraction(pose_stack):
    pbt = pose_stack.packed_block_types
    coords = []
    atom_type_names = []
    charges = []
    ligand_coords = []
    for block_index, block_type_index_tensor in enumerate(pose_stack.block_type_ind[0]):
        block_type_index = int(block_type_index_tensor)
        if block_type_index < 0:
            continue
        block_type = pbt.active_block_types[block_type_index]
        n_atoms = int(pbt.n_atoms[block_type_index])
        offset = int(pose_stack.block_coord_offset64[0, block_index])
        block_coords = pose_stack.coords[0, offset : offset + n_atoms]
        if block_type.name3 == LIGAND_RES_NAME:
            ligand_coords.append(block_coords)
            continue
        coords.append(block_coords)
        atom_type_names.extend(atom.atom_type for atom in block_type.atoms[:n_atoms])
        charges.append(pbt.elec_partial_charge[block_type_index, :n_atoms])
    return (
        torch.cat(coords).to(torch.float64).contiguous(),
        tuple(atom_type_names),
        torch.cat(charges).to(torch.float64).contiguous(),
        torch.cat(ligand_coords).to(torch.float64).contiguous(),
    )


def _sample_voxel_indices(geometry, receptor_coords):
    dims = torch.tensor(geometry.dims, dtype=torch.int64)
    indices = {
        (0, 0, 0),
        tuple((dims // 2).tolist()),
        tuple((dims - 1).tolist()),
    }
    continuous = geometry.zero_based_index_from_cartesian(receptor_coords)
    for index in torch.round(continuous).to(torch.int64):
        if bool(torch.all(index >= 0) and torch.all(index < dims)):
            indices.add(tuple(index.tolist()))
        if len(indices) >= 12:
            break
    return sorted(indices)


def test_cdk2_grid_matches_independent_pair_sums(tmp_path):
    from tmol.tests.score.ljlk.potentials._compiled import (
        elec_V,
        lj_score_V,
        lk_isotropic_score_V,
    )

    pose_stack, param_db = _load_cdk2_pose()
    probe_names = probe_atom_types_from_name3(pose_stack, LIGAND_RES_NAME)
    grids = build_grid(
        pose_stack,
        probe_names,
        param_db,
        ligand_name3=LIGAND_RES_NAME,
    )

    receptor_coords, receptor_type_names, receptor_charges, ligand_coords = (
        _independent_receptor_extraction(pose_stack)
    )
    expected_geometry = compute_grid_geometry(ligand_coords)
    assert grids.geometry.dims == expected_geometry.dims
    assert grids.geometry.voxel_spacing == 0.25
    torch.testing.assert_close(grids.geometry.origin, expected_geometry.origin)

    # This oracle deliberately resolves every atom type by name and calls the
    # pre-existing test bindings for the production pair potentials. It does
    # not share gridscore's type-index packing or its C++ pair evaluator.
    ljlk_params = LJLKParamResolver.from_database(
        param_db.chemical, param_db.scoring.ljlk, torch.device("cpu")
    )
    elec_params = ElecParamResolver.from_database(
        param_db.scoring.elec, torch.device("cpu")
    )
    receptor_params = [
        ljlk_params.type_params[ljlk_params.atom_type_index.get_loc(receptor_type_name)]
        for receptor_type_name in receptor_type_names
    ]
    elec_global = elec_params.global_params

    max_errors = {"fa_atr": 0.0, "fa_rep": 0.0, "fa_sol": 0.0, "fa_elec": 0.0}
    sample_indices = _sample_voxel_indices(grids.geometry, receptor_coords)
    for index in sample_indices:
        xyz = grids.geometry.cartesian_from_zero_based_index(index)
        distances = torch.linalg.vector_norm(receptor_coords - xyz, dim=1)
        reference_elec = sum(
            elec_V(
                float(distance),
                1.0,
                float(charge),
                5.0,
                float(elec_global.elec_sigmoidal_die_D),
                float(elec_global.elec_sigmoidal_die_D0),
                float(elec_global.elec_sigmoidal_die_S),
                float(elec_global.elec_min_dis),
                float(elec_global.elec_max_dis),
            )
            for distance, charge in zip(distances, receptor_charges)
        )
        for probe_grid_index, probe_name in enumerate(grids.atom_type_names):
            probe_params = ljlk_params.type_params[
                ljlk_params.atom_type_index.get_loc(probe_name)
            ]
            reference_atr = 0.0
            reference_rep = 0.0
            reference_sol = 0.0
            for distance, recept_params in zip(distances, receptor_params):
                atr, rep = lj_score_V(
                    float(distance),
                    5.0,
                    probe_params,
                    recept_params,
                    ljlk_params.global_params,
                )
                reference_atr += atr
                reference_rep += rep
                reference_sol += lk_isotropic_score_V(
                    float(distance),
                    5.0,
                    probe_params,
                    recept_params,
                    ljlk_params.global_params,
                )
            reference = torch.tensor(
                [reference_atr, reference_rep, reference_sol, reference_elec],
                dtype=torch.float64,
            )
            actual = torch.stack(
                [
                    grids.fa_atr[probe_grid_index][index],
                    ixform_rep(grids.fa_rep[probe_grid_index][index]),
                    grids.fa_sol[probe_grid_index][index],
                    grids.fa_elec[index],
                ]
            )
            torch.testing.assert_close(
                grids.fa_rep[probe_grid_index][index],
                xform_rep(reference[1]),
                rtol=1e-6,
                atol=1e-6,
            )
            for term, error in zip(max_errors, torch.abs(actual - reference)):
                max_errors[term] = max(max_errors[term], float(error))
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)

    print(
        "cdk2 grid max-abs errors: "
        + ", ".join(f"{term}={error:.3e}" for term, error in max_errors.items())
    )
    assert torch.isfinite(grids.fa_atr).all()
    assert torch.isfinite(grids.fa_rep).all()
    assert torch.isfinite(grids.fa_sol).all()
    assert torch.isfinite(grids.fa_elec).all()
    assert float(grids.fa_atr.min()) < 0
    assert float(ixform_rep(grids.fa_rep).max()) > 10
    assert float(grids.fa_elec.min()) < 0 < float(grids.fa_elec.max())

    serialized = tmp_path / "cdk2_grid.pt"
    grids.save(serialized)
    reloaded = GridSet.load(serialized)
    assert reloaded.atom_type_to_grid == grids.atom_type_to_grid
    torch.testing.assert_close(reloaded.fa_atr, grids.fa_atr)
    torch.testing.assert_close(reloaded.fa_rep, grids.fa_rep)
    torch.testing.assert_close(reloaded.fa_sol, grids.fa_sol)
    torch.testing.assert_close(reloaded.fa_elec, grids.fa_elec)

    assert math.isfinite(max(max_errors.values()))


def _ligand_scoring_inputs(pose_stack):
    pbt = pose_stack.packed_block_types
    coords = []
    atom_type_names = []
    charges = []
    for block_index, block_type_index_tensor in enumerate(pose_stack.block_type_ind[0]):
        block_type_index = int(block_type_index_tensor)
        if block_type_index < 0:
            continue
        block_type = pbt.active_block_types[block_type_index]
        if block_type.name3 != LIGAND_RES_NAME:
            continue
        n_atoms = int(pbt.n_atoms[block_type_index])
        offset = int(pose_stack.block_coord_offset64[0, block_index])
        coords.append(pose_stack.coords[0, offset : offset + n_atoms])
        atom_type_names.extend(atom.atom_type for atom in block_type.atoms[:n_atoms])
        charges.append(pbt.elec_partial_charge[block_type_index, :n_atoms])
    return (
        torch.cat(coords).to(dtype=torch.float64).contiguous(),
        tuple(atom_type_names),
        torch.cat(charges).to(dtype=torch.float64).contiguous(),
    )


@pytest.mark.slow
@pytest.mark.parametrize("target", ("cdk2", "ache"))
def test_grid_interface_characterization_against_t1(target):
    from tmol.tests.ligand.test_ligand_receptor_perterm_baseline import (
        _GOLDEN_PER_TERM,
    )

    pose_stack, param_db = _load_target_pose(target)
    probe_names = probe_atom_types_from_name3(pose_stack, LIGAND_RES_NAME)
    grids = build_grid(
        pose_stack,
        probe_names,
        param_db,
        ligand_name3=LIGAND_RES_NAME,
    )
    ligand_coords, ligand_atom_types, ligand_charges = _ligand_scoring_inputs(
        pose_stack
    )
    grid_score = score_ligand_grid(
        grids, ligand_coords, ligand_atom_types, ligand_charges
    )
    exact = torch.tensor(_GOLDEN_PER_TERM[target]["interface"][:4], dtype=torch.float64)
    absolute_error = torch.abs(grid_score - exact)
    relative_error = absolute_error / torch.clamp(torch.abs(exact), min=1.0e-12)

    term_names = ("fa_atr", "fa_rep", "fa_sol", "fa_elec")
    print(
        f"{target} grid-vs-T1: "
        + ", ".join(
            f"{name} abs={float(abs_err):.6g} rel={float(rel_err):.6g}"
            for name, abs_err, rel_err in zip(
                term_names, absolute_error, relative_error
            )
        )
    )
    assert torch.isfinite(grid_score).all()
    assert torch.isfinite(absolute_error).all()
    assert torch.isfinite(relative_error).all()
    # Measurement-derived per-term regression bound (F11). The original 0.5 A
    # characterization observed max-abs across cdk2+ache of fa_atr 0.022, fa_rep
    # 0.253, fa_sol 0.022, and fa_elec 0.042 REU. These bounds are retained for
    # the finer 0.25 A grid, whose interpolation error should be smaller
    # (fa_rep remains the stiff outlier).
    bound = torch.tensor([0.05, 0.50, 0.05, 0.10], dtype=torch.float64)
    assert torch.all(absolute_error <= bound), (
        f"{target} grid-vs-T1 per-term abs error {absolute_error.tolist()} "
        f"exceeds bound {bound.tolist()}"
    )


def _direct_grid_inputs(param_db, atom_type_name):
    """Pack one named type explicitly for focused compiled-builder tests."""
    ljlk_params = LJLKParamResolver.from_database(
        param_db.chemical, param_db.scoring.ljlk, torch.device("cpu")
    )
    elec_params = ElecParamResolver.from_database(
        param_db.scoring.elec, torch.device("cpu")
    )
    type_index = ljlk_params.atom_type_index.get_loc(atom_type_name)
    type_params = ljlk_params.type_params[type_index]
    packed_type = torch.tensor(
        [
            float(type_params.lj_radius),
            float(type_params.lj_wdepth),
            float(type_params.lk_dgfree),
            float(type_params.lk_lambda),
            float(type_params.lk_volume),
            float(type_params.is_donor),
            float(type_params.is_hydroxyl),
            float(type_params.is_polarh),
            float(type_params.is_acceptor),
            float(type_params.is_carbon_lk),
        ],
        dtype=torch.float64,
    ).reshape(1, 10)
    ljlk_global = torch.tensor(
        [
            float(ljlk_params.global_params.max_dis),
            float(ljlk_params.global_params.lj_dlin_sigma_factor),
            float(ljlk_params.global_params.lj_hbond_dis),
            float(ljlk_params.global_params.lj_hbond_OH_donor_dis),
            float(ljlk_params.global_params.lj_hbond_hdis),
        ],
        dtype=torch.float64,
    ).reshape(1, 5)
    elec_global_params = elec_params.global_params
    elec_global = torch.tensor(
        [
            float(elec_global_params.elec_sigmoidal_die_D),
            float(elec_global_params.elec_sigmoidal_die_D0),
            float(elec_global_params.elec_sigmoidal_die_S),
            float(elec_global_params.elec_min_dis),
            float(elec_global_params.elec_max_dis),
        ],
        dtype=torch.float64,
    )
    return (
        packed_type,
        ljlk_global,
        elec_global,
        type_params,
        ljlk_params.global_params,
    )


def _build_single_voxel_grid(
    receptor_coords, receptor_charges, packed_type, ljlk_global, elec_global, origin
):
    from tmol.score.gridscore.potentials.compiled import build_grid as build_grid_op

    n_receptor_atoms = receptor_coords.shape[0]
    return build_grid_op(
        receptor_coords.to(dtype=torch.float64).contiguous(),
        torch.zeros(n_receptor_atoms, dtype=torch.int64),
        receptor_charges.to(dtype=torch.float64).contiguous(),
        torch.zeros(1, dtype=torch.int64),
        origin.to(dtype=torch.float64).contiguous(),
        torch.ones(3, dtype=torch.int64),
        0.5,
        packed_type,
        ljlk_global,
        elec_global,
    )


def test_grid_electrostatic_sign_for_isolated_signed_charge():
    from tmol.tests.score.ljlk.potentials._compiled import elec_V

    param_db = ParameterDatabase.get_default()
    packed_type, ljlk_global, elec_global, _, _ = _direct_grid_inputs(param_db, "CH1")
    origin = torch.tensor([-3.0, -4.0, -5.0])
    receptor_coords = torch.tensor([[-2.0, -4.0, -5.0]])

    for charge in (-0.5, 0.5):
        grids = _build_single_voxel_grid(
            receptor_coords,
            torch.tensor([charge]),
            packed_type,
            ljlk_global,
            elec_global,
            origin,
        )
        actual = float(grids[3][0, 0, 0])
        expected = elec_V(1.0, 1.0, charge, 5.0, *elec_global.tolist())
        assert math.copysign(1.0, actual) == math.copysign(1.0, charge)
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_grid_cutoff_and_negative_bin_boundary():
    from tmol.tests.score.ljlk.potentials import (
        lj_score_V,
        lk_isotropic_score_V,
    )

    param_db = ParameterDatabase.get_default()
    packed_type, ljlk_global, elec_global, type_params, global_params = (
        _direct_grid_inputs(param_db, "CH1")
    )
    cutoff = float(ljlk_global[0, 0])
    delta = 0.01
    # float64 coords: near the steep LJ cutoff, float32 coord rounding (~1e-7)
    # would exceed the 1e-12 parity tolerance below (fa_atr ~0 there inflates the
    # relative error). The grid math is float64; keep the test inputs float64 too.
    origin = torch.tensor([-0.25, -1.0, -1.0], dtype=torch.float64)
    inside_coord = torch.tensor(
        [[-0.25 - cutoff + delta, -1.0, -1.0]], dtype=torch.float64
    )
    outside_coord = torch.tensor(
        [[-0.25 - cutoff - delta, -1.0, -1.0]], dtype=torch.float64
    )
    charges = torch.zeros(2, dtype=torch.float64)

    both = _build_single_voxel_grid(
        torch.cat((inside_coord, outside_coord)),
        charges,
        packed_type,
        ljlk_global,
        elec_global,
        origin,
    )
    inside_only = _build_single_voxel_grid(
        inside_coord,
        charges[:1],
        packed_type,
        ljlk_global,
        elec_global,
        origin,
    )
    expected_inside_lj = lj_score_V(
        cutoff - delta, 5.0, type_params, type_params, global_params
    )
    expected_inside_sol = lk_isotropic_score_V(
        cutoff - delta, 5.0, type_params, type_params, global_params
    )
    expected_outside_lj = lj_score_V(
        cutoff + delta, 5.0, type_params, type_params, global_params
    )
    expected_outside_sol = lk_isotropic_score_V(
        cutoff + delta, 5.0, type_params, type_params, global_params
    )

    assert float(inside_coord[0, 0]) < 0
    assert float(outside_coord[0, 0]) < 0
    assert expected_inside_lj[0] != 0.0
    assert tuple(expected_outside_lj) == (0.0, 0.0)
    assert expected_outside_sol == 0.0
    for both_grid, inside_grid in zip(both, inside_only):
        torch.testing.assert_close(both_grid, inside_grid, rtol=0, atol=0)
    actual_inside = torch.stack(
        (
            both[0][0, 0, 0, 0],
            ixform_rep(both[1][0, 0, 0, 0]),
            both[2][0, 0, 0, 0],
        )
    )
    expected_inside = torch.tensor(
        [expected_inside_lj[0], expected_inside_lj[1], expected_inside_sol],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual_inside, expected_inside, rtol=1e-12, atol=1e-12)
