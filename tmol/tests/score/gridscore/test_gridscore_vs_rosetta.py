"""Cross-validate tmol's isotropic grids against a Rosetta GridScorer dump."""

from __future__ import annotations

import math
import json
import os
from pathlib import Path

import numpy
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite
from tmol.score.gridscore import build_grid
from tmol.score.ljlk import LJLKParamResolver
from tmol.tests.data import data_path
from tmol.tests.ligand.test_protein_ligand_ddg import (
    LIGAND_RES_NAME,
    _load_complex_cif,
)

ROSETTA_GRID_DUMP = Path(
    os.environ.get(
        "TMOL_ROSETTA_GRID_DUMP",
        # Parameter-matched beta_genpot dump (-beta): fa_standard_genpot ATS +
        # beta_nov16 LK_DGFREE / charges / dielectric == tmol's params. See F15.
        "/home/gzhou/git/rosetta_private/source/tools/"
        "gridscorer_debug_dump/output/cdk2_genpot.grids",
    )
)
MAX_ABS_TOLERANCE = 1.0e-3
# fa_elec is NOT gated at bit-parity yet: its ~0.35 residual is the deferred
# receptor charge-convention difference (F21, charges frozen). This loose bound only
# catches unexpected drift/regression from that known value. Tighten to
# MAX_ABS_TOLERANCE once F21 is resolved.
FAELEC_CHARACTERIZATION_BOUND = 0.40
PLI_DATA_DIR = data_path("protein_ligand_test")


def _load_cdk2_pose():
    param_db = ParameterDatabase.get_default()
    pose_stack, context = pose_stack_from_biotite(
        _load_complex_cif("cdk2"),
        torch.device("cpu"),
        prepare_ligands=True,
        ligand_params_files=[str(PLI_DATA_DIR / "cdk2.xtal-lig.mmff94.tmol")],
        no_optH=True,
        param_db=param_db,
        return_context=True,
    )
    pose_stack.coords = pose_stack.coords.to(dtype=torch.float64)
    return pose_stack, context.parameter_database


def _raw_entries_by_channel(manifest):
    entries = {}
    for entry in manifest["grids"]:
        if entry["kind"] != "raw":
            continue
        channel = entry["channel"]
        if channel not in ("faatr", "farep", "fasol", "faelec"):
            continue
        key = entry["atom_type_name"]
        channel_entries = entries.setdefault(channel, {})
        assert key not in channel_entries
        channel_entries[key] = entry
    return entries


@pytest.mark.slow
def test_raw_grids_match_rosetta_dump():
    manifest_path = ROSETTA_GRID_DUMP / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"Rosetta grid dump is absent: {ROSETTA_GRID_DUMP}")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["format_version"] == 1
    assert manifest["dtype"] == "float32"
    assert manifest["fortran_order"] is True
    assert manifest["origin_index"] == [1, 1, 1]
    assert manifest["voxel_spacing"] == 0.25
    raw_entries = _raw_entries_by_channel(manifest)
    assert set(raw_entries) == {"faatr", "farep", "fasol", "faelec"}

    typed_channels = ("faatr", "farep", "fasol")
    probe_name_sets = [set(raw_entries[channel]) for channel in typed_channels]
    assert all(names == probe_name_sets[0] for names in probe_name_sets[1:])
    assert set(raw_entries["faelec"]) == {None}
    probe_names = tuple(sorted(probe_name_sets[0]))

    pose_stack, param_db = _load_cdk2_pose()
    resolver = LJLKParamResolver.from_database(
        param_db.chemical, param_db.scoring.ljlk, torch.device("cpu")
    )
    expected_type_indices = torch.tensor(
        [resolver.atom_type_index.get_loc(name) for name in probe_names],
        dtype=torch.int64,
    )
    grids = build_grid(
        pose_stack,
        probe_names,
        param_db,
        ligand_name3=LIGAND_RES_NAME,
    )
    assert grids.atom_type_names == probe_names
    torch.testing.assert_close(grids.atom_type_indices, expected_type_indices)

    # This is a required gate: Rosetta array [i, j, k] and tmol array [i, j, k]
    # must describe the same Cartesian node before any values are compared.
    assert grids.geometry.dims == tuple(manifest["dims"])
    assert grids.geometry.voxel_spacing == manifest["voxel_spacing"]
    # Origins come from independently-parsed coordinates (Rosetta reads the PDB,
    # tmol reads the CIF), so the ligand COM — and thus the origin — agree only to
    # coordinate-source roundoff (~1e-7), not to machine epsilon. A sub-milliangstrom
    # bound still catches any real geometry misalignment (which would be >= 0.25 A),
    # and a ~1e-7 origin shift is negligible for the voxelwise value comparison.
    torch.testing.assert_close(
        grids.geometry.origin,
        torch.tensor(manifest["origin"], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-3,
    )

    tmol_by_channel = {
        "faatr": grids.fa_atr,
        "farep": grids.fa_rep,
        "fasol": grids.fa_sol,
    }
    errors = {}
    expected_shape = tuple(manifest["dims"])
    for channel in typed_channels:
        actual_grids = tmol_by_channel[channel]
        for grid_index, atom_type_name in enumerate(probe_names):
            entry = raw_entries[channel][atom_type_name]
            reference_array = numpy.load(ROSETTA_GRID_DUMP / entry["file"])
            assert reference_array.shape == expected_shape
            reference = torch.from_numpy(reference_array.copy()).to(torch.float64)
            errors[(channel, atom_type_name)] = float(
                torch.max(torch.abs(actual_grids[grid_index] - reference))
            )

    elec_entry = raw_entries["faelec"][None]
    reference_array = numpy.load(ROSETTA_GRID_DUMP / elec_entry["file"])
    assert reference_array.shape == expected_shape
    reference_elec = torch.from_numpy(reference_array.copy()).to(torch.float64)
    errors[("faelec", "unit-charge")] = float(
        torch.max(torch.abs(grids.fa_elec - reference_elec))
    )

    print("Rosetta raw-grid voxelwise max-abs errors:")
    for (channel, atom_type_name), error in sorted(errors.items()):
        print(f"  {channel}/{atom_type_name}: {error:.6e}")
    for channel in ("faatr", "farep", "fasol", "faelec"):
        channel_max = max(
            error
            for (error_channel, _), error in errors.items()
            if error_channel == channel
        )
        print(f"  {channel} channel max: {channel_max:.6e}")

    assert all(math.isfinite(error) for error in errors.values())

    # HARD GATE — fa_atr / fa_rep / fa_sol are BIT-PARITY with the Rosetta beta_genpot
    # dump (F19/F20): the grid kernel is proven correct and the C-terminal COO typing
    # fix brought fa_sol to ~6.5e-6. These must not regress.
    # The dump is produced with -beta (== gen_potential + beta_nov16): fa_standard is
    # loaded from fa_standard_genpot (ligand type 'Nim' resolves) and
    # init_beta_nov16_correction applies the beta LK_DGFREE / charges / dielectric that
    # tmol also uses (verified value-by-value, F15).
    gated_channels = ("faatr", "farep", "fasol")
    failures = {
        f"{channel}/{atom_type_name}": error
        for (channel, atom_type_name), error in errors.items()
        if channel in gated_channels and error > MAX_ABS_TOLERANCE
    }
    assert not failures, (
        f"fa_atr/fa_rep/fa_sol vs the beta_genpot Rosetta dump exceed "
        f"{MAX_ABS_TOLERANCE} (max-abs per voxel): {failures}"
    )

    # CHARACTERIZATION (NOT bit-parity) — fa_elec. The grid KERNEL is proven correct
    # (running tmol's kernel over Rosetta's own receptor charges gives fa_elec 6.75e-6,
    # F19). The residual ~0.35 is entirely a receptor CHARGE-CONVENTION difference
    # (ASN/GLN amide-H syn/anti + N-terminal charges, F20/F21). Reaching bit-parity
    # requires changing tmol's SHARED elec charge DB, which also changes the continuous
    # elec score and needs golden-baseline regeneration — a scope decision the user has
    # DEFERRED (F21; charges frozen). So fa_elec is tracked here, not gated: a loose
    # drift bound catches unexpected regressions without failing on the known,
    # documented value. Remove this branch (fold fa_elec back into gated_channels) once
    # F21 is resolved.
    faelec_max = max(
        error for (channel, _), error in errors.items() if channel == "faelec"
    )
    print(f"  faelec CHARACTERIZATION (deferred, F21): {faelec_max:.6e}")
    assert faelec_max < FAELEC_CHARACTERIZATION_BOUND, (
        f"fa_elec drifted to {faelec_max:.6e} (> {FAELEC_CHARACTERIZATION_BOUND} "
        "characterization bound). This is the deferred charge-convention residual "
        "(F21); a larger value means an unexpected regression — investigate."
    )
