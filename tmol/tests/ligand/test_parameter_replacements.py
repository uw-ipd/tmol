"""Guarded bundles preserve coupled chemistry through the ordinary loader."""

from dataclasses import replace

import attr
import pytest
import torch
import yaml

from tmol.database import ParameterDatabase, inject_residue_params
from tmol.io import pose_stack_from_biotite
from tmol.ligand import load_params_file, prepare_ligands, write_params_file
from tmol.ligand._local_conjugate_params import (
    generate_conjugate_parameters,
    install_conjugate_parameters,
)
from tmol.ligand._registry import LigandPreparation, inject_ligand_preparations
from tmol.score import beta2016_score_function
from tmol.tests.ligand import test_conjugate_model

conjugate_input = test_conjugate_model.conjugate_input


@pytest.fixture(scope="module")
def bundle(conjugate_input, tmp_path_factory):
    _, array, _ = conjugate_input
    path = tmp_path_factory.mktemp("replacement") / "baseline.tmol"
    baseline, _ = prepare_ligands(array, seed=20250828, params_output=str(path))
    additions = load_params_file(path)
    result = generate_conjugate_parameters(array, baseline)
    corrections = [
        LigandPreparation(
            residue_type=r.residue_type,
            partial_charges=r.partial_charges,
            cartbonded_params=r.cartbonded_params,
            baseline_sha256=r.baseline_sha256,
            connection_params=result.connections if i == 0 else (),
        )
        for i, r in enumerate(result.residues)
    ]
    return array, baseline, additions, corrections, result


def _assert_same(actual, expected):
    # Ordinary additions retain insertion order; residue/type names define
    # identity. Patch order remains significant and is checked unchanged.
    assert attr.evolve(
        actual.chemical,
        residues=tuple(sorted(actual.chemical.residues, key=lambda r: r.name)),
        atom_types=tuple(sorted(actual.chemical.atom_types, key=lambda r: r.name)),
    ) == attr.evolve(
        expected.chemical,
        residues=tuple(sorted(expected.chemical.residues, key=lambda r: r.name)),
        atom_types=tuple(sorted(expected.chemical.atom_types, key=lambda r: r.name)),
    )
    assert actual.scoring.cartbonded == expected.scoring.cartbonded

    # Effective last-record precedence, independent of override history.
    def charge_index(db):
        return {
            (p.res, p.atom): p.charge for p in db.scoring.elec.atom_charge_parameters
        }

    assert charge_index(actual) == charge_index(expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_complete_bundle_fresh_prepared_and_repeated(bundle, tmp_path, reverse):
    array, baseline, additions, corrections, result = bundle
    preps = additions + corrections
    if reverse:
        preps = preps[::-1]
    path = tmp_path / "coupled.tmol"
    write_params_file(preps, path, format="tmol")
    assert yaml.safe_load(path.read_text())["version"] == "5.0"
    loaded = load_params_file(path)
    expected = install_conjugate_parameters(baseline, result)
    for database in (ParameterDatabase.get_default(), baseline, expected):
        restored = inject_ligand_preparations(database, loaded)
        _assert_same(restored, expected)
        assert inject_ligand_preparations(restored, loaded) is restored
        if database is expected:
            assert restored is expected
    # Exercise the public preparation entry point, including its early return.
    restored, _ = prepare_ligands(array, params_files=[str(path)])
    _assert_same(restored, expected)
    # In-memory combined preparations obey the same contract.
    _assert_same(inject_ligand_preparations(baseline, preps), expected)


def test_combined_bundle_does_not_hide_changed_baseline(bundle):
    _, baseline, additions, corrections, _ = bundle
    targets = {p.residue_type.name for p in corrections}
    name, charges = next(
        (name, q)
        for p in additions
        for name, q in (p.variant_partial_charges or {}).items()
        if name in targets
    )
    atom, charge = next(iter(charges.items()))
    changed = inject_residue_params(
        baseline, [], partial_charges={name: {atom: charge + 0.125}}
    )
    snapshot = changed.scoring.elec
    with pytest.raises(ValueError, match="baseline changed"):
        inject_ligand_preparations(changed, additions + corrections)
    assert changed.scoring.elec is snapshot


def test_explicit_baseline_bonded_records_survive_combined_export(bundle, tmp_path):
    _, baseline, additions, corrections, result = bundle
    records = baseline.scoring.cartbonded.residue_params
    old = {
        p.residue_type.name: records.get(
            p.residue_type.name, records[p.residue_type.base_name]
        )
        for p in corrections
    }
    additions = [
        replace(additions[0], additional_cartbonded_params=old),
        *additions[1:],
    ]
    path = tmp_path / "bonded.tmol"
    write_params_file(additions + corrections, path, format="tmol")
    loaded = load_params_file(path)
    assert loaded[0].additional_cartbonded_params == old
    expected = install_conjugate_parameters(baseline, result)
    actual = inject_ligand_preparations(ParameterDatabase.get_default(), loaded)
    _assert_same(actual, expected)
    assert inject_ligand_preparations(actual, loaded) is actual


def test_conflicting_connection_cannot_be_hidden_by_bundle(bundle):
    _, baseline, additions, corrections, result = bundle
    record = result.connections[0]
    length = record.length_parameters[0]
    incompatible = attr.evolve(
        record,
        length_parameters=(
            attr.evolve(length, K=length.K + 1),
            *record.length_parameters[1:],
        ),
    )
    changed = inject_residue_params(baseline, [], connection_params=(incompatible,))
    with pytest.raises(ValueError, match="Existing connection parameters differ"):
        inject_ligand_preparations(changed, additions + corrections)
    assert changed.scoring.cartbonded.connection_params == (incompatible,)


@pytest.mark.parametrize("damage", ["hash", "charge", "nan", "duplicate", "missing"])
def test_invalid_replacement_is_atomic(bundle, damage):
    _, baseline, _, corrections, _ = bundle
    row = corrections[0]
    if damage == "hash":
        records = [replace(row, baseline_sha256="not-a-digest")]
        error = "Invalid replacement baseline"
    elif damage in ("charge", "nan"):
        charges = dict(row.partial_charges)
        atom = next(iter(charges))
        if damage == "charge":
            del charges[atom]
        else:
            charges[atom] = float("nan")
        records = [replace(row, partial_charges=charges)]
        error = "complete finite atom charges"
    elif damage == "duplicate":
        records = [row, replace(row, baseline_sha256="0" * 64)]
        error = "Conflicting residue replacements"
    else:
        records = [
            replace(row, residue_type=attr.evolve(row.residue_type, name="ABSENT"))
        ]
        error = "Missing replacement baseline residue"
    snapshot = (baseline.chemical, baseline.scoring)
    with pytest.raises(ValueError, match=error):
        inject_ligand_preparations(baseline, records)
    assert baseline.chemical is snapshot[0]
    assert baseline.scoring is snapshot[1]


@pytest.mark.parametrize(
    "damage", ["legacy", "unknown", "hash", "duplicate", "rosetta", "bonded"]
)
def test_serialized_replacement_cannot_lose_guard(bundle, tmp_path, damage):
    _, _, _, corrections, _ = bundle
    path = tmp_path / "guard.tmol"
    if damage == "rosetta":
        with pytest.raises(ValueError, match="cannot preserve replacement baselines"):
            write_params_file(corrections, path, format="rosetta")
        assert not path.exists()
        return
    write_params_file(corrections, path, format="tmol")
    raw = yaml.safe_load(path.read_text())
    name = corrections[0].residue_type.name
    if damage == "legacy":
        raw["version"] = "3.0"
        error = "require .tmol format version 4"
    elif damage == "unknown":
        raw["chemical"]["replacement_baselines"]["ABSENT"] = "0" * 64
        error = "must name residues"
    elif damage == "hash":
        raw["chemical"]["replacement_baselines"][name] = None
        error = "Invalid replacement baseline"
    elif damage == "bonded":
        del raw["cartbonded"]["residue_params"][name]
        error = "explicit complete bonded record"
    else:
        raw["chemical"]["residues"].append(raw["chemical"]["residues"][0])
        error = "unique residue definitions"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=error):
        load_params_file(path)


def test_native_replacement_bundle_scores_and_builds_identically(
    bundle, tmp_path, torch_device, record_property
):
    array, baseline, additions, corrections, result = bundle
    path = tmp_path / "native.tmol"
    write_params_file(additions + corrections, path, format="tmol")
    actual, _ = prepare_ligands(array, params_files=[str(path)])
    expected = install_conjugate_parameters(baseline, result)
    _assert_same(actual, expected)
    values, gradients, poses = [], [], []
    repeat_values, repeat_gradients = [], []
    for db in (expected, actual):
        pose = pose_stack_from_biotite(array, torch_device, param_db=db, no_optH=True)
        poses.append(pose)
        module = beta2016_score_function(
            torch_device, param_db=db
        ).render_whole_pose_scoring_module(pose)
        coords = pose.coords.detach().clone().requires_grad_(True)
        score = module(coords, sum_terms=False, apply_weights=False)
        values.append(score.detach())
        gradients.append(torch.autograd.grad(score.sum(), coords)[0])
        repeat_coords = pose.coords.detach().clone().requires_grad_(True)
        repeat_score = module(repeat_coords, sum_terms=False, apply_weights=False)
        repeat_values.append(repeat_score.detach())
        repeat_gradients.append(
            torch.autograd.grad(repeat_score.sum(), repeat_coords)[0]
        )
    for field in (
        "coords",
        "inter_residue_connections",
        "block_coord_offset",
        "block_type_ind",
    ):
        torch.testing.assert_close(
            getattr(poses[0], field), getattr(poses[1], field), rtol=0, atol=0
        )
    for name, first, repeated in (
        ("score", values, repeat_values),
        ("gradient", gradients, repeat_gradients),
    ):
        record_property(name + "_max_abs", float((first[0] - first[1]).abs().max()))
        record_property(
            "repeat_" + name + "_max_abs",
            max(float((a - b).abs().max()) for a, b in zip(first, repeated)),
        )
        # CUDA reductions may accumulate contributions in a different order
        # even on an unchanged database; retain exact CPU/coordinate checks.
        tolerances = {"rtol": 0, "atol": 0} if torch_device.type == "cpu" else {}
        torch.testing.assert_close(first[0], first[1], **tolerances)
        for a, b in zip(first, repeated):
            torch.testing.assert_close(a, b, **tolerances)
