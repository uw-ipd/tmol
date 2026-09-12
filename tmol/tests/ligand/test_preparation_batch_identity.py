"""Multiple parameter sources must agree on each named chemical identity."""

from dataclasses import replace

import attr
import pytest
import yaml

from tmol.database import ParameterDatabase
from tmol.ligand import inject_params_files, load_params_file, write_params_file
from tmol.ligand._registry import inject_ligand_preparations, rebuild_canonical_ordering
from tmol.tests.ligand.test_ligand_entry_paths import _single_prep


@pytest.fixture(scope="module")
def prep():
    return _single_prep()


def _assert_same(actual, expected):
    assert actual.chemical == expected.chemical
    assert actual.scoring.cartbonded == expected.scoring.cartbonded
    left = actual.scoring.elec.atom_charge_parameters
    right = expected.scoring.elec.atom_charge_parameters
    assert len(left) == len(right)
    assert {(p.res, p.atom): p.charge for p in left} == {
        (p.res, p.atom): p.charge for p in right
    }
    assert len({r.name for r in actual.chemical.residues}) == len(
        actual.chemical.residues
    )
    rebuild_canonical_ordering(actual)


@pytest.mark.parametrize("copies", [2, 20])
def test_repeated_new_preparation_has_one_identity(prep, copies):
    base = ParameterDatabase.get_default()
    expected = inject_ligand_preparations(base, [prep])
    actual = inject_ligand_preparations(base, [replace(prep) for _ in range(copies)])
    _assert_same(actual, expected)
    assert inject_ligand_preparations(actual, [prep] * copies) is actual


def test_repeated_files_and_repeated_export_have_one_identity(prep, tmp_path):
    base = ParameterDatabase.get_default()
    one, many = tmp_path / "one.tmol", tmp_path / "many.tmol"
    write_params_file(prep, one, format="tmol")
    write_params_file([prep, replace(prep)], many, format="tmol")
    assert len(load_params_file(many)) == 1
    expected = inject_params_files(base, [one])
    actual = inject_params_files(base, [one, many, one])
    _assert_same(actual, expected)
    assert inject_params_files(actual, [one, many]) is actual


@pytest.mark.parametrize("field", ["residue", "charges", "bonded"])
@pytest.mark.parametrize("preinstalled", [False, True])
def test_conflicting_batch_definitions_fail_before_injection(prep, field, preinstalled):
    if field == "residue":
        atom = prep.residue_type.atoms[0]
        changed = replace(
            prep,
            residue_type=attr.evolve(
                prep.residue_type,
                atoms=(
                    attr.evolve(
                        atom, atom_type="CH3" if atom.atom_type != "CH3" else "CH2"
                    ),
                    *prep.residue_type.atoms[1:],
                ),
            ),
        )
    elif field == "charges":
        charges = dict(prep.partial_charges)
        charges[next(iter(charges))] += 0.125
        changed = replace(prep, partial_charges=charges)
    else:
        bonded = prep.cartbonded_params
        row = bonded.length_parameters[0]
        changed = replace(
            prep,
            cartbonded_params=attr.evolve(
                bonded,
                length_parameters=(
                    attr.evolve(row, K=row.K + 1),
                    *bonded.length_parameters[1:],
                ),
            ),
        )
    base = ParameterDatabase.get_default()
    if preinstalled:
        base = inject_ligand_preparations(base, [prep])
    snapshot = base.chemical
    for order in ([prep, changed], [changed, prep]):
        with pytest.raises(ValueError, match="Conflicting residue preparations"):
            inject_ligand_preparations(base, order)
    assert base.chemical is snapshot


def test_disjoint_shared_charges_on_duplicate_preps_are_preserved(prep, tmp_path):
    first = replace(prep, variant_partial_charges={"ALA": {"CA": 0.125}})
    second = replace(prep, variant_partial_charges={"ALA": {"N": -0.25}})
    merged = replace(prep, variant_partial_charges={"ALA": {"CA": 0.125, "N": -0.25}})
    base = ParameterDatabase.get_default()
    expected = inject_ligand_preparations(base, [merged])
    for order in ([first, second], [second, first]):
        _assert_same(inject_ligand_preparations(base, order), expected)
        path = tmp_path / "shared.tmol"
        write_params_file(order, path, format="tmol")
        _assert_same(inject_params_files(base, [path]), expected)


@pytest.mark.parametrize("metadata", ["charges", "bonded"])
def test_installed_replacement_cannot_hide_conflicting_batch_metadata(prep, metadata):
    from tmol.ligand._parameter_replacements import _local_identity

    base = inject_ligand_preparations(ParameterDatabase.get_default(), [prep])
    name = prep.residue_type.name
    rt = next(r for r in base.chemical.residues if r.name == name)
    charges = dict(prep.partial_charges)
    atom = next(iter(charges))
    charges[atom] += 0.125
    correction = replace(
        prep,
        partial_charges=charges,
        baseline_sha256=_local_identity(
            rt, prep.partial_charges, prep.cartbonded_params
        ),
    )
    corrected = inject_ligand_preparations(base, [correction])
    if metadata == "charges":
        first = replace(prep, variant_partial_charges={name: prep.partial_charges})
        second = replace(prep, variant_partial_charges={name: charges})
        error = "Conflicting partial charges"
    else:
        old = prep.cartbonded_params
        row = old.length_parameters[0]
        changed = attr.evolve(
            old,
            length_parameters=(
                attr.evolve(row, K=row.K + 1),
                *old.length_parameters[1:],
            ),
        )
        first = replace(prep, additional_cartbonded_params={name: old})
        second = replace(prep, additional_cartbonded_params={name: changed})
        error = "Conflicting bonded parameters"
    for database in (base, corrected):
        with pytest.raises(ValueError, match=error):
            inject_ligand_preparations(database, [first, second, correction])


@pytest.mark.parametrize("preinstalled", [False, True])
def test_conflicting_shared_charges_fail_even_when_one_matches_database(
    prep, preinstalled
):
    base = ParameterDatabase.get_default()
    if preinstalled:
        base = inject_ligand_preparations(base, [prep])
    old = {(p.res, p.atom): p.charge for p in base.scoring.elec.atom_charge_parameters}
    first = replace(prep, variant_partial_charges={"ALA": {"CA": old["ALA", "CA"]}})
    second = replace(
        prep, variant_partial_charges={"ALA": {"CA": old["ALA", "CA"] + 0.125}}
    )
    for order in ([first, second], [second, first]):
        with pytest.raises(ValueError, match="Conflicting partial charges"):
            inject_ligand_preparations(base, order)


def test_duplicate_yaml_charge_rows_cannot_hide_a_conflict(prep, tmp_path):
    path = tmp_path / "charges.tmol"
    write_params_file(prep, path, format="tmol")
    raw = yaml.safe_load(path.read_text())
    charges = raw["elec"]["atom_charge_parameters"]
    charges.append({**charges[0], "charge": charges[0]["charge"] + 0.125})
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Conflicting partial charges"):
        load_params_file(path)


def test_element_maps_merge_across_duplicate_sources(prep):
    atom = next(a for a in prep.residue_type.atoms if a.atom_type.startswith("C"))
    rt = attr.evolve(
        prep.residue_type,
        atoms=tuple(
            attr.evolve(a, atom_type="BatchCarbon") if a.name == atom.name else a
            for a in prep.residue_type.atoms
        ),
    )
    first = replace(prep, residue_type=rt, atom_type_elements=None)
    second = replace(first, atom_type_elements={"BatchCarbon": "C"})
    expected = inject_ligand_preparations(
        ParameterDatabase.get_default(), [second], strict_atom_types=True
    )
    for order in ([first, second], [second, first]):
        actual = inject_ligand_preparations(
            ParameterDatabase.get_default(), order, strict_atom_types=True
        )
        _assert_same(actual, expected)


def test_conflicting_atom_type_elements_are_rejected(prep):
    atom = next(a for a in prep.residue_type.atoms if a.atom_type.startswith("C"))
    rt = attr.evolve(
        prep.residue_type,
        atoms=tuple(
            attr.evolve(a, atom_type="BatchType") if a.name == atom.name else a
            for a in prep.residue_type.atoms
        ),
    )
    first = replace(prep, residue_type=rt, atom_type_elements={"BatchType": "C"})
    second = replace(first, atom_type_elements={"BatchType": "N"})
    for order in ([first, second], [second, first]):
        with pytest.raises(ValueError, match="Conflicting atom type elements"):
            inject_ligand_preparations(ParameterDatabase.get_default(), order)


def test_merging_shared_charges_does_not_mutate_reusable_preparations(prep):
    from copy import deepcopy

    first = replace(prep, variant_partial_charges={"ALA": {"CA": 0.125}})
    second = replace(prep, variant_partial_charges={"ALA": {"N": -0.25}})
    expected = deepcopy([first, second])
    inject_ligand_preparations(ParameterDatabase.get_default(), [first, second])
    assert [first, second] == expected


def test_each_path_is_read_once_per_batch(prep, tmp_path, monkeypatch):
    from tmol.ligand import _params_file

    path = tmp_path / "one.tmol"
    write_params_file(prep, path, format="tmol")
    original = _params_file.load_params_file
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(_params_file, "load_params_file", counted)
    inject_params_files(ParameterDatabase.get_default(), [str(path), path, str(path)])
    assert calls == [path]
    # This is deliberately a batch-local optimization, not a stale file cache.
    inject_params_files(ParameterDatabase.get_default(), [path])
    assert calls == [path, path]
