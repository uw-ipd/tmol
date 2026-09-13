"""Declared atom-type elements must survive reusable parameter bundles."""

from dataclasses import replace
from types import SimpleNamespace

import attr
import pytest
import yaml

from tmol.database import ParameterDatabase
from tmol.ligand import collect_new_atom_types, load_params_file, write_params_file
from tmol.ligand._registry import inject_ligand_preparations
from tmol.tests.ligand.test_ligand_entry_paths import _single_prep


@pytest.fixture(scope="module")
def prep():
    return _single_prep()


@pytest.mark.parametrize("element", ["N", "O", "S", "P", "H", "Cl", "Br", "Vr"])
def test_declared_element_survives_export_without_name_inference(
    prep, tmp_path, element
):
    name = "UnfamiliarType"
    atom = prep.residue_type.atoms[0]
    custom = replace(
        prep,
        residue_type=attr.evolve(
            prep.residue_type,
            atoms=(attr.evolve(atom, atom_type=name), *prep.residue_type.atoms[1:]),
        ),
        atom_type_elements={name: element},
    )
    path = tmp_path / "elements.tmol"
    write_params_file(custom, path, format="tmol")
    restored = load_params_file(path)[0]
    assert restored.atom_type_elements == custom.atom_type_elements
    # This checks element registration only. No arbitrary new scoring rows are
    # implied, and the deliberately relabeled topology is not scored.
    types = collect_new_atom_types(
        ParameterDatabase.get_default().chemical,
        restored.residue_type,
        restored.atom_type_elements,
        strict_atom_types=True,
    )
    assert next(t.element for t in types if t.name == name) == element


def test_real_preparation_maps_roundtrip_and_strict_injection(prep, tmp_path):
    path = tmp_path / "ordinary.tmol"
    write_params_file(prep, path, format="tmol")
    restored = load_params_file(path)
    assert restored[0].atom_type_elements == prep.atom_type_elements
    assert yaml.safe_load(path.read_text())["version"] == "5.0"
    base = ParameterDatabase.get_default()
    expected = inject_ligand_preparations(base, [prep], strict_atom_types=True)
    actual = inject_ligand_preparations(base, restored, strict_atom_types=True)
    assert actual.chemical == expected.chemical
    assert actual.scoring.elec == expected.scoring.elec
    assert actual.scoring.cartbonded == expected.scoring.cartbonded


def test_element_declaration_cannot_disagree_with_existing_type(prep):
    base = ParameterDatabase.get_default()
    atom = prep.residue_type.atoms[0]
    known = {t.name: t.element for t in base.chemical.atom_types}
    bad = replace(
        prep,
        atom_type_elements={
            atom.atom_type: "N" if known[atom.atom_type] != "N" else "C"
        },
    )
    for database in (base, inject_ligand_preparations(base, [prep])):
        with pytest.raises(ValueError, match="element disagrees with database"):
            inject_ligand_preparations(database, [bad])


@pytest.mark.parametrize(
    "mapping", [{"X": None}, {"X": ""}, {"": "N"}, {"X": 7}, ["N"]]
)
def test_invalid_element_metadata_is_rejected(prep, tmp_path, mapping):
    path = tmp_path / "invalid.tmol"
    write_params_file(replace(prep, atom_type_elements=None), path, format="tmol")
    raw = yaml.safe_load(path.read_text())
    raw["version"] = "5.0"
    raw["chemical"]["atom_type_elements"] = mapping
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="atom_type_elements"):
        load_params_file(path)


def test_old_version_cannot_silently_drop_declared_elements(prep, tmp_path):
    path = tmp_path / "old.tmol"
    write_params_file(prep, path, format="tmol")
    raw = yaml.safe_load(path.read_text())
    raw["version"] = "4.0"
    raw["chemical"]["atom_type_elements"] = {"UnfamiliarType": "N"}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(
        ValueError, match="atom_type_elements require .tmol format version 5"
    ):
        load_params_file(path)


def test_legacy_missing_mapping_keeps_strict_mode_error():
    residue = SimpleNamespace(
        name="LEGACY", atoms=(SimpleNamespace(name="N1", atom_type="UnfamiliarType"),)
    )
    with pytest.raises(ValueError, match="Unknown element mapping"):
        collect_new_atom_types(
            ParameterDatabase.get_default().chemical, residue, strict_atom_types=True
        )


def test_patch_introduced_atom_types_use_declared_elements(prep, tmp_path):
    from tmol.database.chemical import VariantScope

    base = ParameterDatabase.get_default()
    template = next(v for v in base.chemical.variants if v.name == "CarboxyTerminus")
    patch = attr.evolve(
        template,
        name="ElementTerminal",
        display_name="elemterm",
        applies_to=VariantScope(base_names=("ALA",)),
        add_atoms=tuple(
            attr.evolve(a, atom_type="UnfamiliarO") for a in template.add_atoms
        ),
    )
    custom = replace(
        prep,
        adds_patches=(patch,),
        atom_type_elements={**prep.atom_type_elements, "UnfamiliarO": "O"},
    )
    path = tmp_path / "patch.tmol"
    write_params_file(custom, path, format="tmol")
    for sources in ([custom], load_params_file(path)):
        database = inject_ligand_preparations(base, sources, strict_atom_types=True)
        assert (
            next(
                t.element
                for t in database.chemical.atom_types
                if t.name == "UnfamiliarO"
            )
            == "O"
        )
        variant = next(
            r for r in database.chemical.residues if r.name == "ALA:elemterm"
        )
        assert (
            next(a.atom_type for a in variant.atoms if a.name == "OXT") == "UnfamiliarO"
        )
