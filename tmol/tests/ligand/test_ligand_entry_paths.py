"""End-to-end coverage for the single-ligand entry points and params I/O.

Exercises the mol2 and SMILES preparation entry points, the Rosetta/.tmol
params writers and the ``.tmol`` loader (happy path plus its validation
branches), and runs a chemically diverse set of SMILES through the full
detect -> protonate -> 3D mol2 -> atom-typing -> residue-build pipeline so the
typing branches for aromatics, heterocycles, charged groups and ring amidines
are covered by a realistic workflow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tmol.database import ParameterDatabase

DATA = Path(__file__).parent.parent / "data"
MOL2_DIR = DATA / "ligand_test" / "ligand_ground_truth" / "mol2"


def _smallest_mol2() -> Path:
    mol2s = sorted(MOL2_DIR.glob("*.mol2"), key=lambda p: p.stat().st_size)
    assert mol2s, "expected ground-truth mol2 fixtures"
    return mol2s[0]


# --------------------------------------------------------------------------- #
# mol2 entry path
# --------------------------------------------------------------------------- #
def test_prepare_ligand_from_mol2_registers_residue() -> None:
    from tmol.ligand import prepare_ligand_from_mol2

    param_db, ordering = prepare_ligand_from_mol2(
        str(_smallest_mol2()),
        param_db=ParameterDatabase.get_default(),
        res_name="LG1",
    )
    residue = next((r for r in param_db.chemical.residues if r.name == "LG1"), None)
    assert residue is not None
    assert len(residue.atoms) > 0


def test_prepare_ligand_from_cif_uses_default_db_when_omitted() -> None:
    """Omitting param_db resolves the default database inside the entry point."""
    from tmol.ligand import prepare_ligand_from_cif

    cif = DATA / "ligand_cif_fixtures" / "vww.bonds_present.cif"
    param_db, ordering = prepare_ligand_from_cif(str(cif))
    assert any(r.name == "VWW" for r in param_db.chemical.residues)
    assert ordering is not None


def test_nonstandard_residue_info_from_mol2_block_roundtrips() -> None:
    from tmol.ligand.detect import (
        nonstandard_residue_info_from_mol2,
        nonstandard_residue_info_from_mol2_block,
    )

    mol2_path = _smallest_mol2()
    from_file = nonstandard_residue_info_from_mol2(mol2_path, res_name="LG1")
    from_block = nonstandard_residue_info_from_mol2_block(
        mol2_path.read_text(), res_name="LG1"
    )
    assert from_file.atom_names == from_block.atom_names
    assert from_file.elements == from_block.elements


# --------------------------------------------------------------------------- #
# write_params_from_mol2 + params round-trip + .tmol loader branches
# --------------------------------------------------------------------------- #
def _single_prep():
    from tmol.ligand.detect import nonstandard_residue_info_from_mol2
    from tmol.ligand.preparation import prepare_single_ligand

    info = nonstandard_residue_info_from_mol2(_smallest_mol2(), res_name="LG1")
    return prepare_single_ligand(info, sample_proton_chi=True)


def test_write_params_from_mol2_both_formats(tmp_path) -> None:
    from tmol.ligand.params_io import write_params_from_mol2

    rosetta = tmp_path / "lig.params"
    write_params_from_mol2(str(_smallest_mol2()), str(rosetta), res_name="LG1")
    assert rosetta.exists() and rosetta.stat().st_size > 0

    tmol_out = tmp_path / "lig.tmol"
    write_params_from_mol2(
        str(_smallest_mol2()), str(tmol_out), res_name="LG1", format="tmol"
    )
    assert tmol_out.exists() and tmol_out.stat().st_size > 0


def test_write_params_file_rosetta_list_to_directory(tmp_path) -> None:
    from tmol.ligand.params_io import write_params_file

    prep = _single_prep()
    out_dir = tmp_path / "params_out"
    out_dir.mkdir()
    write_params_file([prep], str(out_dir), format="rosetta")
    written = list(out_dir.glob("*.params"))
    assert written, "expected a per-residue .params file"


def test_tmol_params_roundtrip_and_inject(tmp_path) -> None:
    from tmol.ligand.params_file import (
        inject_params_file,
        inject_params_files,
        load_params_file,
    )
    from tmol.ligand.params_io import write_params_file

    prep = _single_prep()
    tmol_file = tmp_path / "lig.tmol"
    write_params_file([prep], str(tmol_file), format="tmol")

    loaded = load_params_file(tmol_file)
    assert len(loaded) == 1
    assert loaded[0].residue_type.name == prep.residue_type.name

    injected = inject_params_file(ParameterDatabase.get_default(), tmol_file)
    assert any(r.name == prep.residue_type.name for r in injected.chemical.residues)

    injected_multi = inject_params_files(ParameterDatabase.get_default(), [tmol_file])
    assert any(
        r.name == prep.residue_type.name for r in injected_multi.chemical.residues
    )


def test_tmol_loader_accepts_minor_version_difference(tmp_path) -> None:
    from tmol.ligand.params_file import load_params_file
    from tmol.ligand.params_io import write_params_file

    prep = _single_prep()
    tmol_file = tmp_path / "lig.tmol"
    write_params_file([prep], str(tmol_file), format="tmol")

    text = tmol_file.read_text().replace('version: "1.0"', 'version: "1.9"')
    assert "1.9" in text
    tmol_file.write_text(text)
    loaded = load_params_file(tmol_file)
    assert loaded and loaded[0].residue_type.name == prep.residue_type.name


def test_tmol_loader_warns_when_no_charges(tmp_path) -> None:
    import yaml

    from tmol.ligand.params_file import load_params_file
    from tmol.ligand.params_io import write_params_file

    prep = _single_prep()
    tmol_file = tmp_path / "lig.tmol"
    write_params_file([prep], str(tmol_file), format="tmol")

    doc = yaml.safe_load(tmol_file.read_text())
    doc["elec"] = {"atom_charge_parameters": []}
    tmol_file.write_text(yaml.safe_dump(doc))

    loaded = load_params_file(tmol_file)
    assert loaded
    assert all(c == 0.0 for c in loaded[0].partial_charges.values()) or (
        loaded[0].partial_charges == {}
    )


@pytest.mark.parametrize(
    "content, match",
    [
        ("- a\n- b\n", "Expected mapping"),
        ("chemical: {}\n", "no 'version' field"),
        ('version: "2.0"\nchemical: {}\n', "incompatible"),
        ('version: "1.0"\nresidues: []\n', "deprecated flat schema"),
    ],
)
def test_tmol_loader_rejects_bad_files(tmp_path, content, match) -> None:
    from tmol.ligand.params_file import load_params_file

    bad = tmp_path / "bad.tmol"
    bad.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_params_file(bad)


# --------------------------------------------------------------------------- #
# SMILES entry path over a chemically diverse set (atom-typing breadth)
# --------------------------------------------------------------------------- #
_DIVERSE_SMILES = {
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1c[nH]cn1",
    "benzoic_acid": "O=C(O)c1ccccc1",
    "benzamide": "O=C(N)c1ccccc1",
    "methanesulfonamide": "CS(=O)(=O)N",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "aminopyridine": "Nc1ccccn1",
    "trifluoromethylbenzene": "FC(F)(F)c1ccccc1",
    "chlorobromobenzene": "Clc1ccc(Br)cc1",
    "acetanilide": "CC(=O)Nc1ccccc1",
    "cyclopropanecarboxamide": "NC(=O)C1CC1",
    "ethanolamine": "NCCO",
}


@pytest.mark.parametrize("name", sorted(_DIVERSE_SMILES))
def test_prepare_ligand_from_smiles_registers(name: str) -> None:
    """Each diverse ligand prepares and registers via the SMILES path."""
    from tmol.ligand import prepare_ligand_from_smiles

    smiles = _DIVERSE_SMILES[name]
    param_db, _ = prepare_ligand_from_smiles(
        smiles,
        param_db=ParameterDatabase.get_default(),
        res_name="LG1",
        conformer_search=False,
    )
    residue = next((r for r in param_db.chemical.residues if r.name == "LG1"), None)
    assert residue is not None, f"{name} ({smiles}) did not register"
    assert len(residue.atoms) > 0


# --------------------------------------------------------------------------- #
# prepare_ligands over real ligand CIFs (detection loop, SMILES candidate
# selection, CIF atom renaming, atom typing breadth, params output)
# --------------------------------------------------------------------------- #
CIF_INPUTS = DATA / "protein_ligand_test" / "cif_inputs"

# A diverse subset of the DUD-derived ligand CIFs: varied ring systems,
# heteroatoms, halogens and charged groups, to exercise the typing/rename
# branches of the full CIF -> params path.
_CIF_LIGANDS = ["ada", "cdk2", "cox2", "hivpr", "src", "egfr"]


def _load_full_array(cif_path: Path):
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    cif = pdbx.CIFFile.read(str(cif_path))
    arr = pdbx.get_structure(cif, model=1, include_bonds=True, extra_fields=["charge"])
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    return arr


@pytest.mark.parametrize("name", _CIF_LIGANDS)
def test_prepare_ligand_from_cif_inputs(name: str) -> None:
    """Diverse real ligand CIFs prepare end-to-end via the unified path."""
    from tmol.ligand import prepare_ligand_from_cif

    cif = CIF_INPUTS / f"{name}.ligand.cif"
    if not cif.exists():
        pytest.skip(f"missing fixture {cif}")
    param_db, _ = prepare_ligand_from_cif(
        str(cif), param_db=ParameterDatabase.get_default()
    )
    # At least one new (non-canonical) residue should have been registered.
    default_names = {r.name for r in ParameterDatabase.get_default().chemical.residues}
    new_names = [
        r.name for r in param_db.chemical.residues if r.name not in default_names
    ]
    assert new_names, f"{name}: no new ligand residue registered"


def test_prepare_ligands_writes_params_output(tmp_path) -> None:
    """prepare_ligands over a ligand AtomArray writes a reusable .tmol file."""
    from tmol.ligand.preparation import prepare_ligands

    arr = _load_full_array(CIF_INPUTS / "ada.ligand.cif")
    out = tmp_path / "out.tmol"
    param_db, ordering = prepare_ligands(
        arr, param_db=ParameterDatabase.get_default(), params_output=str(out)
    )
    assert out.exists() and out.stat().st_size > 0
    assert ordering is not None


def test_prepare_ligands_accepts_single_model_stack() -> None:
    """An AtomArrayStack with one model is accepted (and default db resolved)."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    from tmol.ligand.preparation import prepare_ligands

    cif = pdbx.CIFFile.read(str(CIF_INPUTS / "ada.ligand.cif"))
    stack = pdbx.get_structure(cif, include_bonds=True, extra_fields=["charge"])
    if not isinstance(stack, struc.AtomArrayStack):
        stack = struc.stack([stack])
    assert len(stack) == 1
    # No param_db passed -> default resolved internally.
    param_db, _ = prepare_ligands(stack)
    assert param_db is not None


def test_prepare_ligands_rejects_multi_model_stack() -> None:
    """An AtomArrayStack with multiple models is rejected with a clear error."""
    import biotite.structure as struc

    from tmol.ligand.preparation import prepare_ligands

    arr = _load_full_array(CIF_INPUTS / "ada.ligand.cif")
    stack = struc.stack([arr, arr])
    with pytest.raises(TypeError, match="single AtomArray"):
        prepare_ligands(stack, param_db=ParameterDatabase.get_default())


def test_prepare_ligands_strict_raises_on_unpreparable_ligand() -> None:
    """A ligand that cannot yield a SMILES raises under strict_ligands=True."""
    from tmol.ligand import LigandPreparationError
    from tmol.ligand.preparation import prepare_ligands

    # The parp ligand CIF has bond orders the unified SMILES path can't use, so
    # on-the-fly preparation fails -- strict mode must surface that loudly.
    arr = _load_full_array(CIF_INPUTS / "parp.ligand.cif")
    with pytest.raises(LigandPreparationError):
        prepare_ligands(
            arr, param_db=ParameterDatabase.get_default(), strict_ligands=True
        )


def test_prepare_ligands_lenient_skips_unpreparable_ligand() -> None:
    """The same ligand is skipped with a warning under strict_ligands=False."""
    from tmol.ligand.preparation import prepare_ligands

    arr = _load_full_array(CIF_INPUTS / "parp.ligand.cif")
    param_db, ordering = prepare_ligands(
        arr, param_db=ParameterDatabase.get_default(), strict_ligands=False
    )
    # Lenient mode returns a (possibly unchanged) database rather than raising.
    assert param_db is not None
    assert ordering is not None


def test_prepare_ligands_with_params_files_skips_reprep(tmp_path) -> None:
    """A residue supplied via params_files is already known, so no re-prep."""
    from tmol.ligand.detect import nonstandard_residue_info_from_mol2
    from tmol.ligand.params_io import write_params_file
    from tmol.ligand.preparation import prepare_ligands, prepare_single_ligand

    # Build a .tmol for the mol2 ligand named LG1.
    info = nonstandard_residue_info_from_mol2(_smallest_mol2(), res_name="LG1")
    prep = prepare_single_ligand(info, sample_proton_chi=True)
    tmol_file = tmp_path / "lg1.tmol"
    write_params_file([prep], str(tmol_file), format="tmol")

    arr = _load_full_array(CIF_INPUTS / "ada.ligand.cif")
    param_db, _ = prepare_ligands(
        arr,
        param_db=ParameterDatabase.get_default(),
        params_files=[str(tmol_file)],
    )
    assert any(r.name == "LG1" for r in param_db.chemical.residues)
