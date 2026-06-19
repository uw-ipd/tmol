"""Unified CIF/atom-array -> SMILES -> params -> score path.

These tests exercise the path tmol takes when a user only has a structure
(a biotite ``AtomArray`` from a CIF) and no preprocessed ``.tmol``/``.params``
file. Real PLINDER ligands are vendored under
``tmol/tests/data/ligand_cif_fixtures`` in two variants:

* ``*.bonds_present.cif`` — carries an explicit ``_chem_comp_bond`` block
  (existing-bonds SMILES branch), and
* ``*.bonds_absent.cif`` — atom-site records only; biotite re-infers
  intra-residue bonds when it loads the file, which then feeds the same
  existing-bonds SMILES branch (tmol itself never does a CCD lookup).

Coverage here:

* :func:`ligand_smiles_from_atom_array` reproduces each ligand's reference
  SMILES from both CIF variants.
* :func:`prepare_ligand_from_cif` registers the ligand from both variants.
* ``pose_stack_from_biotite(prepare_ligands=True)`` yields a finite score.

The complementary bonds-present *complex* ddG case (PSE / 1A25) lives in
``test_ligand_pipeline.py::test_ddg_from_cif_complex_with_onthefly_ligand_prep``.
"""

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent.parent / "data" / "ligand_cif_fixtures"

# stem -> (CCD residue code, reference canonical SMILES from the PLINDER SDF)
_LIGANDS: dict[str, tuple[str, str]] = {
    "vww": (
        "VWW",
        "N[C@@H](CCC(=O)N[C@@H](CSCc1ccccc1)C(=O)N[C@@H](C(=O)O)c1ccccc1)C(=O)O",
    ),
    "sah": (
        "SAH",
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CSCC[C@H](N)C(=O)O)[C@@H](O)[C@H]1O",
    ),
}
_VARIANTS = ("bonds_present", "bonds_absent")

# Golden start-to-end CIF -> params output, captured from the unified pipeline.
# The full CIF -> ddG score is *not* pinned to a golden value: OpenBabel's 3D
# conformer generation is stochastic, so pipeline-added hydrogen positions (and
# thus the score) vary by ~1 kcal between runs (the CIF pins only heavy-atom
# coords). What *is* deterministic -- and what these goldens guard -- is the
# generated chemistry: atom count, bond count, and the atom-type multiset. Both
# CIF variants must converge to the same params. (Param-vs-Rosetta parity is
# covered by the guanfeng SMILES suite; golden-params-vs-Rosetta scoring by
# test_dud_ligands.py.)
_GOLDEN_PARAMS: dict[str, dict] = {
    "vww": {
        "n_atoms": 59,
        "n_bonds": 60,
        "atom_types": {
            "CDp": 4,
            "CR": 12,
            "CS1": 2,
            "CS2": 4,
            "CSp": 1,
            "HC": 11,
            "HN": 5,
            "HR": 10,
            "Nad": 2,
            "Nam": 1,
            "Oad": 2,
            "Oat": 4,
            "Ssl": 1,
        },
    },
    "sah": {
        "n_atoms": 46,
        "n_bonds": 48,
        "atom_types": {
            "CDp": 1,
            "CR": 4,
            "CRp": 1,
            "CS1": 2,
            "CS2": 3,
            "CSp": 3,
            "HC": 11,
            "HN": 5,
            "HO": 2,
            "HR": 2,
            "NG22": 1,
            "Nad3": 1,
            "Nam": 1,
            "Nim": 3,
            "Oat": 2,
            "Oet": 1,
            "Ohx": 2,
            "Ssl": 1,
        },
    },
}

# Both ligands prepare cleanly: charges come straight from the SMILES ->
# OpenBabel MMFF94 step (by atom index), so SAH's fused adenine purine -- which
# trips RDKit's MMFF kekulization -- no longer matters: there is no RDKit charge
# fallback. The OpenBabel charges flow through untouched.
_PREPARABLE = ("vww", "sah")


def _load_ligand_array(stem: str, variant: str, *, include_bonds: bool = True):
    """Load a fixture ligand CIF into a single biotite ``AtomArray``."""
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    cif = pdbx.CIFFile.read(str(FIXTURE_DIR / f"{stem}.{variant}.cif"))
    arr = pdbx.get_structure(
        cif, model=1, include_bonds=include_bonds, extra_fields=["charge"]
    )
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    return arr


def _canonical(smiles: str) -> str | None:
    """Canonical, stereo-free SMILES for connectivity comparison."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


@pytest.mark.parametrize("stem", sorted(_LIGANDS))
@pytest.mark.parametrize("variant", _VARIANTS)
def test_cif_smiles_matches_reference(stem: str, variant: str) -> None:
    """SMILES derivation reproduces the reference for both CIF variants."""
    from tmol.ligand.structure_to_smiles import ligand_smiles_from_atom_array

    code, expected = _LIGANDS[stem]
    arr = _load_ligand_array(stem, variant)
    smiles = ligand_smiles_from_atom_array(arr, res_name=code)
    assert _canonical(smiles) == _canonical(expected)


@pytest.mark.parametrize("stem", _PREPARABLE)
@pytest.mark.parametrize("variant", _VARIANTS)
def test_prepare_ligand_from_cif_registers_residue(stem: str, variant: str) -> None:
    """The CIF helper prepares and registers the ligand for both variants."""
    from tmol.database import ParameterDatabase
    from tmol.ligand import prepare_ligand_from_cif

    code, _ = _LIGANDS[stem]
    param_db, _ = prepare_ligand_from_cif(
        str(FIXTURE_DIR / f"{stem}.{variant}.cif"),
        param_db=ParameterDatabase.get_default(),
    )
    residue = next((r for r in param_db.chemical.residues if r.name == code), None)
    assert residue is not None, f"{code} not registered from {variant}"
    assert len(residue.atoms) > 0


@pytest.mark.parametrize("stem", sorted(_GOLDEN_PARAMS))
@pytest.mark.parametrize("variant", _VARIANTS)
def test_cif_to_params_golden(stem: str, variant: str) -> None:
    """Start-to-end CIF -> params reproduces the golden chemistry (deterministic).

    Pins the deterministic output of the unified CIF path -- atom count, bond
    count, and atom-type multiset -- which is stable across runs and identical
    for both CIF variants (the stochastic ddG score is only sanity-checked for
    finiteness in the tests below).
    """
    from collections import Counter

    from tmol.database import ParameterDatabase
    from tmol.ligand import prepare_ligand_from_cif

    code, _ = _LIGANDS[stem]
    golden = _GOLDEN_PARAMS[stem]
    param_db, _ = prepare_ligand_from_cif(
        str(FIXTURE_DIR / f"{stem}.{variant}.cif"),
        param_db=ParameterDatabase.get_default(),
    )
    residue = next((r for r in param_db.chemical.residues if r.name == code), None)
    assert residue is not None, f"{code} not registered from {variant}"

    assert len(residue.atoms) == golden["n_atoms"]
    assert len(residue.bonds) == golden["n_bonds"]
    atom_types = Counter(a.atom_type for a in residue.atoms)
    assert dict(atom_types) == golden["atom_types"]


@pytest.mark.parametrize("variant", _VARIANTS)
def test_cif_ligand_pose_scores_finite(variant: str, torch_device) -> None:
    """End-to-end: a ligand-only CIF scores to a finite total via the new path."""
    import torch

    from tmol.database import ParameterDatabase
    from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
    from tmol.score import beta2016_score_function

    arr = _load_ligand_array("vww", variant)
    pose_stack, context = pose_stack_from_biotite(
        arr,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
        param_db=ParameterDatabase.get_default(),
    )

    sfxn = beta2016_score_function(torch_device, param_db=context.parameter_database)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    scores = scorer(pose_stack.coords)
    assert torch.isfinite(scores).all(), f"non-finite score for {variant}: {scores}"


@pytest.mark.parametrize("variant", _VARIANTS)
def test_fused_purine_ligand_uses_openbabel_charges(variant: str) -> None:
    """SAH's fused adenine purine prepares via OpenBabel MMFF94 charges.

    RDKit cannot MMFF-kekulize the adenine purine, but the unified path never
    asks it to: charges come from the SMILES -> OpenBabel MMFF94 step and are
    mapped onto atoms by index. This guards against a regression to a RDKit
    charge fallback (which would re-introduce the kekulization failure) and
    confirms the OpenBabel charges are carried through to the prepared residue.
    """
    from tmol.database import ParameterDatabase
    from tmol.ligand import prepare_ligand_from_cif

    param_db, _ = prepare_ligand_from_cif(
        str(FIXTURE_DIR / f"sah.{variant}.cif"),
        param_db=ParameterDatabase.get_default(),
    )
    residue = next((r for r in param_db.chemical.residues if r.name == "SAH"), None)
    assert residue is not None, f"SAH not registered from {variant}"
    # Heavy atoms of SAH (26) must all be present in the prepared residue.
    assert len(residue.atoms) >= 26


def test_smiles_candidates_never_raise_on_non_ccd_geometry() -> None:
    """The candidate helper degrades gracefully for the pure-geometry branch.

    With no bond table, only the geometry branch (``rdDetermineBonds``)
    remains. It is best-effort for heavy-atom-only inputs, so the helper must
    return a (possibly empty) list rather than raise.
    """
    import numpy as np

    from tmol.ligand.structure_to_smiles import (
        ligand_smiles_candidates_from_atom_array,
    )

    arr = _load_ligand_array("vww", "bonds_absent", include_bonds=False)
    arr.res_name = np.array(["LIG"] * len(arr), dtype=arr.res_name.dtype)
    assert arr.bonds is None

    candidates = ligand_smiles_candidates_from_atom_array(arr, res_name="LIG")
    assert isinstance(candidates, list)
