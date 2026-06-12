#!/usr/bin/env python
"""Run tmol ligand-prep over the ligand-test set, save outputs, and check parity.

For every molecule in the ground-truth manifest:
  * SMILES path  — generate a tmol mol2 (OpenBabel), Rosetta .params, and tmol
    .tmol, and SAVE all three under ligand_tmol_generated/ for manual checks.
  * compare the SMILES-path prep AND the mol2-direct prep to the Rosetta
    ground-truth .params (name-agnostic, via the heavy-atom graph isomorphism).

Layout (default):
  tests/data/ligand_test/ligand_ground_truth/   <- Rosetta ground truth (manifest)
  tests/data/ligand_test/ligand_tmol_generated/ <- tmol mol2/ params/ tmol/  (written here)

"chemistry" = atoms / types / bonds / charges (cartbonded skipped). "full" also
requires CHI axes to match (the CHI/PROTON_CHI port is partial, so full < chem).

Usage:  python validate_dud80.py [manifest.json]
Exit code is non-zero if any molecule fails the chemistry check on either path.
"""
import sys
from pathlib import Path

from tmol.ligand import params_io, prepare_single_ligand
from tmol.ligand.detect import (
    _normalize_radical_oxygens,
    nonstandard_residue_info_from_mol2,
)
from tmol.ligand.openbabel_compat import obabel_smiles_to_mol2
from tmol.ligand.params_reference import compare_semantic, parse_reference_params
from tmol.ligand.parity_manifest import default_dataset_manifest, load_parity_manifest
from tmol.tests.ligand._parity_helpers import (
    chi_axes_equivalent,
    reference_view_from_params,
)

SKIP = frozenset({"cartbonded_params"})


def _evaluate(prep, ref):
    view = reference_view_from_params(ref)
    res = compare_semantic(prep, view, charge_tolerance=0.05, skip_checks=SKIP)
    chi_ok = chi_axes_equivalent(prep, ref, view=view)
    details = {k: v for k, v in res.details.items() if v != "skipped"}
    return res.is_equivalent, chi_ok, details


def main() -> int:
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dataset_manifest()
    entries = load_parity_manifest(manifest)
    gen_root = manifest.parent.parent / "ligand_tmol_generated"
    mol2_dir = gen_root / "mol2"
    params_dir = gen_root / "params"
    tmol_dir = gen_root / "tmol"
    for d in (mol2_dir, params_dir, tmol_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"loaded {len(entries)} entries from {manifest}")
    print(f"saving tmol-generated files under {gen_root}")

    a_chem = a_full = b_chem = b_full = 0
    a_fail, b_fail = [], []
    for e in entries:
        ref = parse_reference_params(e.params)
        # --- (A) SMILES -> mol2 (generate + SAVE mol2 / params / tmol) ---
        try:
            smi = _normalize_radical_oxygens(e.expected_prot_smiles)  # no-op if clean
            tmol_mol2 = mol2_dir / f"{e.name}.mol2"
            obabel_smiles_to_mol2(smi, tmol_mol2)
            info = nonstandard_residue_info_from_mol2(tmol_mol2, res_name=e.name)
            prep = prepare_single_ligand(
                info, charge_mode=e.charge_mode, sample_proton_chi=e.sample_proton_chi
            )
            params_io.write_params_file(
                prep, params_dir / f"{e.name}.params", format="rosetta"
            )
            params_io.write_params_file(
                prep, tmol_dir / f"{e.name}.tmol", format="tmol"
            )
            chem_ok, chi_ok, details = _evaluate(prep, ref)
            a_chem += chem_ok
            a_full += chem_ok and chi_ok
            if not chem_ok:
                a_fail.append((e.name, details))
        except Exception as exc:  # noqa: BLE001
            a_fail.append((e.name, f"EXC {type(exc).__name__}: {exc}"[:90]))
        # --- (B) mol2 direct (read the ground-truth mol2) ---
        try:
            info = nonstandard_residue_info_from_mol2(e.mol2, res_name=e.name)
            prep = prepare_single_ligand(
                info, charge_mode=e.charge_mode, sample_proton_chi=e.sample_proton_chi
            )
            chem_ok, chi_ok, details = _evaluate(prep, ref)
            b_chem += chem_ok
            b_full += chem_ok and chi_ok
            if not chem_ok:
                b_fail.append((e.name, details))
        except Exception as exc:  # noqa: BLE001
            b_fail.append((e.name, f"EXC {type(exc).__name__}: {exc}"[:90]))

    n = len(entries)
    print("\n### (A) SMILES -> mol2 (tmol-generated, saved)")
    print(f"  chemistry: {a_chem}/{n}   full (+CHI): {a_full}/{n}")
    for name, info in a_fail:
        print(f"    FAIL {name}: {info}")
    print("\n### (B) mol2 direct")
    print(f"  chemistry: {b_chem}/{n}   full (+CHI): {b_full}/{n}")
    for name, info in b_fail:
        print(f"    FAIL {name}: {info}")
    return 0 if (a_chem == n and b_chem == n) else 1


if __name__ == "__main__":
    sys.exit(main())
