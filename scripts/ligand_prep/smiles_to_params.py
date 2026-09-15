#!/usr/bin/env python
"""Run tmol's SMILES -> tmol .tmol pipeline manually.

Pipeline (all in tmol.ligand):
  1. nonstandard_residue_info_from_smiles_via_mol2(smiles)
       normalize [O]->[O-]  ->  Dimorphite protonate (pH)  ->  OpenBabel 3D +
       MMFF94 mol2  ->  read the mol2 (names/coords/charges/bonds preserved).
  2. prepare_single_ligand(info)  ->  LigandPreparation
       (residue_type + partial_charges + cartbonded_params).
  3. params_io.write_params_file(prep, path)  ->  tmol .tmol

Requires the optional `openbabel` package (the SMILES->mol2 step).

Usage:
  python smiles_to_params.py "<SMILES>" <out_prefix> [--res-name LG1]
                             [--ph 7.4] [--no-protonate] [--sample-proton-chi]
Writes <out_prefix>.tmol .
"""

import argparse

from tmol.ligand import _params_io as params_io
from tmol.ligand import prepare_single_ligand
from tmol.ligand._detect import nonstandard_residue_info_from_smiles_via_mol2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("smiles", help="input SMILES string")
    ap.add_argument("out_prefix", help="output path prefix (.tmol appended)")
    ap.add_argument("--res-name", default="LG1", help="residue name (default LG1)")
    ap.add_argument(
        "--ph", type=float, default=7.4, help="protonation pH (default 7.4)"
    )
    ap.add_argument(
        "--no-protonate",
        action="store_true",
        help="treat the SMILES as already protonated (skip Dimorphite)",
    )
    ap.add_argument(
        "--sample-proton-chi",
        action="store_true",
        help="emit PROTON_CHI samples (default off)",
    )
    ap.add_argument(
        "--no-conformer-search",
        action="store_true",
        help="skip the rotor conformer search (faster single-conformer 3D)",
    )
    args = ap.parse_args()

    # 1. SMILES -> mol2 -> NonStandardResidueInfo
    info = nonstandard_residue_info_from_smiles_via_mol2(
        args.smiles,
        res_name=args.res_name,
        ph=args.ph,
        protonate=not args.no_protonate,
        conformer_search=not args.no_conformer_search,
    )
    # 2. prepare (type, charges, cartbonded)
    prep = prepare_single_ligand(info, sample_proton_chi=args.sample_proton_chi)
    # 3. tmol .tmol
    tmol_path = f"{args.out_prefix}.tmol"
    params_io.write_params_file(prep, tmol_path)
    print(f"wrote {tmol_path}  ({len(prep.residue_type.atoms)} atoms)")


if __name__ == "__main__":
    main()
