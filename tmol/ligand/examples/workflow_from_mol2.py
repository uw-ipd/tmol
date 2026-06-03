"""End-to-end workflow: load a ligand from a mol2, register it, build a PoseStack.

This example mirrors what ``mol2genparams.py`` does in Rosetta (read a mol2
verbatim, run Rosetta-style atom typing, produce a params file) and then
plugs the resulting residue type into tmol's PoseStack construction path
*without* re-protonating, regenerating 3D coordinates, or recomputing
partial charges.

Pipeline:
    1. ``prepare_ligand_from_mol2(mol2)`` -> ``(RawResidueType, charges)``.
       Reads the mol2 file once: atom names, 3D coordinates, partial
       charges and bond orders come straight out of the TRIPOS sections.
    2. ``register_ligand(param_db, restype, charges)`` injects the ligand
       into a ParameterDatabase (chemical DB + Elec + CartBonded).
    3. ``atom_array_from_mol2(mol2)`` builds a Biotite ``AtomArray`` from
       the same mol2 (preserving names + coords + bonds). Annotations
       ``res_name``/``chain_id``/``res_id`` must match the values used in
       step 1 so the PoseStack builder can look the ligand up.
    4. Concatenate ``protein + ligand`` and call
       ``pose_stack_from_biotite(combined, device,
       prepare_ligands=False, param_db=param_db)``. The
       ``prepare_ligands=False`` is critical -- it prevents the second,
       destructive ligand-rebuild path that the CIF-driven flow uses.
    5. Verify the PoseStack has finite coordinates.

Usage::

    python examples/workflow_from_mol2.py [PROTEIN] [LIGAND_MOL2]

Defaults (when invoked from the repo root with no arguments)::

    PROTEIN       tmol/tests/data/pdb/1ubq.pdb
    LIGAND_MOL2   tmol/tests/data/ligand_ground_truth/ref1.mol2

The ligand is dropped into chain "L" with a residue id outside the
protein's range. Coordinates from the mol2 are kept as-is; this example
does not attempt to dock or position the ligand inside the protein's
binding site.

Note on scoring:
    The beta2016 score function bundled with tmol includes the Dunbrack
    rotamer term, which currently does not support ligand blocks and will
    raise SIGFPE on ligand-containing pose stacks. This script stops at
    PoseStack construction + finite-coordinate sanity check. To score, you
    must use a score function that excludes Dunbrack/Rotamer terms for
    ligand residues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import biotite.structure.io as strucio
import torch

from tmol.database import ParameterDatabase
from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
from tmol.ligand import atom_array_from_mol2, prepare_ligand_from_mol2
from tmol.ligand.registry import register_ligand

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTEIN = REPO_ROOT / "tmol" / "tests" / "data" / "pdb" / "1ubq.pdb"
DEFAULT_LIGAND = (
    REPO_ROOT / "tmol" / "tests" / "data" / "ligand_ground_truth" / "ref1.mol2"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "protein",
        nargs="?",
        default=str(DEFAULT_PROTEIN),
        help="Protein PDB or CIF file (default: %(default)s)",
    )
    parser.add_argument(
        "ligand",
        nargs="?",
        default=str(DEFAULT_LIGAND),
        help="Ligand mol2 file (default: %(default)s)",
    )
    parser.add_argument(
        "--res-name",
        default="LG1",
        help="Three-letter residue name for the ligand (default: %(default)s)",
    )
    parser.add_argument(
        "--chain-id",
        default="L",
        help="Chain id for the ligand (default: %(default)s)",
    )
    parser.add_argument(
        "--rename-atoms",
        action="store_true",
        help="Regenerate atom names using Rosetta convention "
        "(C1/HC1/HO1/...) instead of preserving mol2 names.",
    )
    parser.add_argument(
        "--out-params",
        type=Path,
        default=None,
        help="Optional path to also write a Rosetta-compatible .params file "
        "for the registered ligand.",
    )
    args = parser.parse_args()

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {torch_device}")

    # --- Step 1: read mol2 verbatim ----------------------------------------
    print(f"[info] reading ligand mol2: {args.ligand}")
    restype, charges = prepare_ligand_from_mol2(
        args.ligand,
        res_name=args.res_name,
        rename_atoms=args.rename_atoms,
    )
    print(
        f"[info] ligand restype: name={restype.name} "
        f"atoms={len(restype.atoms)} bonds={len(restype.bonds)}"
    )

    if args.out_params is not None:
        from tmol.ligand import write_params_from_mol2

        write_params_from_mol2(
            args.ligand,
            args.out_params,
            res_name=args.res_name,
            rename_atoms=args.rename_atoms,
        )
        print(f"[info] wrote params file: {args.out_params}")

    # --- Step 2: register into a fresh ParameterDatabase -------------------
    param_db = ParameterDatabase.get_fresh_default()
    inserted = register_ligand(param_db, restype, partial_charges=charges)
    if not inserted:
        print(
            f"[warn] residue {restype.name} was already in the parameter database",
            file=sys.stderr,
        )
    elec_entries = sum(
        1 for p in param_db.scoring.elec.atom_charge_parameters if p.res == restype.name
    )
    cart_entry = param_db.scoring.cartbonded.residue_params.get(restype.name)
    print(
        f"[info] registered: elec_charges={elec_entries} "
        f"cartbonded={'present' if cart_entry else 'missing'}"
    )

    # --- Step 3: build the ligand AtomArray --------------------------------
    print(f"[info] loading protein: {args.protein}")
    protein = strucio.load_structure(args.protein)
    if hasattr(protein, "stack_depth"):  # AtomArrayStack
        protein = protein[0]

    # Pick a residue id outside the protein's range so nothing collides.
    lig_res_id = int(protein.res_id.max()) + 100

    ligand_arr = atom_array_from_mol2(
        args.ligand,
        res_name=args.res_name,
        chain_id=args.chain_id,
        res_id=lig_res_id,
        rename_atoms=args.rename_atoms,
    )
    print(
        f"[info] ligand AtomArray: {len(ligand_arr)} atoms "
        f"(chain={args.chain_id} res_id={lig_res_id})"
    )

    # --- Step 4: concatenate and build the PoseStack -----------------------
    combined = protein + ligand_arr
    print(
        f"[info] combined: {len(combined)} atoms "
        f"(protein={len(protein)} + ligand={len(ligand_arr)})"
    )

    pose_stack = pose_stack_from_biotite(
        combined,
        torch_device,
        prepare_ligands=False,  # critical: do not rebuild the ligand
        param_db=param_db,
    )
    print(
        f"[info] PoseStack: n_poses={pose_stack.coords.shape[0]} "
        f"max_n_blocks={pose_stack.coords.shape[1]} "
        f"max_n_atoms_per_block={pose_stack.coords.shape[2]}"
    )

    # --- Step 5: sanity check coordinates ----------------------------------
    nonzero = pose_stack.coords[pose_stack.coords != 0]
    if torch.isnan(nonzero).any() or torch.isinf(nonzero).any():
        print("[error] PoseStack contains NaN or inf coordinates", file=sys.stderr)
        return 1
    print("[info] coordinates: all finite")
    print("[done] mol2 -> PoseStack pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
