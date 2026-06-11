"""Example: prepare a ligand from a mol2 and emit a Rosetta ``.params`` file.

This mirrors what ``mol2genparams.py`` does in Rosetta (read a mol2 verbatim,
run Rosetta-style atom typing, produce a params file) without re-protonating,
regenerating 3D coordinates, or recomputing partial charges when the mol2
already carries authoritative charges.

Pipeline (real public API):
    1. ``prepare_ligand_from_mol2(mol2)`` -> ``(ParameterDatabase, CanonicalOrdering)``.
       Reads the mol2 once (atom names, coordinates, partial charges, and bond
       orders come straight out of the TRIPOS sections) and returns a fresh
       ``ParameterDatabase`` with the ligand's chemical/Elec/CartBonded entries
       injected, plus the rebuilt canonical ordering.
    2. ``write_params_from_mol2(mol2, out_params)`` writes a Rosetta-compatible
       ``.params`` file for the same ligand.

For the full mol2 -> PoseStack path, drive ``pose_stack_from_biotite`` with the
ligand-bearing structure and ``prepare_ligands=True`` (the integrated,
CIF/biotite-driven flow); this example focuses on the params-emitting half.

Usage::

    python examples/workflow_from_mol2.py [LIGAND_MOL2] [--out-params PATH]

Default ligand (invoked from the repo root with no arguments)::

    LIGAND_MOL2   tmol/tests/data/ligand_ground_truth/ref1.mol2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tmol.ligand import prepare_ligand_from_mol2, write_params_from_mol2

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIGAND = (
    REPO_ROOT / "tmol" / "tests" / "data" / "ligand_ground_truth" / "ref1.mol2"
)


def run(ligand_mol2: str, res_name: str = "LG1", out_params: Path | None = None):
    """Prepare a ligand mol2 and optionally write its ``.params`` file.

    Args:
        ligand_mol2: Path to the ligand mol2.
        res_name: Three-letter residue name for the ligand.
        out_params: Optional ``.params`` output path.

    Returns:
        The ``(ParameterDatabase, CanonicalOrdering)`` from preparation.
    """
    param_db, canonical_ordering = prepare_ligand_from_mol2(
        ligand_mol2, res_name=res_name
    )
    if out_params is not None:
        write_params_from_mol2(ligand_mol2, str(out_params), res_name=res_name)
    return param_db, canonical_ordering


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        "--out-params",
        type=Path,
        default=None,
        help="Optional path to write a Rosetta-compatible .params file.",
    )
    args = parser.parse_args()

    print(f"[info] reading ligand mol2: {args.ligand}")
    param_db, _ordering = run(
        args.ligand, res_name=args.res_name, out_params=args.out_params
    )
    n_elec = sum(
        1
        for p in param_db.scoring.elec.atom_charge_parameters
        if p.res == args.res_name
    )
    print(f"[info] registered ligand '{args.res_name}': elec_charges={n_elec}")
    if args.out_params is not None:
        print(f"[info] wrote params file: {args.out_params}")
    print("[done] mol2 -> params pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
