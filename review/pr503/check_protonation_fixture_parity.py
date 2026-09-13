"""Compare all literal SMILES fixtures under identical protonation policies.

Collects valid molecular strings from both repositories' Python tests and the
vendor's built-in rule tests. It compares the ordered products and heavy-atom
maps against a saved tmol engine, recording default AtomWorks differences too.
This is compatibility validation, not a pKa refit or validation of hydrogen
placement in full structures.
"""

import argparse
import ast
import importlib.util
import json
from pathlib import Path

from rdkit import Chem, rdBase

from atomworks.external.dimorphite_dl import dimorphite_dl as shared


def products(engine, mol, ph):
    return [
        Chem.MolToSmiles(m)
        for m in engine.protonate_mol_variants(
            mol,
            min_ph=ph,
            max_ph=ph,
            pka_precision=0.1,
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--atomworks", type=Path, required=True)
    parser.add_argument("--old-atomworks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("baseline_protonation", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    tmol = Path(__file__).resolve().parents[2]
    current_spec = importlib.util.spec_from_file_location(
        "current_protonation", tmol / "tmol/ligand/_dimorphite_dl.py"
    )
    current = importlib.util.module_from_spec(current_spec)
    current_spec.loader.exec_module(current)
    roots = {
        "tmol": tmol / "tmol/tests",
        "atomworks": args.atomworks / "tests",
        "old_atomworks": args.old_atomworks / "tests",
    }
    files = [
        (label, p, p.relative_to(root))
        for label, root in roots.items()
        for p in root.rglob("test*.py")
    ]
    files.extend(
        ("vendor", Path(shared.__file__), Path("dimorphite_dl.py")) for _ in range(1)
    )
    fixtures = {}
    with rdBase.BlockLogs():
        for label, path, relative in files:
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Constant) or not isinstance(
                    node.value, str
                ):
                    continue
                value = node.value
                if not 2 <= len(value) <= 1000 or any(c.isspace() for c in value):
                    continue
                mol = Chem.MolFromSmiles(value)
                if mol is None or mol.GetNumHeavyAtoms() < 2:
                    continue
                fixtures.setdefault(value, []).append(
                    f"{label}/{relative}:{node.lineno}"
                )
        rows = []
        for smiles, sources in sorted(fixtures.items()):
            mol = Chem.MolFromSmiles(smiles)
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
            for ph in (2.0, 7.4, 12.0):
                row = dict(smiles=smiles, sources=sources, ph=ph)
                for name, engine in (
                    ("baseline", baseline),
                    ("shared_tmol_rules", current),
                    ("atomworks_default", shared),
                ):
                    try:
                        row[name] = products(engine, mol, ph)
                    except Exception as exc:
                        row[name] = {"error": type(exc).__name__, "message": str(exc)}
                rows.append(row)
    mismatches = [r for r in rows if r["baseline"] != r["shared_tmol_rules"]]
    defaults = [r for r in rows if r["baseline"] != r["atomworks_default"]]
    report = dict(
        molecules=len(fixtures),
        comparisons=len(rows),
        shared_mismatches=len(mismatches),
        default_differences=len(defaults),
        rows=rows,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k != "rows"})
    for row in mismatches[:10]:
        print(row)
    return bool(mismatches)


if __name__ == "__main__":
    raise SystemExit(main())
