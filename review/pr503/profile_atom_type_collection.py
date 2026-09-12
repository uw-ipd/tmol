"""Benchmark shared atom-type collection without patching or scoring."""

import argparse
import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import attr

from tmol.database import ParameterDatabase
from tmol.ligand import _registry
from tmol.tests.ligand.test_ligand_entry_paths import _single_prep
from review.pr503.profile_parameter_replacements import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="74fda5c53")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:tmol/ligand/_registry.py"], text=True
    )
    tree = ast.Module(
        body=[
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "collect_new_atom_types"
        ],
        type_ignores=[],
    )
    scope = dict(vars(_registry))
    exec(compile(tree, args.baseline + ":collect_new_atom_types", "exec"), scope)
    previous = scope["collect_new_atom_types"]
    prep = _single_prep()
    chem = ParameterDatabase.get_default().chemical
    cases = {}
    for custom in (False, True):
        row = prep
        if custom:
            atoms = prep.residue_type.atoms
            row = replace(
                prep,
                residue_type=attr.evolve(
                    prep.residue_type,
                    atoms=(
                        attr.evolve(atoms[0], atom_type="UnfamiliarType"),
                        *atoms[1:],
                    ),
                ),
                atom_type_elements={**prep.atom_type_elements, "UnfamiliarType": "C"},
            )
        for count in (1, 1000):
            residues = [
                SimpleNamespace(name=f"ligand{i}", atoms=row.residue_type.atoms)
                for i in range(count)
            ]

            def old():
                unique = {}
                for rt in residues:
                    for at in previous(
                        chem, rt, row.atom_type_elements, strict_atom_types=True
                    ):
                        unique.setdefault(at.name, at)
                return list(unique.values())

            def new():
                return _registry._collect_new_atom_types(
                    chem,
                    ((r.atoms, f"residue {r.name}") for r in residues),
                    row.atom_type_elements,
                    strict_atom_types=True,
                )

            assert old() == new()
            cases[f'{"custom" if custom else "known"}_{count}'] = {
                "types": len(new()),
                "old": measure(old, 9),
                "new": measure(new, 9),
            }
    args.output.write_text(
        json.dumps({"baseline": args.baseline, "cases": cases}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
