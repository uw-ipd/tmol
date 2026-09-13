"""Measure repeated-file batches, including parsing and residue installation.

The old path produces duplicate residue definitions for repeated new inputs;
this is a correctness repair, not equivalent full-scoring workloads.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess
import tempfile

import attr

from tmol.database import ParameterDatabase
from tmol.ligand import _params_file, _registry, inject_params_files, write_params_file
from tmol.tests.ligand.test_ligand_entry_paths import _single_prep
from review.pr503.profile_parameter_replacements import measure


def previous_function(ref, path, name, namespace):
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
    tree = ast.Module(
        body=[
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        ],
        type_ignores=[],
    )
    assert len(tree.body) == 1
    scope = dict(vars(namespace))
    exec(compile(tree, f"{ref}:{path}", "exec"), scope)
    return scope[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="c8efb34ca")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_load = previous_function(
        args.baseline, "tmol/ligand/_params_file.py", "load_params_file", _params_file
    )
    old_inject = previous_function(
        args.baseline, "tmol/ligand/_registry.py", "_inject_additions", _registry
    )
    prep = _single_prep()
    base = ParameterDatabase.get_default()
    cases = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ligand.tmol"
        write_params_file(prep, path, format="tmol")
        for copies in (1, 20, 100):

            def old():
                return old_inject(
                    base, [p for _ in range(copies) for p in old_load(path)], False
                )

            def new():
                return inject_params_files(base, [path] * copies)

            before, after = old(), new()
            assert {r.name: r for r in before.chemical.residues} == {
                r.name: r for r in after.chemical.residues
            }
            assert before.scoring.cartbonded == after.scoring.cartbonded
            assert before.scoring.elec == after.scoring.elec
            for field in attr.fields(type(before.scoring)):
                if field.name not in ("cartbonded", "elec"):
                    assert getattr(before.scoring, field.name) is getattr(
                        after.scoring, field.name
                    )
            old_added = len(before.chemical.residues) - len(base.chemical.residues)
            new_added = len(after.chemical.residues) - len(base.chemical.residues)
            assert old_added == copies * new_added
            cases[str(copies)] = {
                "old_added_definitions": old_added,
                "new_added_definitions": new_added,
                "old": measure(old, args.repeats),
                "new": measure(new, args.repeats),
            }
    args.output.write_text(
        json.dumps(
            {"baseline": args.baseline, "repeats": args.repeats, "cases": cases},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
