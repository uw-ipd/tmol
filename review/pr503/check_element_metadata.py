"""Check legacy element loss and old-reader rejection of the new format."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import attr

from tmol.database import ParameterDatabase
from tmol.ligand import _params_file, _params_io, collect_new_atom_types
from tmol.tests.ligand.test_ligand_entry_paths import _single_prep
from review.pr503.profile_parameter_batches import previous_function


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="74fda5c53")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old_writer = previous_function(
        args.baseline, "tmol/ligand/_params_io.py", "write_params_file", _params_io
    )
    old_reader = previous_function(
        args.baseline, "tmol/ligand/_params_file.py", "load_params_file", _params_file
    )
    old_reader.__globals__["TMOL_FORMAT_VERSION"] = "4.0"
    prep = _single_prep()
    chem = ParameterDatabase.get_default().chemical
    rows = []
    for element in ("N", "S", "Cl", "H"):
        atoms = prep.residue_type.atoms
        custom = replace(
            prep,
            residue_type=attr.evolve(
                prep.residue_type,
                atoms=(attr.evolve(atoms[0], atom_type="UnfamiliarType"), *atoms[1:]),
            ),
            atom_type_elements={"UnfamiliarType": element},
        )
        old_path = args.output / f"{element}-legacy.tmol"
        new_path = args.output / f"{element}-declared.tmol"
        old_writer(custom, old_path, format="tmol")
        _params_io.write_params_file(custom, new_path, format="tmol")
        values = {}
        for name, path in (("legacy", old_path), ("current", new_path)):
            loaded = _params_file.load_params_file(path)[0]
            types = collect_new_atom_types(
                chem, loaded.residue_type, loaded.atom_type_elements
            )
            values[name] = next(t.element for t in types if t.name == "UnfamiliarType")
        assert values == {"legacy": "C", "current": element}
        try:
            old_reader(new_path)
        except ValueError as error:
            assert "incompatible" in str(error)
            old_error = str(error)
        else:
            raise AssertionError(
                "Old reader silently accepted mandatory element metadata"
            )
        rows.append({"declared": element, **values, "old_reader_error": old_error})
    (args.output / "checks.json").write_text(
        json.dumps({"baseline": args.baseline, "rows": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
