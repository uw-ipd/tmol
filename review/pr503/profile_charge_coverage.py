"""Check all default charges and time the added used-charge validation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import attr
import numpy
import torch

from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.score.elec import ElecParamResolver
from profile_parameter_coverage import measure

BASELINE = "b1c4834d8382cc5efc11479042adf311896f4fbe"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:tmol/score/elec/_params.py"], text=True
    )
    namespace = {
        "__name__": "tmol.score.elec._params",
        "__package__": "tmol.score.elec",
    }
    exec(compile(source, "previous_elec_params.py", "exec"), namespace)
    database = ParameterDatabase.get_default()
    # Reconstruct the preceding catalog's only absent residue charge table.
    old_elec = attr.evolve(
        database.scoring.elec,
        atom_charge_parameters=tuple(
            r for r in database.scoring.elec.atom_charge_parameters if r.res != "HOH"
        ),
    )
    device = torch.device("cpu")
    resolvers = {
        "before": namespace["ElecParamResolver"].from_database(old_elec, device),
        "after": ElecParamResolver.from_database(database.scoring.elec, device),
    }
    restypes = ResidueTypeSet.from_database(database.chemical).residue_types
    for block in restypes:
        numpy.testing.assert_array_equal(
            *(r.get_partial_charges_for_block(block) for r in resolvers.values())
        )
    ala = next(r for r in restypes if r.name == "ALA")
    output = dict(
        baseline=BASELINE,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        all_default_charges_exact=True,
        n_types=len(restypes),
        n_atoms=sum(len(r.atoms) for r in restypes),
        measurements={},
    )
    for label, blocks in [("ALA", [ala]), ("default_catalog", restypes)]:
        functions = {
            k: lambda resolver=resolver: [
                resolver.get_partial_charges_for_block(bt) for bt in blocks
            ]
            for k, resolver in resolvers.items()
        }
        output["measurements"][label] = measure(functions, device)
    output["scope"] = (
        "Nine alternating warm rounds of five calls. Host charge lookup only; excludes constructor, packed geometry, transfer, scoring, and native/RSS/GPU storage. Validation overhead is reported, not a speedup. Before catalog omits HOH as in the pinned parent; after explicitly stores the same zero charges."
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
