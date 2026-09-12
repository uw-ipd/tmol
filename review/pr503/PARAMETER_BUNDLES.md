# Guarded residue replacements

Ordinary preparations still add residue definitions and their patch/scoring
metadata. An existing base definition is skipped. A preparation can now opt
into replacing an **exact residue name** by supplying `baseline_sha256` along
with its complete residue definition, atom charges and bonded record.

The guard accepts either the original baseline or the complete requested
result. A different baseline raises before any bundle metadata can reset its
charges. Identical reloads return the same database object. All operations
leave the input database unchanged.

This is parameter delivery, not a new default force field. The private coupled
MMFF generator remains opt-in; its scientific charge policy, three-block
angles and context-dependent residue naming still need resolution.

## Using the coupled generator

Starting with `array` and its ordinarily prepared `database`:

```python
from tmol.ligand import load_params_file, write_params_file
from tmol.ligand._local_conjugate_params import generate_conjugate_parameters
from tmol.ligand._registry import LigandPreparation, inject_ligand_preparations

result = generate_conjugate_parameters(array, database)
corrections = [
    LigandPreparation(
        residue_type=row.residue_type,
        partial_charges=row.partial_charges,
        cartbonded_params=row.cartbonded_params,
        baseline_sha256=row.baseline_sha256,
        connection_params=result.connections if i == 0 else (),
    )
    for i, row in enumerate(result.residues)
]
write_params_file(corrections, "corrections.tmol", format="tmol")
corrected = inject_ligand_preparations(database, load_params_file("corrections.tmol"))
assert inject_ligand_preparations(corrected, corrections) is corrected
```

A correction-only bundle requires its baseline residue names to exist. To
create a standalone bundle, retain the ordinary preparations exported with
`prepare_ligands(..., params_output="baseline.tmol")`, then write
`load_params_file("baseline.tmol") + corrections` together. Loading that file
through `prepare_ligands(..., params_files=[...])` installs ordinary definitions
and patches first, validates the resulting baseline, and installs corrections.
Addition/replacement list order does not change the named parameters. Ordinary
residue insertion order can still change database indices.

## Format and scope

Guarded bundles use `.tmol` version **4.0**. Their residue/charge/bonded target
records retain the usual schema. `chemical.replacement_baselines` maps exact
residue names to baseline digests. If a combined bundle also carries old patch
parameters for those names, `elec.replacement_baseline_charges` and
`cartbonded.replacement_baseline_params` preserve that addition-stage metadata
separately from complete target records. These mappings may only name guarded
residues. Existing target residues are checked before those old values can be
applied. Connection conflicts are rejected as well.

The digest covers the complete `RawResidueType`, effective atom charges under
the resolver's exact/patch/base precedence, and the selected local `CartRes`.
Declared scalar types and JSON string encoding make it stable across NumPy
strings, Python strings, and serialization of integer-valued float fields.
The digest has its own `tmol-residue-replacement-v1` domain tag. It does **not**
cover every atom-type definition, score weight or global force-field table.
It therefore cannot establish scientific compatibility by itself.

Versions 1–3 remain readable. Writers emit version 2 for ordinary bundles and
version 3 for generic atom references; only guarded replacements require
version 4. An older version-3 reader rejects the version-4 bundle, and Rosetta
`.params` export rejects replacements because it cannot carry their guard.
Experimental private conjugate results made with the earlier representation-
dependent digest must be regenerated; ordinary legacy bundles are unaffected.

A replacement applies to its exact named type, including any explicitly
supplied terminal forms. It does not automatically correct other variants or
resolve two chemically different attachments that share one type name.

## Validation and cost

The replacement tests exercise biotin, N-glycans and O-glycans; fresh/prepared/
corrected databases; reversed bundle order; public preparation; baseline
mismatches; incomplete charges/bonded records; duplicate/conflicting records;
legacy readers; exact coordinates; and native scores and gradients. CUDA
comparisons include repeated unchanged-database evaluations to characterize
reduction roundoff. Default chemistry selection and scoring kernels are
unchanged.

`profile_parameter_replacements.py` compares the previous private installer
with the shared implementation on the same generated parameters. It uses each
implementation's own baseline digest and excludes chemistry generation from
timing. The shared installer hashes the completed bonded database once instead
of twice. Python traced allocation measurements are not process RSS or GPU
memory. See [results/parameter-replacement-validation.json](results/parameter-replacement-validation.json)
for the executable checks, timings and remaining limitations.
