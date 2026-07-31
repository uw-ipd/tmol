# Small-molecule ligand prep

`smiles_to_params.py` turns a SMILES string into a Rosetta `.params` file and a
tmol `.tmol` params file, driving tmol's ligand pipeline (`tmol/ligand/`).

Requires `tmol` to be importable and the optional **`openbabel`** package (the
SMILES→mol2 step).

### Python API (preferred)

```python
from tmol.ligand import (
    prepare_ligand_from_smiles,   # SMILES -> (ParameterDatabase, CanonicalOrdering)
    prepare_ligand_from_mol2,     # mol2   -> (ParameterDatabase, CanonicalOrdering)
    write_params_from_mol2,       # mol2   -> Rosetta .params file
)
```

### CLI driver

```bash
python scripts/ligand_prep/smiles_to_params.py "<SMILES>" <out_prefix> \
    [--res-name LG1] [--ph 7.4] [--no-protonate] \
    [--sample-proton-chi] [--no-conformer-search]
# writes <out_prefix>.params (Rosetta) and <out_prefix>.tmol (tmol)
```

The SMILES path runs the canonical protocol end to end: normalize bare `[O]`
→ `[O-]`, Dimorphite-DL protonate at the given pH, OpenBabel 3D + MMFF94 mol2
(rotor conformer search on by default), then read that mol2 verbatim. Pass
`--no-conformer-search` for faster single-conformer generation, or
`--no-protonate` to pin an already-protonated SMILES.

Parity of the pipeline against RosettaVS is covered by the test suite
(`tmol/tests/ligand/`, e.g. `test_dud_ligands.py`, `test_smiles_semantic.py`)
against committed ground truth under `tmol/tests/data/`.
