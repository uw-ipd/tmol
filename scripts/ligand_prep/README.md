# Small-molecule ligand prep & parity

tmol turns a small molecule (a SMILES string **or** a prepared mol2) into a
Rosetta `.params` file and a tmol `.tmol` params file. This directory holds the
scripts that drive that pipeline and reproduce/validate its parity against
RosettaVS `mol2genparams.py` over an 80-molecule DUD-80 set.

The implementation lives in `tmol/ligand/`; these scripts are thin drivers.
Running them requires `tmol` to be importable and the optional **`openbabel`**
package (the SMILES→mol2 step). The ground-truth *generator* additionally needs
the external `dimorphite_dl` and RosettaVS `mol2genparams.py` tools.

```
scripts/ligand_prep/
  smiles_to_params.py               # 1 SMILES  -> .params + .tmol   (CLI driver)
  validate_dud80.py                 # run the 80-case parity test
  build_ligand_test_ground_truth.sh # regenerate the ground truth (RosettaVS)
  legacy/                           # superseded manual shell pipeline (reference)

tmol/tests/data/ligand_test/
  ligand_ground_truth/              # COMMITTED ground truth (see below)
    dud80.smi                       #   error-free input SMILES (80)
    dud80.prot.smi                  #   dimorphite-protonated SMILES
    manifest.json                   #   per-molecule test manifest
    mol2/<name>.mol2                #   RosettaVS-prepped mol2  (80)
    params/<name>.params            #   RosettaVS .params ground truth (80)
  ligand_tmol_generated/            # NOT committed: written by validate_dud80.py
```

---

## 1. Process a small molecule → tmol params

### Python API (preferred)

```python
from tmol.ligand import (
    prepare_ligand_from_smiles,   # SMILES -> (ParameterDatabase, CanonicalOrdering)
    prepare_ligand_from_mol2,     # mol2   -> (ParameterDatabase, CanonicalOrdering)
    write_params_from_mol2,       # mol2   -> Rosetta .params file
)

# From a mol2 (atom names, 3D coords, charges, bond orders preserved verbatim):
write_params_from_mol2("ligand.mol2", "ligand.params")

# From a SMILES, injected into an in-memory database:
param_db, canonical_ordering = prepare_ligand_from_smiles(
    "CC(=O)Oc1ccccc1C(=O)O", res_name="ASP"
)
```

The SMILES path runs the canonical protocol end to end: normalize bare `[O]`
→ `[O-]`, Dimorphite-DL protonate at the given pH, OpenBabel 3D + MMFF94 mol2
(with a rotor **conformer search**, on by default), then read that mol2 verbatim.
Pass `conformer_search=False` for faster single-conformer generation, or
`protonate=False` to pin an already-protonated SMILES.

### CLI driver

```bash
python scripts/ligand_prep/smiles_to_params.py "<SMILES>" <out_prefix> \
    [--res-name LG1] [--ph 7.4] [--no-protonate] \
    [--sample-proton-chi] [--no-conformer-search]
# writes <out_prefix>.params (Rosetta) and <out_prefix>.tmol (tmol)
```

---

## 2. Run the 80-case parity test

`validate_dud80.py` prepares every molecule in the manifest two ways and checks
each against the RosettaVS `.params` ground truth (name-agnostic, via heavy-atom
graph isomorphism):

- **(A) SMILES → mol2** — tmol generates the mol2 (OpenBabel + conformer search)
  and saves the mol2 / `.params` / `.tmol` under `ligand_tmol_generated/`.
- **(B) mol2 direct** — reads the committed ground-truth mol2.

```bash
python scripts/ligand_prep/validate_dud80.py
# ... per molecule, then:
#   ### (A) SMILES -> mol2 (tmol-generated, saved)
#     chemistry: 80/80   full (+CHI): 80/80
#   ### (B) mol2 direct
#     chemistry: 80/80   full (+CHI): 80/80
```

"chemistry" = atoms / types / bonds / charges; "full" also requires the CHI /
PROTON_CHI axes to match. The script exits non-zero if any molecule fails the
chemistry check on either path. It only writes under `ligand_tmol_generated/`
(git-ignored) — the committed ground truth is never modified.

The focused parity assertions also run under pytest:

```bash
pytest tmol/tests/ligand/test_smiles_semantic.py \
       tmol/tests/ligand/test_serialization_consistency.py \
       tmol/tests/ligand/test_parity_manifest.py \
       tmol/tests/ligand/test_ligand_pipeline.py
```

---

## 3. Reproduce the ground truth

The committed ground truth was produced by RosettaVS, so others don't need to
regenerate it. To rebuild it from `dud80.smi`:

```bash
# needs the `openvs` conda env (obabel, dimorphite_dl, mol2genparams.py)
scripts/ligand_prep/build_ligand_test_ground_truth.sh
# -> tmol/tests/data/ligand_test/ligand_ground_truth/{dud80.prot.smi,mol2/,params/,manifest.json}
```

Pipeline (the `legacy/` protocol, applied per molecule):
dimorphite-dl (pH 7.4) → OpenBabel `smi → pdb → mol2 -h` MMFF94 (conformer-search
fallback where `mol2genparams` fails) → `mol2genparams.py` (`--resname LG1
--rename_atoms`). The input `dud80.smi` is assumed already correct — the
`[O]`→`[O-]` carboxylate fix is applied once when preparing `dud80.smi`, **not**
in this pipeline.
