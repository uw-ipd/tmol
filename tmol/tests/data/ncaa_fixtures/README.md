# Noncanonical polymer residue fixtures

Small real structures carrying common noncanonical residues, trimmed to a
single chain of model 1 with solvent and non-polymer heteroatoms removed.

| file | source | residues | noncanonical |
|------|--------|----------|--------------|
| `phosphopeptide_5ema.cif` | PDB 5EMA chain B | 7 | `SEP` (phosphoserine) |
| `collagen_hyp_1bkv.cif` | PDB 1BKV chain A | 29 | `HYP` (4-hydroxyproline) |

`SEP` is an open-chain sidechain carrying a dianionic phosphate; `HYP` closes a
ring back onto the backbone nitrogen, so the two together cover both sidechain
topologies the alpha-amino-acid profile has to handle.

Read these with `include_bonds=True`: the polymer path derives its chemistry
from the bond table, not from geometry perception.
