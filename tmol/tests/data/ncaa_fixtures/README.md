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

## Mirror-image pair

`6dmz_mod_l.cif` and `6dmz_mod_d.cif` are 6DMZ chain A with its hydrogens kept.
The D file is the same structure with every coordinate negated -- point
inversion is improper, so it is the mirror image -- and every residue renamed to
its D code.

47 residues covering four disulfides (3-47, 14-34, 20-41, 24-43), two prolines
and two histidines, so one pair exercises CYD, the disulfide term and the
pre-proline tables together.

Two modifications, marked by `_mod`:

* every alanine becomes a glycine, so the pair also exercises the symmetrized
  glycine tables. CB is kept as HA3 at a hydrogen bond length along the same
  direction, and HA becomes HA2, so both alpha hydrogens come from the file
  rather than being built -- a built pair of prochiral hydrogens would not be
  guaranteed to mirror exactly.

Hydrogens matter here beyond geometry: a histidine whose HD1 and HE2 are both
absent has its tautomer chosen arbitrarily, while one carrying HE2 has it
inferred. Keeping them makes the comparison deterministic.
