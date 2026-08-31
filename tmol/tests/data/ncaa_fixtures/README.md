# Noncanonical polymer residue fixtures

Small real structures carrying common noncanonical residues, trimmed to a
single chain of model 1 with solvent and non-polymer heteroatoms removed.

| file | source | residues | noncanonical |
|------|--------|----------|--------------|
| `phosphopeptide_5ema.cif` | PDB 5EMA chain B | 7 | `SEP` (phosphoserine) |
| `collagen_hyp_1bkv.cif` | PDB 1BKV chain A | 29 | `HYP` (4-hydroxyproline) |
| `capped_peptide_ace_nme.cif` | built | 3 | `ACE`, `NME` (terminal caps) |
| `capped_peptide_ace_nh2.cif` | built | 3 | `ACE`, `NH2` (terminal caps) |
| `nmethyl_peptide_6mvz.cif` | PDB 6MVZ chain A | 4 | `MLE` (N-methylleucine) |
| `gamma_peptide_1gac.cif` | PDB 1GAC chain A | 5 | `FGA` (gamma-D-glutamate) |
| `beta_peptide_3c3g.cif` | PDB 3C3G chain A | 31 | eight beta backbones |

`SEP` is an open-chain sidechain carrying a dianionic phosphate; `HYP` closes a
ring back onto the backbone nitrogen, so the two together cover both sidechain
topologies the alpha-amino-acid profile has to handle.

Read these with `include_bonds=True`: the polymer path derives its chemistry
from the bond table, not from geometry perception.

## Capped peptide

`capped_peptide_ace_nme.cif` is Ace-Ala-NMe, 10 heavy atoms, built rather than
taken from a deposited entry: alanine on its CCD ideal coordinates, with the two
caps placed on standard amide internal coordinates.

Both caps are `NON-POLYMER` components in the CCD, so nothing in their component
type says they belong to the chain. The peptide bonds joining them to the
alanine exist only because the file declares them in `struct_conn`, the way a
deposited entry does; without that block biotite leaves them unbonded and they
are free molecules rather than caps.

`capped_peptide_ace_nh2.cif` is the same peptide with the methylamide
replaced by a bare amide. `NH2` is a single heavy atom, too few to supply a
reference frame, so the stubs completing it have to be placed against
invented points.

## Nonstandard backbones

These three are whole structures rather than single residues, so the pipeline
has to find the connections itself instead of being handed them.

`nmethyl_peptide_6mvz.cif` carries N-methylleucine at two positions. A
substituent on the amide nitrogen that closes no ring back onto the mainchain
is what separates it from proline, so it is prepared as a ligand.

`gamma_peptide_1gac.cif` is a peptidoglycan stem peptide, `ALA-FGA-LYS-DAL-DAL`.
The chain leaves gamma-glutamate through `CD`, four bonds along the sidechain
from the alpha carbon, while its alpha fragment stays intact -- so neither the
conventional names nor the shortest path between connections finds the backbone.

`beta_peptide_3c3g.cif` is a beta-peptide foldamer with eight distinct beta
backbones interleaved with canonical residues. Each mainchain runs through four
atoms rather than three; declaring the alpha `N-CA-C` instead would measure phi,
psi and omega across a bond that does not exist.

Trifluoroacetate, methanol, glycerol and acetate were dropped along with the
solvent: they are unbonded in these entries, so they would arrive as free
molecules on the ligand path and have nothing to do with the backbone.

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
