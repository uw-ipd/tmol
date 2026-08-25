# Programmatic modified-component implementation

## Design

TMol now uses one chemistry-generation pipeline and four topology routes:

| Route | Recognition | Added semantics |
| --- | --- | --- |
| protein | CCD peptide-linking type plus a bonded `N-CA-C` path | canonical-parent lower/upper connections, backbone torsions, unchanged backbone atom parameters, and Dunbrack base identity |
| nucleic acid | CCD RNA/DNA-linking type plus a bonded phosphate/sugar backbone path | the corresponding canonical-parent polymer contract |
| carbohydrate | CCD saccharide type plus the explicit inter-component bond graph | graph-derived `down`, `up`, and branch connections |
| general | everything else | generated ligand chemistry and explicitly named covalent connections |

There is no PTM or sugar residue-name catalog. Unknown components are prepared
through the existing SMILES → OpenBabel mol2 → atom typing/charge → residue-type
pipeline. CCD parent metadata supplies canonical polymer identity only where the
input topology confirms the claimed backbone. For example, `SEP` is generated
as modified chemistry, then inherits the unchanged `SER` backbone and
Dunbrack identity.

Explicit inter-residue bonds remain the source of truth. A protein attachment
such as SER `OG` receives a generic connection and its displaced hydrogen is
made virtual and score-inert. For a glycan, a breadth-first traversal rooted at
the protein or non-sugar anchor assigns `down`, the first child `up`, and
additional children deterministic branch names. The connection-capable sugar
clone uses a private internal I/O class so arbitrary graph topology is not
mistaken for sequence adjacency; export restores the deposited `name3`.

## Supported operations

The generated systems can be:

- imported from mmCIF/PDB bond tables and exported with the bonds preserved;
- scored with beta2016 on CPU or CUDA with finite coordinate gradients;
- repacked for canonical and modified protein residues using the inherited
  parent Dunbrack model; and
- Cartesian-minimized as one covalently connected system.

Carbohydrates deliberately use the current input conformation during discrete
packing. Cartesian minimization can optimize them, and their generated
cartbonded, genbonded, electrostatic, LJ/LK, and hydrogen-bond parameters are
active, but no residue-name rotamer or torsion library is introduced.

## Validation inputs

The compact checked-in fixtures retain coordinates, atom names, chemistry, and
tested bonds from these PDB entries:

| Source | Content |
| --- | --- |
| [3U3Z](https://www.rcsb.org/structure/3U3Z), [7BA9](https://www.rcsb.org/structure/7BA9) | phosphorylated peptide fragments |
| [6L1F](https://www.rcsb.org/structure/6L1F), [2H6N](https://www.rcsb.org/structure/2H6N), [4IGQ](https://www.rcsb.org/structure/4IGQ) | methylated lysine peptide fragments |
| [5VVU](https://www.rcsb.org/structure/5VVU) | SER-OG–NAG-C1 O-glycopeptide |

Tests require component/parent identity, explicit connection topology, finite
scores and gradients before and after packing, non-increasing Cartesian
minimization energy, and import/export/import topology preservation on CPU and
CUDA.

## Rosetta parity and intentional differences

The canonical protein portion uses the same parent atom types, charges,
backbone torsions, and Dunbrack identity as the unmodified residue. Ion support
in the next stack layer uses Rosetta fa_standard Mg/Ca/Zn charge and LJ/LK
numbers. Explicit connection bonds are excluded from ordinary nonbonded count
pairs in the same spirit as Rosetta residue connections.

Exact total-score equality is not expected for generated chemistry. TMol's
general ligand path uses OpenBabel MMFF94 partial charges and generated bonded
parameters, whereas Rosetta PTM patches and carbohydrate residue types carry
curated per-component charges and statistical carbohydrate terms. These are
different parameterizations, not a numerical implementation discrepancy.
For example, generated phosphates use Rosetta's general-ligand `PG3`/`OG2`
atom types instead of silently substituting the curated PTM-patch
`Phos`/`OOC` types. Standard terminus charge and count-pair deltas still come
from the canonical polymer parent.
Parity tests therefore compare shared force-field parameters and invariants;
workflow tests compare finite differentiability and monotonic refinement.

Rosetta's `sugar_bb`, linkage-conformer statistics, carbohydrate ring movers,
and carbohydrate-specific rotamer libraries are intentionally absent. Adding
them would require a statistical carbohydrate model, which conflicts with the
current requirement to generate arbitrary sugars without a sugar library.
The clean extension point is a future optional topology-derived conformer
sampler, not residue-name conditionals in import or scoring.
