# AtomWorks reuse audit

**Reuse AtomWorks for shared structure input and chemistry metadata.** The best
immediate integration is to pass its completed AtomArray directly to tmol,
without reading or completing the CIF a second time. The tmol branch now reads
AtomWorks' `chem_comp_type` and `is_polymer` annotations. An explicit
`chem_comp_types` mapping takes precedence; contradictory annotations for one
component name raise before preparation.

This audit used the local `atomworks-dev` checkout at
`a1bda7edfcf325bc140091889b9745220adb5eba`. The improvements are on the separate
local branch `review/tmol-pr503-shared-chemistry`, commit `4762d7e5`, in
`/mnt/home/kdidi/projects/atomworks-tmol-pr503-review`.

## Overlap and recommended ownership

| Area | Existing overlap | Decision |
|---|---|---|
| Missing atoms and residues | tmol's `_cif.py` duplicates AtomWorks template insertion and NaN representation | Consume completed AtomArrays now. Replace the default reader only after identifier, alternate-location and authority contracts are explicit. |
| CIF component definitions | Both read `chem_comp_atom` / `chem_comp_bond` and resolve dictionary templates | AtomWorks should own shared file chemistry. Its scope/cache correctness fixes are prerequisites for reuse across files. |
| Polymer identity | tmol rereads `_chem_comp` and creates a private entity flag; AtomWorks already supplies type and polymer annotations | tmol now consumes the existing annotations. Capping and reconstruction of tmol residue types remain in tmol. |
| Leaving atoms and bond sanitation | Both distinguish unresolved atoms from atoms displaced by covalent attachment | Share generic chemistry in AtomWorks; keep tmol's parameter-generation caps and terminal patches in tmol. The integration exposed and fixed two capping/completion bugs. |
| RDKit conversion | tmol wraps Biotite `to_mol` and restores source Kekulé/subtype information; AtomWorks has its own converter | Candidate for consolidation after parity tests cover atom maps, explicit charge, aromatic orders, NaNs and stereochemistry. No untested converter swap. |
| Protonation / carboxylate correction | AtomWorks contains logic explicitly ported/aligned from tmol | Candidate for a shared chemistry API. AtomWorks' hydrogen placement and tmol's MMFF94 charge/conformer generation have different outputs; they are not interchangeable preparation pipelines. |
| Histidine identity | AtomWorks ports tmol's tautomer-resolution rules | Share a tested chemical contract. Retain tmol's tensor/native implementation for batched CPU/GPU work. |
| Scoring, parameter databases, rotamers and fold trees | These implement tmol-specific numerical and sampling contracts | Retain in tmol. Parser reuse does not accelerate its scoring kernels. |

## Fixes made while testing the integration

AtomWorks now isolates temporary CCD entries across threads, asyncio tasks and
nested scopes, with copies at public ownership boundaries. Standard residue
codes overridden by a file cannot reuse stale global cached chemistry.
A bounded per-scope cache restores reuse without retaining other files' data.
Component classification avoids copying complete templates; dictionary-code
lookup avoids rebuilding a large unchanged set. Leaving-group traversal no
longer allocates a NetworkX subgraph and BFS tree for each candidate atom.

Three covalent fixtures failed because AtomWorks passed the generic entity type
`polymer` to a polymer-subtype enum. It now retains the inferred chemistry when
`entity_poly` is absent. Dependency metadata was also corrected: the core I/O
requires jaxtyping and NetworkX, and its existing import guard requires Biotite
**1.6.0**, not arbitrary newer versions. A fresh core-only install parsed 1UBQ
without PyTorch installed.

On the tmol side, full free termini exposed a cap-generation error: adding an
amide cap while retaining the acid's terminal hydroxyl produced valence five
at carbon. Capping now replaces that hydroxyl while preserving sidechain acids.
Completing either missing phosphate oxygen now respects the input's existing
double bond rather than creating a second one.

## Measurements and compatibility

All 19 AtomWorks reader fixtures now parse. Across the 16 that parsed before,
seven warm uninstrumented samples per fixture gave a median **1.23×** speed
ratio. Full atom-coordinate and bond inventories matched for every previously
passing fixture. Examples: 1UBQ **79.49 → 60.58 ms**; beta peptide 3c3g
**52.95 → 39.36 ms**; ACE–ALA–NH2 **8.33 → 5.04 ms**. Raw paired measurements
and tests are recorded in the AtomWorks branch's `review/tmol-pr503/` directory.

These gains compare AtomWorks with itself in the same environment. Full
AtomWorks parsing remains more expensive than tmol's narrower reader on these
small inputs; it performs more completion and sanitation. Reusing an AtomArray
already parsed by an upstream application avoids the entire duplicate parse.

The tmol integration matrix exercises the 19 PR chemistry fixtures through
preparation, construction, finite scoring/coordinate gradients and rotamer
construction. CPU: **19/19 pass**; H200 CUDA: **19/19 pass**. The expanded
chemistry suite has **216 passes and 8 pre-existing score-reference failures**
(the four backbone classes on both devices). Those failures remain tracked in
`FOLLOWUP.md`; they were not hidden or refreshed to force a pass.
These checks do not independently validate fitted force-field parameters or
establish coverage of every possible chemistry/topology/batch combination.

## Why the default reader has not been replaced yet

AtomWorks selects label identifiers and clears CIF insertion codes; tmol's
current reader preserves author identifiers. AtomWorks selects alternate
locations per chain, while tmol currently inherits Biotite's selection. Full
AtomWorks completion can insert entirely unresolved sequence residues and
terminal/leaving atoms that tmol's current reader omits. Its default parser also
removes waters/crystallization aids and builds assemblies, so those defaults
must be overridden for a matching tmol input route.

A strict `use_ccd=False` contract also needs a shared explicit dictionary policy:
AtomWorks currently supplements file declarations from the bundled CCD.
Switching the default reader before resolving these differences would silently
change the user's structure. The current integration uses explicit parser
settings and consumes the resulting chemical annotations; it does not perform
a second CIF completion or infer an authority policy from installed packages.

See [the executable example](atomworks_example.py). Use the reviewed local
AtomWorks branch for the fixes above. No remote review comments were posted.
