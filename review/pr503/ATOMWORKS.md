# AtomWorks reuse audit

**Reuse AtomWorks for shared structure input and chemistry metadata.** The best
immediate integration is to pass its completed AtomArray directly to tmol,
without reading or completing the CIF a second time. The tmol branch now reads
AtomWorks' `chem_comp_type` and `is_polymer` annotations. An explicit
`chem_comp_types` mapping takes precedence; contradictory annotations for one
component name raise before preparation.

This audit used the local `atomworks-dev` checkout at
`a1bda7edfcf325bc140091889b9745220adb5eba`. The improvements are on the separate
local branch `review/tmol-pr503-shared-chemistry`, commit `0e4ffe8f`, in
`/mnt/home/kdidi/projects/atomworks-tmol-pr503-review`.

## Overlap and recommended ownership

| Area | Existing overlap | Decision |
|---|---|---|
| Missing atoms and residues | tmol's `_cif.py` duplicates AtomWorks template insertion and NaN representation | Consume completed AtomArrays now. Replace the default reader only after identifier, alternate-location and authority contracts are explicit. |
| CIF component definitions | Both read `chem_comp_atom` / `chem_comp_bond` and resolve dictionary templates | AtomWorks should own shared file chemistry. Its scope/cache correctness fixes are prerequisites for reuse across files. |
| Polymer identity | tmol rereads `_chem_comp` and creates a private entity flag; AtomWorks already supplies type and polymer annotations | tmol now consumes the existing annotations. Capping and reconstruction of tmol residue types remain in tmol. |
| Leaving atoms and bond sanitation | Both distinguish unresolved atoms from atoms displaced by covalent attachment | Share generic chemistry in AtomWorks; keep tmol's parameter-generation caps and terminal patches in tmol. The integration exposed and fixed two capping/completion bugs. |
| RDKit conversion | tmol wraps Biotite `to_mol` and restores source Kekulé/subtype information; AtomWorks has its own converter | AtomWorks conversion is now faster and avoids copying unused annotations. Retain tmol's source-typing layer until a shared converter explicitly preserves its Kekulé/subtype/atom-map contract. |
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

## Conversion and shared-rule follow-up

The RDKit converter now reads columns directly and copies only retained output
annotations. Input/output independence, hydrogen policies, charges, aromaticity,
stereochemistry and coordinate inventories pass. It also honors explicit
`set_coord=False`, and automatic coordinate selection requires all values to be
finite. Explicit `True` still preserves NaNs. The carboxylate correction copied
between the projects accepted NaN geometry as evidence for changing bond orders;
both implementations now reject that local correction. AtomWorks also resets
both oxygen charges consistently after assigning C(=O)[O-].

All **71 affected AtomWorks tests** and **43 tmol tests** pass. Replaying the old
methods reproduces 12 AtomWorks and seven tmol failures in the new regressions.
The 19-fixture AtomWorks input/preparation/scoring/gradient/rotamer CPU matrix
passes again after these changes. This is additional CPU validation; the earlier
GPU matrix is not presented as a rerun of this follow-up.

Seven alternating-order warm pairs give **1.23–1.47×** faster conversion for
ALA, NAD and NAG across keep/remove/infer hydrogen policies, with matching full
molecule inventories. A synthetic 3,000-atom input takes **30.41→20.94 ms**;
traced peak Python allocations fall **512,285→402,173 bytes**. With 32 unused
U128 annotation columns, it takes **109.91→21.31 ms**, and traced peak allocation
falls **49.67→0.40 MB**. That is an explicit metadata stress case, not a typical
ligand or total native/process memory claim. Converter gains are separate from
the parser measurements above and cannot be multiplied into an end-to-end
speedup. See `results/atomworks-rdkit-conversion.json` and
`results/atomworks-conversion-tests.json`; the executable profiler lives on the
local AtomWorks branch at `review/tmol-pr503/profile_rdkit_conversion.py`.

The duplicated Dimorphite rule engine and pre-protonation corrections are good
candidates for one shared AtomWorks API. The assembled hydrogen-placement and
tmol charge/conformer-generation pipelines still need separate entry points.
Package consolidation should pin a released shared contract instead of silently
selecting different chemistry based on whether AtomWorks happens to be installed.

## Dimorphite compatibility before consolidation

The two vendored engines also differ behaviorally. Before follow-up, AtomWorks
lost heavy-atom map identities in **12 of 21** mapped molecule/pH cases; organic
azidoethane at pH 2 also differed in charge state. Its direct molecule API
deduplicated through an unordered SMILES set, losing atom properties and stable
variant ordering. Tmol's neutralization provenance and azide exception, plus
stable product retention, are now implemented on the AtomWorks branch.

The seven classes are acetate, ethylammonium, azidoethane, nitroethane, cysteine,
histidine and phosphoserine, each at pH 2, 7.4 and 12 with precision 0.1. All 21
ordered chemical-state and heavy-map inventories now match tmol; **34 AtomWorks
identity/protonation tests pass**. This is chemical compatibility evidence, not
an independent validation of pKa predictions. AtomWorks' unchanged pattern-loader
still recompiles SMARTS per call; tmol caches by pH without a size bound. A shared
bounded cache with ownership-safe results is a further simplification opportunity.

See [check_dimorphite_contract.py](check_dimorphite_contract.py),
`results/dimorphite-contract.json` (AtomWorks `cdda3c07`) and
`results/dimorphite-contract-after.json` (`44641189`), both compared against
tmol's engine from `6b7071d54` in the same RDKit 2026.3.6 CPU environment.

## Fixed rule compilation and remaining rule-file difference

Both engines now compile one fixed rule set and derive states for each request.
Public loader results own independent query molecules and nested state lists;
the direct molecule API borrows private queries for read-only matching. The
cache stores neither input molecules nor an expanding history of pH values.
All 40 affected AtomWorks tests and six tmol cache regressions pass.

The paired direct-API benchmark improves AtomWorks **1.493→0.515 ms per molecule
(2.90×)**. Tmol's previously cached warm calls take **0.509→0.527 ms (3.5% slower)**,
while retained traced Python allocations across 500 distinct pH requests fall
**16.10→0.065 MB**. AtomWorks adds about 10 KB of retained traced Python data for
its fixed cache (34→44 KB). These are seven alternating-order warm pairs over
21 molecule/pH cases, excluding imports and full parameter/conformer generation.
Full ordered chemical/map inventories and all rules under 15 pH-range/precision
settings match each project's unchanged baseline. Tracemalloc does not include
native RDKit allocations. See `profile_dimorphite_cache.py` and
`results/{atomworks,tmol}-dimorphite-cache.json`.

The expanded compatibility audit is now 33 molecule/pH cases. Identity is
preserved in every case, but four ordered chemical-state inventories differ:
enamine and vinylogous amide at pH 2 and 7.4. Tmol includes a distinct `Enamine`
rule with pKa 1 ± 1 ahead of the generic amine rule; AtomWorks does not. The
benchmark preserves that difference. Which rule set becomes the shared default,
and whether tmol needs an explicit versioned rule profile, remain open scientific/
API questions. Neither engine should be selected implicitly by import availability.
See `results/dimorphite-contract-expanded.json`.

The 19-fixture CPU input/preparation/construction/scoring/gradient/rotamer matrix
passes again after the rule-cache change. See
`results/atomworks-rule-cache-matrix.json`. This is stage validation, not proof
of correct Cartesian minimization or of the missing attachment bond potentials.
