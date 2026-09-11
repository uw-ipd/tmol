# Review of tmol PR #503

Reviewed PR: https://github.com/uw-ipd/tmol/pull/503  
Author branch: `dimaio/noncanonicals_through_ligand_pipeline`  
Initial pinned head: `c03c1e745f3bc655948ea12dac44d6c74620358f`\
Updated head: `0f4c3bc426bca78e8681f0b730fa23c3e26ef261`\
Diff merge base: `08d82941b6b5bfcd405303f8730b36b54dcfd28a`  
Improvement branch: `review/pr503-chemistry-efficiency`  
Review date: 2026-09-11

The six-file update has been reviewed separately. Comments 1–46 retain their
original `c03c1e745` anchors; comments 47–52 address the updated head. The
fold-forest expectations and HYP count are now corrected upstream. See
[FOLLOWUP.md](FOLLOWUP.md) for reconciliation and validation details.

**Recommendation: request changes.** The chemistry expansion is substantial, but the exact submitted tree cannot import its I/O package. After supplying the missing module, targeted checks expose batch identity errors, lost/misassigned group torsions, residue identity errors, and a mixed D/L disulfide score that depends on residue ordering. The passing chemistry fixtures do not cover these cases.

This document and the comments below are drafts for the user; no review or comments were posted to Frank's PR. Links in the comments point to the pinned upstream commit, so their line numbers remain valid after changes to this branch.

## What the PR implements

- **Input and chemical identity:** a CIF reader adds unresolved atoms at NaN; preparation classifies polymer residues from declared entities, connections, and chemistry. Noncanonical backbones are capped for parameter generation, then reconstructed as polymer types with generated terminal patches. Conjugation patches represent attached glycans and ligands as connected blocks.
- **D amino acids:** generated chemical/charge/cartbonded/reference records, mirrored Dunbrack libraries, mirrored backbone grids, and D-aware disulfide parameters. Optional symmetric glycine tables make full mirror-image comparisons possible.
- **Scoring:** hybrid generic/Rosetta typing partitions torsion ownership; cartbonded enumerates connection-spanning impropers instead of treating them as bonded paths. Nucleic-acid references can be borrowed from similar bases.
- **Packing and kinematics:** chemical fold-tree edges, more general single-residue trees, chemistry-derived sampling references, group conformer enumeration and energy collapse onto one representative.

The core design is useful: explicit reference fields are more extensible than residue-name inference; preserving unresolved atoms prevents accidental truncation of chemical identities; and a covalent group needs correlated conformers. The main weaknesses are identity/scope assumptions that hold for one fixture but fail across poses, repeated residues, or reused samplers, and incomplete enforcement of the advertised sampling budget.

The initial 213-file inventory is in [upstream-files.tsv](upstream-files.tsv); the updated 214-file inventory is in [upstream-files-0f4c3bc42.tsv](upstream-files-0f4c3bc42.tsv). Static review concentrated on the new preparation, CIF completion, database mirroring/caching, group-packing, fold-forest, and scoring changes. Generated databases and fixture coordinates were assessed through their generators, schema, provenance notes, and executable checks; this is not an independent refit or scientific validation of those parameters.

## Branch improvements

1. Supply the absent `_cyclic_search.py` using the documented API: preserve explicit closures, infer first/last connections within each contiguous polymer run, exclude missing atoms/nonpolymers/padding/single residues, and keep all operations on the tensor device. **This is a new implementation inferred from the committed callers and documentation, not Frank's missing original file.** It is independently tested and must be reconciled if he supplies that file.
2. Replace whole-structure masks with residue boundaries, including insertion codes; select representative residues lazily; prevent conjugation preparation from merging equally numbered residues across chains.
3. Cache CCD templates once per component **within each completion call**, concatenate arrays once, and construct/remap bond tables in batches. No global mutable CCD cache is introduced.
4. Identify hydrogen atoms by element when checking whether a template covers the observed heavy atoms; names such as `1H` no longer block completion.
5. Retain original chi indices through budgeting, separating the anchor library's multiplicity from the child's chi numbering. Update the conformer count by division while freezing chi instead of recomputing the product each iteration.
6. Key anchor library entries by `(pose, block)` and scope group kinforest caches to their `PackedBlockTypes`, so equal numeric type indices in another database cannot reuse a different chemistry's tree.
7. Reuse invariant node/scan/generation tensors and fold group conformers in bounded batches. The follow-up also bounds shape caches and verifies scalar/batched CPU/CUDA parity; timing and temporary-memory tradeoffs are recorded in [FOLLOWUP.md](FOLLOWUP.md).
8. Validate equal lockstep counts before native scoring; read counts to the host once and upload one completed group-ID tensor. Preserve the exclusive-end offset when collapsing groups with trailing zero-rotamer blocks.
9. Frame content-hash values to avoid metadata collisions and hash tensor memory directly instead of allocating `.tobytes()` copies.
10. Remove a duplicated chi-pruning pass from `remove_atom()`.
11. Use the actual prepared parameter database in group-packing tests, and update two stale fold-tree assertions to require a split at each branch point. The latter changes test expectations to match the existing documented kinematics invariant; no fold-tree algorithm changes.
12. Keep explicitly prepared ligand fragments eligible for block-type selection. Remove their cut bonds from the intermediate AtomArray bond table so the existing fragment mapping installs those connections once, while retaining bonds within fragments and between original residues.

Subsequent follow-up changes add the Rosetta mixed-chirality disulfide distribution with independent energy/gradient checks, repair terminal-cap and nucleotide proton geometry, and consume AtomWorks chemistry annotations. See [FOLLOWUP.md](FOLLOWUP.md) for current completion gates and [ATOMWORKS.md](ATOMWORKS.md) for the reuse and performance audit. Explicit task limits and sampler cache settings are now enforced; aggregate/default budget policy remains open (see [BUDGETS.md](BUDGETS.md)).

## Measured performance

`benchmark_cif_completion.py` compares the original and new `_inserted()` in one Python process. Each residue has two observed carbon atoms and one missing carbon; every atom, annotation, coordinate (including NaNs), and bond is checked for equality. Five timed runs per size follow an untimed call; numbers are medians, excluding imports and fixture construction.

| Residues | Upstream insertion | Branch insertion | Speedup |
|---:|---:|---:|---:|
| 100 | 6.46 ms | 3.26 ms | 1.98× |
| 500 | 39.66 ms | 16.51 ms | 2.40× |
| 2,000 | 266.93 ms | 67.98 ms | 3.93× |

This measures CIF missing-atom insertion, **not end-to-end packing/scoring acceleration**. Host load can affect timings. The script is included for rerunning on larger/more realistic structures. The run-local template cache and lazy representative selection are additional changes not isolated by this microbenchmark.

## Review questions for Frank

1. **Delivery and validation:** Can you add the missing cyclic-search file and enable CI on a fresh checkout? No GitHub checks were reported for the pinned head when reviewed. Which Python/PyTorch/Biotite/RDKit/OpenBabel versions generated the baselines?
2. **Scientific partitioning:** What independent Rosetta/Hahnbeom reference validates the cartbonded correction and generic/Rosetta boundary, beyond regenerated tmol goldens? Can we separate these global score changes from adding new residue classes, so canonical score shifts are auditable?
3. **Mixed chirality:** What should the S–S torsion distribution be for L–D versus D–L? It must be invariant to input residue ordering, and full mirror tests should cover LL↔DD and LD↔DL, including gradients.
4. **Budget semantics:** Is `set_chi_sample_budget()` meant to bound library rotamers, sampled heavy chi, proton combinations, group conformers, total block rotamers, or pairwise memory? What happens when the anchor library alone exceeds the limit, when all child chi freeze, or when required proton samples exceed it?
5. **Group constraints:** What is the contract for a partially disabled group, multiple polymer anchors joined by a conjugate, free oligosaccharides, and groups with cycles? Should unsupported topologies fail clearly or remain frozen? Which bond closes a non-tree cycle during sampling/minimization?
6. **Chemical authority:** How should custom residue names with declared bonds/stereochemistry but missing coordinates work under `use_ccd=False`? Currently coordinate completion can still require a CCD component. What explicit input describes ambiguous polymer ends instead of selecting the conventional backbone heuristically?
7. **Sampling versus scoring:** The graph matcher borrows sidechain references for sampling, while scoring ownership is independently inferred. Which tests guarantee that changing a sampling reference cannot silently change the scored potential or leave a rotatable bond unconstrained? How were graph-match thresholds and unknown-base averages chosen?
8. **Lifecycle and reproducibility:** Are samplers intended to be reusable across databases/tasks? Are generated chemistry and cache entries bounded in a long-running service? Can users persist the seed, generated parameters, chosen references, warnings, and exact input authority in a build report?
9. **API and migration:** The PR body says `process_ligands=True`, but the implementation uses `prepare_ligands=True`. Is removing `sample_proton_chi` intentional? Can public docs show the supported CIF, AtomArray, SMILES/mol2, cyclic, and group-packing entry points with executable examples?
10. **Scale and release:** What canonical-protein load/score/pack baseline is acceptable after eagerly adding mirrored tables and invoking group discovery in scoring setup? Can the PR be split along its existing commit sequence into import/preparation, score corrections, and group packing?
11. **Shared AtomWorks chemistry:** Can AtomWorks own CIF completion and chemical annotations, with tmol consuming a completed AtomArray? Before replacing the default reader, which author/label identifiers, insertion codes, alternate locations, unresolved residues and CCD authority must be preserved? Can the copied Dimorphite and pre-protonation rules use one versioned shared API, while tmol keeps parameter generation and numerical scoring?

12. **Shared rule provenance:** Tmol adds an enamine SMARTS rule with pKa 1 ± 1 that AtomWorks does not contain. What evidence supports its scope and values, and should it be a shared default or an explicit preparation profile? Direct replacement currently changes charge states for an enamine and a vinylogous amide at pH 2 and 7.4. Which complete rule inventory and model version should exported parameters record?

13. **Attachment charge policy:** Should local charge changes preserve the curated residue baseline and add a connected-versus-disconnected MMFF correction, or replace the whole capped group's charges? The former preserves remote backbone parameters but is a model choice requiring validation. Complete capped biotin changes formal charge by −1, with heavy-atom-plus-hydrogen deltas on both LYS and BTN, including LYS CE. Applying only a hydrogen-count patch cannot represent this. Which atom-type, torsion-ownership and proton-construction changes must accompany the charge model?

## Suggested inline comments

Each item gives an upstream location, suggested comment, and what this branch does about it. “Reproduced” means exercised against PR code with only the missing import dependency supplied, not merely inferred from source.

### 1. P0 — missing module prevents test collection

[tmol/io/details/__init__.py:10](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/io/details/__init__.py#L10)

> Could you commit `_cyclic_search.py`? This import points to a file absent from the PR tree. Both CPU and H200 pytest runs fail while loading conftest with `ModuleNotFoundError`, before collecting any tests. A clean-checkout import/test smoke check would catch this.

Reproduced on the exact head. Branch supplies a separately tested implementation; original baseline failure is retained.

### 2. P1 — mixed D/L disulfide energy depends on residue ordering

[tmol/score/disulfide/potentials/potentials.hh:122](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/score/disulfide/potentials/potentials.hh#L122) and the derivative path at line 312.

> Could the S–S potential use an explicitly defined pair-chirality model? It currently reads only `params1.dss_*`. Reversing the same mixed D/L pair changes the first parameter row but not the physical structure. In a two-cysteine reproduction using 6DMZ coordinates, reordering blocks changes the disulfide-only score from −0.0337973 to 0.5349466. Please add permutation and mixed-chirality gradient tests; whole-L versus whole-D mirror tests do not detect this.

Reproduced. [reproduce_disulfide_order.py](reproduce_disulfide_order.py) holds geometry fixed while changing block order; it is an invariance test, not a relaxed D/L structure. The follow-up implements Rosetta's mixed-chirality distribution and shared derivatives, with independent LL/DD/LD/DL energy/gradient, permutation and reflection checks on CPU/CUDA. Broader whole-pose and packing parity remains open; see [FOLLOWUP.md](FOLLOWUP.md).

### 3. P1 — group anchors collide across poses

[tmol/pack/rotamer/_conjugated_chi_sampler.py:504](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_chi_sampler.py#L504), also the lookup at line 116.

> Could this dictionary be keyed by `(group.pose, group.anchor)`? Two poses commonly have an anchor at the same block index. The second currently overwrites the first's chi atoms/values, which can substitute the wrong chemistry or backbone-dependent library samples into the first pose.

Reproduced with different library values/types at block 0 in two poses. Fixed with matching producer and consumer keys.

### 4. P1 — anchor chi numbering removes child linkage samples

[tmol/pack/rotamer/_conjugated_groups.py:89](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_groups.py#L89)

> The budget's `n_library_chi` both estimates library size and removes samples numbered `chi1..chiN`. Here those samples belong to attached blocks, not the anchor. Can these two responsibilities be separated? A lysine anchor's chi count should multiply a child's sampling cost without deleting that child's independently numbered linkage torsions.

Reproduced. Fixed by a separate library multiplier in the indexed budget helper.

### 5. P1 — retained chi can be assigned to the wrong attached residue

[tmol/pack/rotamer/_conjugated_groups.py:97](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_groups.py#L97)

> Could the budget return original input indices along with samples? `chi_dihedral` is only unique within a residue. If one child's `chi1` freezes while another child's `chi1` survives, this walk can attach the surviving sample to the earlier child and still satisfy the final assertion.

Fixed by carrying indices directly, eliminating the name-based reconstruction. Regression uses repeated names with different values and depths.

### 6. P1 — unsafe native pair enumeration before count validation

[tmol/pose/_conjugated_groups.py:144](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pose/_conjugated_groups.py#L144)

> Could we validate equal sampled-member counts here, before exposing a lockstep ID to native scorers? `_calculate_packer_energies()` scores first and only then calls the validating collapse function. In `sphere_overlap.impl.hh`, count uses `nr1` while fill writes `min(nr1,nr2)`; other bonded dispatchers derive partner indices directly. A mismatch can therefore reach uninitialized pairs or invalid indices before the intended Python error.

Python failure-to-reject reproduced; unsafe native execution deliberately not required for the test. Fixed at ID construction, also removing per-block device scalar reads.

### 7. P1 — zero-rotamer trailing blocks index beyond the remap

[tmol/pack/rotamer/_conjugated_groups.py:280](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_groups.py#L280)

> Could the prefix map include its exclusive endpoint? For counts `[2,2,0]`, valid offsets are `[0,2,4]`, but `orig_to_compact` only has four entries. Collapsing a group in the first two blocks raises an out-of-range error on the trailing zero-count block. Please cover jagged/empty trailing slots and preserve negative sentinels too.

Reproduced. Fixed with an N+1 prefix map and explicit negative-sentinel handling.

### 8. P1 — task sampling budget is currently a no-op

[tmol/pack/_packer_task.py:344](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/_packer_task.py#L344)

> Where are these values propagated to the samplers? `SetPackerTask.from_packer_task()` does not copy them, and the samplers read their own default fields. Setting a smaller task budget currently has no consumer. Please test a material reduction in generated rotamers after calling this method, including setting it before and after adding samplers.

Reproduced and addressed: explicit limits survive task conversion; private NA/OptH sampler views avoid mutating reusable caller objects; actual group-library cardinality and native Dunbrack count checks enforce bounds before final sample allocation. CPU/CUDA tests cover setting limits before/after adding samplers and repeated reuse. Aggregate/default budget policy remains open; see [BUDGETS.md](BUDGETS.md).

### 9. P1 — group-tree cache aliases unrelated type tables

[tmol/pack/rotamer/_conjugated_chi_sampler.py:192](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_chi_sampler.py#L192)

> Could this cache live on `PackedBlockTypes` or include its identity? `types` are indices local to a type table. Reusing one sampler with another prepared database whose group also has indices `(0,1)` returns the first chemistry's tree and offsets.

Reproduced by reusing a sampler with two distinct type tables. Fixed by scoping the cache to the owning `PackedBlockTypes`.

### 10. P1 — CIF completion merges insertion-coded residues

[tmol/io/_cif.py:334](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/io/_cif.py#L334)

> Can we slice using `get_residue_starts()` boundaries instead of rebuilding a mask from name, chain and residue number? Residues 42A and 42B share those three fields. Their combined atom-name set can make one residue appear complete because the other has its missing atom. The mask also scans the full structure once per residue.

Reproduced. Fixed in completion and representative selection. Cross-residue bond detection now also distinguishes insertion codes.

### 11. P1 — canonical conjugation preparation merges chains

[tmol/ligand/_preparation.py:838](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_preparation.py#L838)

> Could this select one residue instance including chain and insertion code? It currently selects all atoms with the first residue number after filtering only by residue name. Equally numbered lysines from multiple chains can be combined before conjugated chemistry is computed; the caller then suppresses chemistry errors and may fall back to incorrect hydrogen counts.

Fixed with a residue-boundary slice; regression verifies that chains A and B each containing residue 1 remain separate.

### 12. P2 — hydrogen naming breaks template coverage

[tmol/io/_cif.py:192](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/io/_cif.py#L192)

> Can heavy atoms be identified from the `element` annotation? `not name.startswith("H")` treats PDB-style names such as `1H` as heavy atoms, then rejects a valid heavy-atom template because it lacks that hydrogen name. This prevents unresolved heavy atoms from being added.

Reproduced; fixed with element-based matching.

### 13. P2 — content hash concatenation is ambiguous

[tmol/database/scoring/_content_hash.py:33](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/database/scoring/_content_hash.py#L33)

> Could each value be type/length framed before hashing? `(1.0,23.0)` and `(1.02,3.0)` both currently feed `1.023.0` to SHA256. Nested sequence boundaries are also lost. This is a serialization collision, not a SHA256 collision, and can alias different grid metadata in a resolver cache. A memoryview would also avoid copying every tensor into a Python bytes object.

Fixed; tests cover metadata boundaries, tensor contents, strides, shape, dtype and empty tensors.

### 14. P2 — sampling caches ignore the new budget configuration

[tmol/pack/rotamer/_opth_sampler.py:301](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_opth_sampler.py#L301) and [tmol/pack/rotamer/_na_chi_sampler.py:103](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_na_chi_sampler.py#L103).

> These annotations now depend on sampler budget fields, but cache validity is still only `hasattr(rt, ...)`. Could configuration-dependent samples be cached by configuration or computed at sampling time? Otherwise whichever sampler annotates a shared residue first determines subsequent tasks' sample sets.

Fixed and tested on CPU/CUDA: cache validity includes sampler settings and relevant chemistry inputs; one most-recent table is retained on each RT/PBT. Explicit task overrides use private sampler views. Concurrent mutation remains outside the supported contract.

### 15. P2 — repeated allocation in the group conformer loop

[tmol/pack/rotamer/_conjugated_chi_sampler.py:403](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_chi_sampler.py#L403)

> Can nodes/scans/generations and the atom-order conversion be constructed once outside this loop? They are invariant across conformers, but currently allocate and transfer on every iteration. A later improvement could combine conformers into one kinforest so forward/inverse kinematics and per-member scalar reads are batched too.

Addressed with invariant allocation hoisting and bounded forward/inverse-kinematics batches. Final scalar parity, chunk-boundary, multi-pose reuse and full group packing tests pass on CPU/CUDA. Paired full-rotamer construction latency falls 6–11% on CPU and 21–33% on CUDA, with a measured temporary-memory increase; see [FOLLOWUP.md](FOLLOWUP.md). This is not an end-to-end annealing/packing speedup claim.

### 16. P2 — quadratic CIF array assembly

[tmol/io/_cif.py:395](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/io/_cif.py#L395)

> Could this use `struc.concatenate(pieces)` and build the final bond array once? Repeated `combined + piece` recopies the growing structure and its annotations/bonds. The reader also repeats CCD lookup for every occurrence of a component. A per-read template map plus single assembly cuts the insertion benchmark by 2–4× while preserving output.

Implemented and benchmarked above.

### 17. Design — make unsupported cyclic groups explicit

[tmol/pose/_conjugated_groups.py:97](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pose/_conjugated_groups.py#L97)

> When this skips a cycle-closing bond, what keeps that bond closed during group sampling? The returned links are a spanning tree, so changing a tree torsion need not preserve the omitted edge. If cyclic/multi-anchor conjugates are outside scope, could they be rejected or held fixed explicitly, with a documented supported-topology check?

Question requiring a supported-chemistry contract; no unvalidated closure algorithm added.

### 18. Maintainability — share the cartbonded improper enumerator

[tmol/score/cartbonded/potentials/cartbonded_pose_score.impl.hh:535](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/score/cartbonded/potentials/cartbonded_pose_score.impl.hh#L535)

> Could the connection-centred improper enumeration and parameter lookup be a device-inline helper with a scoring callback? The same substantial block appears in whole-pose forward/backward and rotamer forward/backward. Sharing only enumeration would reduce the chance that a future chemistry fix changes one path but not its gradient or packing counterpart.

Implemented in follow-up: one device-inline helper now supplies enumeration and lookup to all four paths. Canonical references, numerical gradients, group packing and mirror checks pass CPU/CUDA (238323); an independent Cartesian peptide/proline reference passes energies and gradients on both devices (238529). This is a refactor with unchanged parameter values.

### 19. Maintainability — remove duplicated atom-removal work

[tmol/database/_patched_chemdb.py:60](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/database/_patched_chemdb.py#L60)

> This repeats the same surviving-torsion set and chi filter from four lines above. Could the duplicate be removed?

Removed.

### 20. Design — bound profile cache lifetime

[tmol/ligand/_rotamer_reference.py:643](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_rotamer_reference.py#L643)

> Could this use a cache owned by the database, a weak-reference-aware identity key, or an immutable content key with bounded storage? The process-global dict keys only on `id(chemical_database)` and library names, does not retain/verify the original object, and never evicts. Repeated preparation can accumulate entries, and Python can reuse an object ID after a database is collected.

Static lifecycle concern; left open. The analogous sampler group-tree cache is fixed on this branch.

### 21. P1 — group packing tests silently discard the prepared database

[tmol/tests/pack/test_conjugated_group_packing.py:92](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/pack/test_conjugated_group_packing.py#L92) and the analogous assignments in the other tests.

> Could these use `ctx.parameter_database` directly? `PoseBuildContext` has no `param_db` field, so `getattr(ctx, "param_db", None) or ParameterDatabase.get_default()` always discards the newly prepared chemistry. The score-versus-packer agreement test then compares two consumers of the wrong database, potentially agreeing while missing the generated charges/parameters.

Fixed in the branch's group-packing tests; the final group run uses the actual prepared parameter database. The standalone example runner also uses that database explicitly.

### 22. P1 — prepared terminal caps cannot be selected into a pose

[tmol/tests/ligand/test_nonstandard_backbones.py:894](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/ligand/test_nonstandard_backbones.py#L894), with selection at [tmol/io/details/_select_from_canonical.py:907](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/io/details/_select_from_canonical.py#L907).

> Could this test build and score both capped-peptide fixtures, beyond checking the generated residue names? `pose_stack_from_cif(..., prepare_ligands=True)` fails for ACE–ALA–NME and ACE–ALA–NH2. With cyclic inference disabled, ACE and NME still have no block-type candidates: they intrinsically lack one polymer connection, but candidate classification only marks termini through patch names. Their base types are consequently absent from the requested terminal slots.

Reproduced on CPU and H200; baseline also fails, including with cyclic inference disabled. Follow-up corrections align cap representation and terminal classification, and preserve amide geometry. ACE/NH2/NME construction, bonds, scoring/gradients, rotamer construction and actual packing pass on CPU/CUDA (237089). The initial failed example logs remain historical evidence.

### 23. P1 — reference score assertions do not reproduce

[tmol/tests/score/test_noncanonical_scoring.py:78](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/score/test_noncanonical_scoring.py#L78).

> Could you document the environment and preparation state behind these reference scores and make them reproducible before relaxing tolerances? All four classes currently fail on both CPU and CUDA in the H200 environment. For the beta-peptide fixture, `fa_ljrep` is approximately 121.26 versus the committed 662.72, which is far beyond a precision discrepancy. Please distinguish changes in generated coordinates/parameters from changes in the scoring kernels.

Recorded without updating goldens. Environment and baseline/candidate comparison are in the validation record; numerical discrepancies alone do not identify which implementation or reference is scientifically correct.

### 24. P2 — the D repacking test checks labels, not geometry

[tmol/tests/score/test_mirror_image_scoring.py:120](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/score/test_mirror_image_scoring.py#L120).

> Could `_chirality()` also check the signed tetrahedral volume around stereocentres using the output coordinates? Reading `properties.polymer.sidechain_chirality` verifies that the type label stays D, but a D-labeled block built with inverted geometry would still pass. The current test also would not establish that the mirrored library was actually used.

Follow-up tests measure signed alpha-centre volumes in every offered D rotamer and in actual packed coordinates, and check that the mirrored library supplies samples on CPU/CUDA (237089). Broader sidechain stereocentre and full mixed-pose mirror coverage remains open.

### 25. P2 — fold-tree tests contradict branch-point splitting

[tmol/tests/kinematics/test_fold_forest.py:182](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/kinematics/test_fold_forest.py#L182), also line 234.

> Could the expected polymer edges be split at the conjugation parent? The builder now emits `0→2` and `2→3` so the chemical edge leaving block 2 has a parent edge ending there. This matches the new validator's documented invariant, but the assertion still expects `0→3`. The mid-chain conjugation test has the same stale expectation.

Both failures reproduced on the baseline. Updated these two expectations; all seven executable fold-forest tests then pass on CPU, with the CUDA-only smoke case skipped locally.

Upstream `0f4c3bc42` now makes the same two expectation corrections.

### 26. P1 — conjugation selection excludes every ligand fragment

[tmol/io/details/_select_from_canonical.py:1110](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/io/details/_select_from_canonical.py#L1110), and the intermediate array at [tmol/ligand/_fragmentation.py:742](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_fragmentation.py#L742).

> Could explicitly prepared ligand fragments be exempt from this candidate filter, and their cut bonds remain owned by the fragment mapping? Every fragment has a non-polymer connection, so this filter leaves no candidate for its residue class. Merely allowing the candidate then routes its original cut bonds through conjugation-patch selection, although `apply_fragment_connections()` separately restores those bonds. The baseline fails 35 fragmentation tests before their restoration/scoring assertions can execute.

Reproduced. The branch permits fragment candidates and removes only bonds crossing newly split pieces of the same original residue from the intermediate bond table. The prepared fragment mapping remains authoritative for cut-bond restoration. All 35 previously failing fragment cases pass in the final 133-case fragment/conjugation run (129 passed, four existing skips).

Upstream `0f4c3bc42` adds a different fragment fallback; comments 47–48 record two independently reproduced limitations of that update. Reconciliation preserves our explicit fragment flag and exact cut-bond ownership, while accepting already-declared connection sites.

### 27. P2 — HYP rotamer count disagrees with the stated sampling model

[tmol/tests/pack/rotamer/test_noncanonical_rotamers.py:266](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/pack/rotamer/test_noncanonical_rotamers.py#L266).

> Could this distinguish borrowed-library cardinality from the additional chi sampling/expansion multiplier? HYP currently produces 18 rotamers on both CPU and CUDA, versus the asserted six. The comment describes two library rotamers times three hydroxyl samples. Please pin whether expanded samples are intended here, then test the library and extra-chi counts separately so a factor-of-three change has a clear diagnosis.

Reproduced on the baseline and initial candidate. Follow-up checks independently identify two unique library states and nine hydroxyl angles (three means with ±20° expansions) on CPU/CUDA. The expected count is now 18 with that separate check. Explicit task overrides and actual group-library bounds now have CPU/CUDA checks. Aggregate/default budget policy remains open.

### 28. P1 — terminal proton sampling rotates an entire generated nucleotide

[tmol/ligand/_polymer_builder.py:893](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_polymer_builder.py#L893).

> Could the nucleotide jump root use the sugar side of the glycosidic torsion, as canonical nucleotides do, instead of the second mainchain atom? For generated 5CM that atom is O5′, which carries a proton chi at a free 5′ terminus. OptH writes that chi into the jump degree of freedom and moves heavy atoms by up to 2.44 Å. Please check heavy-coordinate preservation for every offered proton rotamer, not only finite scores or the selected residue label.

Reproduced in the follow-up branch before the root correction. After correction, the maximum heavy displacement in that diagnostic is 0.0000049 Å. Five modified DNA/RNA fixtures preserve heavy atoms in all offered proton rotamers and actual packing on CPU/CUDA (job 237980). Old score references may encode the damaged geometry and must be reconciled independently.

### 29. P1 — polymer profile caches can reuse a dead database's identity

[tmol/ligand/_polymer_profile.py:169](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_polymer_profile.py#L169), and [nucleotide cache at line 1248](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_polymer_profile.py#L1248)

> Could these caches validate the live database referent, release entries when it dies, and bound retained profiles? An integer `id(chemdb)` can be reused after collection; a later custom database could then inherit the previous database's cap geometry, atom types and reference fields. The alpha and nucleotide caches also retain every computed profile indefinitely. Please cover separate database instances, expired identities and repeated preparation in a long-running process.

Addressed with one bounded weak-identity cache implementation shared by alpha, nucleotide and rotamer-reference profiles. Real database lifetime, configuration separation, eviction, simulated stale-identity and simultaneous-miss tests pass. All 136 affected preparation/scoring cases pass on CPU/CUDA. A 100-database churn measurement retains zero profile entries instead of 300, with identical profile contents; the small lookup cost is recorded in [FOLLOWUP.md](FOLLOWUP.md).

### 30. P1 — linkage sampling breaks internal sugar-ring bonds

[tmol/ligand/_conjugation_patches.py:122](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_conjugation_patches.py#L122), and [group axis handling at line 266](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_chi_sampler.py#L266)

> At an anomeric carbon, this central bond can belong to the sugar ring. Turning it as an independent chi opens the ring, even when every inter-block bond remains intact. In the O-glycan fixture, offered NGA conformers stretch C5–O5 from 1.443 Å to 3.768 Å. Can the torsion instead use the actual inter-residue bond at such sites, with both central atoms resolved across connections? Please check every internal bond, bond angle and stereocentre in offered conformers, in addition to inter-block links.

Reproduced and fixed. Ring attachments now use a bond-spanning torsion; the sampler resolves central atoms in group-wide numbering. The paired O-glycan maximum bond error falls from 2.325 Å to 0.000027 Å on CPU and 0.000033 Å on CUDA, with unchanged rotamer counts and coordinate storage. All 51 group geometry/packing/regression cases pass on CPU/CUDA. AtomWorks input also passes the new geometry checks on both devices and full GPU packing checks.

### 31. P1 — a departing hydrogen's icoor is not necessarily a bonded torsion path

[tmol/ligand/_conjugation_patches.py:121](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_conjugation_patches.py#L121)

> Could these references be checked against the chemical bond graph? Hydrogen icoors may use another hydrogen on the same centre as a reference. For the lysine/asparagine attachments, that can put the first and fourth torsion atoms on the same moving side, with an unbonded first–second pair. Writing a different phi then does not produce the requested dihedral. A valid placement frame alone does not establish a rotatable four-atom path.

Fixed by selecting a bonded heavy central neighbor and a bonded reference on its opposite side, preserving valid existing references. Tests independently resolve all four atoms through the pose, verify the bonded path, and measure each requested angle from the final rotamer coordinates. This is a geometry correction; the generic three-angle grid is not presented as an independently fitted distribution for every chemistry.

### 32. P1 — missing coordinates must not authorize a bond-order rewrite

[tmol/ligand/_structure_to_smiles.py:101](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_structure_to_smiles.py#L101), and the angle test at line 105.

> Can both distance and angle checks require finite, nondegenerate geometry? With a NaN carbon, oxygen or third neighbor, these comparisons can both be false and the code rewrites two single C–O bonds to C(=O)[O-] despite having no geometric evidence. This matters especially now that CIF completion intentionally retains unresolved atoms. Please also check a mixed molecule with one unresolved site and one valid correction site.

Seven regression cases fail when replaying the prior helper methods and pass after the finite-geometry correction. All 18 new geometry and 25 existing ligand-unit tests pass. AtomWorks contains the same copied rule and received the same fix, plus consistent resetting of both oxygen charges. The 19-fixture AtomWorks-to-tmol preparation/scoring/gradient/rotamer matrix passes again on CPU. Geometry checks remain a heuristic for correcting known input encodings, not a replacement for explicit chemical authority.

### 33. P1 — capped polymer residues disappear from conjugated groups

[tmol/pose/_conjugated_groups.py:69](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pose/_conjugated_groups.py#L69)

> Could anchor selection use the residue's declared polymer property instead of whether an up/down port survived patching? Two terminal patches remove both ports from a free lysine, but it remains an amino acid whose sidechain can carry a conjugate. A synthetic lysine–linker–lysine molecule currently forms no packing group even though both NZ attachments are present.

Reproduced and fixed using `properties.polymer.is_polymer`. The fully terminal crosslink now forms one three-member group, preserves geometry in all offered conformers, and packs with matching annealer/whole-pose energy on CPU/CUDA.

### 34. P1 — generic conjugated ligands have unsampled internal heavy torsions

[tmol/ligand/_preparation.py:1218](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_preparation.py#L1218), and [the chi-sample consumer](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_groups.py#L58)

> Could conjugated ligand preparation request heavy-chi samples? This call leaves `generate_heavy_chi_samples=False`, so internal torsion definitions alone never reach the group sampler. A movable propyl branch on a two-ended linker remains frozen even when its torsions fit the budget. Please distinguish holding a cyclic core rigid from dropping the sampling records for its movable branches.

The preparation call now enables existing non-ring heavy-chi generation for conjugated ligands. Constrained propylsuccinyl crosslinks retain two independently sampled pendant torsions (nine combinations plus current), while their cyclic core or external attachments stay fixed. These are generic grids, not independently fitted linkage distributions.

### 35. P1 — a spanning tree does not enforce cyclic or external constraints

[tmol/pose/_conjugated_groups.py:97](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pose/_conjugated_groups.py#L97), and [group chi selection](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_groups.py#L58)

> Can group discovery retain every internal bond and external attachment, and sampling check those constraints before turning an axis? A cycle-closing bond disappears from this tree, while a second polymer member can remain attached to an external backbone. Sampling a tree edge can then break a different bond. Independent axes should preserve rigid cycles and fixed boundaries; alternate cyclic conformations require correlated closure-aware sampling.

Implemented complete edge inventories and bridge/subtree checks, projection of constrained library axes, stable deduplication and a single input conformer when no axes remain. Synthetic free, cyclic and externally anchored crosslinks pass full geometry, target-angle and packing/energy checks on CPU/CUDA. Removing only these restrictions in a controlled ablation produces maximum bond errors of 13.82 Å (cycle) and 26.29 Å (external); constrained results stay below 0.000003 Å. This ablation uses current preparation and kernels, not a pristine upstream checkout. Task-imposed partial freezing is addressed in comment 37; alternate ring puckers remain open.

### 36. P1 — independent samplers remain enabled on secondary polymer members

[tmol/pack/rotamer/_conjugated_groups.py:177](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_groups.py#L177)

> Could this mask cover every member owned by the group sampler? A second lysine reached through a crosslink is a child here, but still has its own Dunbrack sampler. Disabling only the first anchor can mix independent rotamers into that member's correlated sequence. IncludeCurrent/Fallback should likewise remain independent only for a primary anchor that the group intentionally does not emit.

The helper now disables independent samplers across owned members, retaining the library-free primary-anchor exception. Full three-member crosslink construction and packing/energy tests pass on both devices. Subsequent disabling of individual group members is addressed in comment 37. Reenabling independent samplers after setup still needs explicit ownership checks.

### 37. P1 — group sampling ignores subsequent task masks

[tmol/pack/rotamer/_conjugated_chi_sampler.py:323](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/pack/rotamer/_conjugated_chi_sampler.py#L323)

> Can sampling consume both the group's sampler mask and each member's packing state before it enumerates or emits rows? Presence in `cons_bt_*` is not proof that this sampler is enabled. Disabling it for a complete synthetic crosslink still emits its 235 conformers plus fallback; disabling packing for just one member produces 235/235/236 counts. Fixed members must constrain the group's permitted motion, and a disabled group sampler must emit no rows.

Reproduced on follow-up commit `6b7071d54` and fixed: sampler/allowed-type masks are applied before enumeration; every atom of an inactive member constrains the permitted axes; only active members receive correlated rows. Fully disabled groups now produce 1/1/1 fallback counts, and the one-fixed-member diagnostic produces 10/10/1. Tests cover geometry, actual packing/energy, mixed-mask batches, reuse, budgets, rigid members and library-free anchors. See [the reproduction](reproduce_group_task_masks.py) and [FOLLOWUP.md](FOLLOWUP.md). Alternative chemical types and later conflicting samplers remain open.

### 38. P1 — conjugation patches omit connection bond and angle energies

[tmol/ligand/_preparation.py:797](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_preparation.py#L797), and [the unmatched-parameter path](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/score/cartbonded/potentials/cartbonded_pose_score.impl.hh#L523).

> Where are bond-length and bond-angle parameters generated for each new connection? This update adds patches and partial charges, but no connection-spanning cartbonded rows. The scorer silently skips paths without a matching row; generic torsion parameters do not replace stretching and bending potentials. Please test a rigid displacement across each attachment, then Cartesian minimization, in addition to packing geometry. Connection icoors describe construction geometry but do not themselves contribute to the score.

The follow-up branch `026475f0e` still reproduces this omission for all **14** attachment bonds in the biotin/N-glycan/O-glycan fixtures (1/7/6 bonds). Rigid-component stretching and perpendicular bending produce cartbonded stiffness below `2e-13` on CPU and CUDA; the same probe gives the expected **369.445 kcal/mol/Å²** for three ordinary peptide-bond controls. Full CPU Cartesian minimization of the biotin atoms, holding the protein fixed, moves the unperturbed amide link from **1.329 to 1.660 Å** while lowering the weighted score from **−84.55 to −93.54**. Starting 0.5 or 1 Å farther out yields final lengths of **2.150 or 2.556 Å**. These are 100-iteration minimization reproductions, not claims of convergence to a global minimum.

CUDA reproduces the full minimization failure: the same three starts finish at **1.658, 2.120 and 2.558 Å** (Slurm 244924). Unresolved. [diagnose_connection_stiffness.py](diagnose_connection_stiffness.py) records energies and analytic force projections, with a finite-difference stiffness from those forces. The required fix must give connection geometry explicit parameter ownership, preserve canonical/fragment parameters, avoid counting bonds twice, and cover score/gradient/packing/Cartesian-minimization paths. Repeated components with different partners cannot share parameters merely because their atom names match.

The follow-up now implements an explicit connection-record backend with validated topology/ownership, sparse pair lookup and one shared native evaluator. CPU/CUDA energy/gradient, packing, minimization and serialization tests pass with explicit synthetic records; canonical and existing fragment checks also pass. Ligand `.tmol` persistence is now covered by comment 41. A private generator now derives harmonic bond/angle records from complete capped conjugates, with independently checked MMFF units, mapped hydrogens, terminal variants and provenance. It covers all 15 source links (including the free disaccharide) and 61 adjacent angles in native CPU/CUDA checks. When explicitly installed for the diagnostic, all 14 anchored links have their expected stiffness and biotin minimization restores lengths to 1.360–1.379 Å from the same three starts. **Default integration and local chemistry corrections remain unresolved**, so this is not yet a fix for default preparation. See the connection sections of [FOLLOWUP.md](FOLLOWUP.md).

### 39. P2 — protonation cache grows with every pH and exposes mutable rules

[tmol/ligand/_dimorphite_dl.py:625](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_dimorphite_dl.py#L625)

> Can the cache retain the fixed compiled SMARTS rules instead of a fresh set of query molecules for every pH/precision tuple? `maxsize=None` retains every requested combination. It also returns the cached dictionaries and RDKit molecules directly, so modifying a public result changes subsequent calls. Keep compiled queries private and return owned nested state/query objects at public boundaries.

Fixed on the follow-up in both engines. The direct molecule API borrows private read-only queries; public rule results own copies. Six new tests per project cover mutation, 500 distinct pH requests and interleaved states; AtomWorks' complete affected suite has 40 passes. The 500-pH direct-API benchmark reduces tmol retained traced Python allocations from **16.10 MB to 0.065 MB**, with a **3.5%** warm latency increase (0.509→0.527 ms per molecule). AtomWorks avoids recompilation and improves **1.493→0.515 ms** (**2.90×**); its retained traced Python allocation increases from 34 to 44 KB for the bounded rule cache. These are rule-engine measurements, not end-to-end preparation or total native/process-memory claims. See [profile_dimorphite_cache.py](profile_dimorphite_cache.py).

### 40. P1 — injected cartbonded parameters keep the previous cache key

[tmol/database/__init__.py:210](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/database/__init__.py#L210)

> Can injecting cartbonded rows recompute the database hash? `attr.evolve()` replaces `residue_params` but retains `hash`; the scorer caches block and packed-block annotations under that hash. Two parameter databases used on the same pose therefore share whichever bonded parameters were annotated first. The public injection route should behave like `CartBondedDatabase.from_cartres_dict()` and leave the original database intact.

Reproduced with four CPU failures: changing alanine's CA–CB equilibrium length and stiffness through `inject_residue_params()` produces zero score difference on the same pose instead of the independent harmonic prediction. The failure occurs in either annotation order, in whole-pose and weighted block-pair scoring. Fixed by rebuilding the content hash. The four CPU regressions pass; Slurm **245573 has 14 passes** across CPU/CUDA, including score/gradient regressions, independent connection-improper checks and the existing manual-parameter replacement test. The test does not rely only on unequal hash strings.

### 41. P1 — conjugate params export omits the canonical partner

[tmol/ligand/_preparation.py:1297](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_preparation.py#L1297), with the loader at [tmol/ligand/_params_file.py:202](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_params_file.py#L202).

> Can the exported bundle include the canonical attachment partner's patches and charge changes? `_inject_canonical_conjugations()` adds them only to the database, while this call writes only the ligand preparations. The loader also attaches patches only to base residues defined in the file, discarding a patch for an existing ASN/LYS/THR partner. Please test preparation → export → fresh-database reload for the actual conjugate fixtures, including atom identity, connections and score/gradient parity.

Reproduced as three CPU failures at `e2107de71`: biotin, N-glycan and O-glycan exports lose patched residue types. Fixed on the follow-up by keeping shared partner metadata in the preparation bundle, applying patches to existing partners during injection, and loading additions even if the ligand base type was registered earlier. Explicit connection records and provenance now roundtrip through `.tmol`. Bond ordering and numeric internal coordinates are preserved instead of being reformatted lossily. Version 2 exports prevent older readers from silently dropping the new metadata; the new reader accepts version 1 files. CPU/CUDA residue, charge, topology, score and gradient roundtrips pass. This does not generate the default missing connection parameters in comment 38.

### 42. P2 — every exported atom record leaves a class in a global registry

[tmol/ligand/_params_io.py:367](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_params_io.py#L367)

> Can `_FlowDict` be defined and registered once at module scope? Defining it inside `_flow_atom()` creates a distinct class for every atom, charge and parameter row. `_CompactDumper.yaml_representers` retains each class, so discarding the output records does not release them. One reusable marker class also avoids the extra dictionary copy.

Fixed and measured with the same prepared records in both writers. Twenty exports retain **10.19/53.51/51.85 MB** of traced Python allocation for biotin/O-glycan/N-glycan before, versus **1.1 KB** each after garbage collection following the fix. Registry growth is **4,220/22,360/22,780** classes before and zero after. Seven alternating-order warm timing pairs give **1.06–1.08×** median export speedups. This is YAML export, not end-to-end preparation or total-process memory. See [profile_params_writer.py](profile_params_writer.py) and [results/params-writer-profile.json](results/params-writer-profile.json).

### 43. P1 — acylation uses only the hydrogen count from the derived chemistry

[tmol/ligand/_preparation.py:869](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_preparation.py#L869), with patch typing in [tmol/ligand/_conjugation_patches.py](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_conjugation_patches.py).

> How should the derived conjugated chemistry update local parameters and torsion ownership? This helper computes the site type and charges but retains only `n_hydrogens`. In the biotin fixture, the helper identifies an amide `Nad`, while the patch installs `Nbb` and cartbonded still fetches the unmodified LYS record. The surviving CE–NZ–HZ1 angle therefore retains its 109.5° amine target, as do the three old lysine proper torsion rows involving HZ1. Please cover the complete local chemistry change, not only the number of hydrogens or the new cross-bond spring.

Reproduced on CPU and CUDA by [diagnose_acylated_lysine.py](diagnose_acylated_lysine.py). The retained angle has `x0=1.91114 rad, K=51.348`; moving the hydrogen from 109.5° to 120° costs **0.862196 kcal/mol** in cartbonded angles. Rotating it by 180° about CE–NZ changes the retained lysine proper energy by **14.758556 kcal/mol**. This latter scan also changes the missing connection angles; it is not an equal-energy symmetry test. There is no NZ-rooted cart improper and no matching generic improper for the installed `Nbb` site, but proper torsions do respond: this is **not** a claim that all planarity energy is absent. The topology-derived MMFF diagnostic has local angle targets around 120° and reports `Nad`; its uncapped pair's total charge includes artificial backbone termini and is not a whole-conjugate charge reference.

The follow-up adds isolated, exact-variant `CartRes` replacements and their `.tmol` persistence as the mechanism for local corrections. Eight before-fix CPU checks show that exact variant rows were ignored; independent whole-pose, weighted block-pair, rotamer and biotin bundle score/gradient checks now pass on CPU/CUDA. **Automatic chemistry-derived local parameters, typing/charge updates and ownership remain unresolved.** Synthetic replacement tests establish the mechanism, not a fitted amide model.

### 44. P2 — polymer capping discards source atom annotations

[tmol/ligand/_polymer_profile.py:1201](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_polymer_profile.py#L1201)

> Can retained atoms keep their input annotations when caps are added? Rebuilding an AtomArray here copies only coordinates, names, elements and four residue fields. Formal charges, source chemistry tags and insertion codes disappear before the molecule converter sees them. Synthetic caps can have empty annotations without discarding the metadata on real atoms. Please also preserve long cap names rather than truncating them to the default atom-name column width.

Fixed on the follow-up. The baseline comparison confirms that `charge` and a source chemistry annotation disappear. Retained annotations now survive, cap annotations start empty/zero except residue identity, and long names remain intact. Coordinates and bonds match the baseline on the shared fields in four backbone examples. A topology-only option avoids geometric frame construction and produces all-NaN coordinates; it is used by the new private capped-conjugate model builder. Seven alternating-order sets of 100 calls measure **2.46–2.64×** faster topology-only capping, while coordinate-producing calls are **7–13% slower** because they now retain the annotations (about 20–33 µs per call in this run). These are cap-construction timings, not full preparation speedups. See [profile_capping.py](profile_capping.py) and [results/capping-profile.json](results/capping-profile.json).

### 45. P1 — electrostatic annotations retain the first database's parameters

[tmol/score/elec/_elec_energy_term.py:26](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/score/elec/_elec_energy_term.py#L26), with the [existing annotation guard](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/score/elec/_elec_energy_term.py#L43).

> Can electrostatic annotations be scoped to the charge/count-pair database and this Rosetta/generic typing configuration? The block and packed-block setup guards check only whether an attribute exists. Reusing a pose with injected charges or changed typing therefore retains the first setup's tables. Each rendered module also needs to retain its own tensors, so refreshing another term cannot change an existing module's score. Please exercise both setup orders, block-pair weighting and rotamer energies/gradients.

The stale charge guard predates this PR; new parameter injection and generic typing expose it to the expanded chemistry workflow. Four focused CPU score checks fail against the pre-fix implementation. The follow-up keeps only the latest annotation on each owner, weakly references its immutable database, and captures owned parameter tensors at render time. Unchanged representative-distance tables are reused across charge changes. Exact complete variant charge/count-pair rows now take precedence over individual patches; previously the resolver rejected multiple patch suffixes. That resolver restriction also predates the PR. Missing applicable charges raise instead of silently becoming NaN. Independent CPU/CUDA energy/gradient checks, a terminal-conjugate bundle and database-lifetime tests are recorded in the electrostatic section of [FOLLOWUP.md](FOLLOWUP.md).

### 46. P2 — unresolved attachment coordinates become NaN construction parameters

[tmol/ligand/_preparation.py:768](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_preparation.py#L768), with [residue identity at line 766](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_preparation.py#L766).

> Can this helper accept only finite, positive measurements and use complete residue-instance boundaries? The reader deliberately preserves unresolved coordinates as NaN. If a bonded endpoint is unresolved, its NaN distance becomes the connection icoor in every patched type and the exported bundle. A later unresolved repeat also overwrites an earlier valid measurement. Comparing only chain and residue number misses a cross-residue bond between insertion-coded residues.

Seven focused CPU checks fail before the fix. The follow-up filters unresolved/coincident endpoints, keeps a prior valid observation, and uses Biotite's full contiguous-residue boundaries. Export/reload retains finite geometry without filling the input's missing coordinates. A missing polymer NZ can still be rebuilt; a missing ligand C11 retains the existing explicit construction error. The same three missing-endpoint cases pass through AtomWorks input. Bond/coordinate temporary arrays are processed in chunks. This preserves the existing no-measurement fallback, which inherits the departing hydrogen's construction frame; it does not provide the chemistry-derived heavy-atom equilibrium geometry or attachment energy model discussed in comment 38. Distinct finite observations still use the last instance, so general context-specific construction remains open. See the attachment-measurement section of [FOLLOWUP.md](FOLLOWUP.md).

### 47. P1 — equal component names do not identify a fragment cut bond

[tmol/io/details/_select_from_canonical.py:1239](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/io/details/_select_from_canonical.py#L1239)

> Can exact fragment cut bonds be removed by the expansion/mapping layer instead of discarding every bond whose endpoints share a base name? This function also handles callers that supply explicit canonical bonds and have no later fragment-restoration mapping. Repeated copies share base names, so the predicate cannot establish which original component instance owns a cut. Declared connections between already prepared fragment blocks must remain in the returned connection list.

Reproduced against the updated selection module: three declared links over a jagged pair of poses, including crossed instances of the same two fragment types, return an empty list. The reconciled branch retains all three links and verifies the exact bidirectional connection tensors through public canonical pose construction. Normal AtomArray fragment expansion still removes only its own cut bonds before this function and restores them through the explicit mapping. The isolated module comparison uses the branch's prepared fixture types; it is not a claim that the otherwise incomplete upstream tree imports cleanly.

### 48. P2 — the conjugated-only fallback retains just the first candidate

[tmol/io/details/_select_from_canonical.py:1132](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/io/details/_select_from_canonical.py#L1132)

> Could this decision use the class's unconjugated candidate inventory before the fallback loop, or the existing explicit fragment flag? After the first conjugated candidate is added, `any(lists[t][s])` becomes true and every later candidate in the same class is skipped. Alternative states or terminal forms therefore disappear according to input type order.

Two CPU reproductions reverse the order of two fragment states in one equivalence class. Both retain only index 0 upstream. The branch's existing explicit fragment predicate retains both states in either order and avoids the extra fallback loop. Ordinary unconnected canonical residues remain protected from accidental selection of conjugated variants.

### 49. P1 — connection impropers are limited to fragments of one source ligand

[tmol/score/genbonded/potentials/genbonded_pose_score.impl.hh:666](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/score/genbonded/potentials/genbonded_pose_score.impl.hh#L666)

> Could this gate follow the center's scoring ownership instead of fragment origin? An amide formed between a canonical residue and a ligand is also a three-coordinate center spanning a connection. Even after assigning a compatible Nad/CDp/CS2/HN neighborhood, this predicate drops its improper in all four scoring paths. The central atom's physical type should determine ownership; generic lookup references on its neighbors should not transfer their canonical sidechain torsions to the generic term.

The isolated lysine–biotin reproduction retains just the existing Nad/CDp/CS2/HN table row and explicitly supplies the corrected site types; it is not a claim that default preparation already installs those types. Moving the retained hydrogen out of plane gives exactly zero attachment energy in both whole-pose and block-pair scoring with the fragment gate. This controlled comparison retains the new Python lookup-reference support and isolates the native predicate; it is not an unmodified upstream checkout. The correction checks center ownership in the shared helper and accepts other chemical connections. An optional per-atom `genbonded_type` provides lookup without changing physical atom typing; a LYS chi4 regression guards against transferring its axis to generic scoring. Full automatic local chemistry, charge and hydrogen reconstruction remain separate completion requirements.

### 50. P2 — missing-leaf construction recognizes only polymer connection names

[tmol/io/details/_build_missing_leaf_atoms.py:367](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/io/details/_build_missing_leaf_atoms.py#L367)

> Could this resolve any connection declared by the block type? A retained amide hydrogen needs the bonded partner as its plane reference. An icoor using `conj_NZ` or `conj_ND2` currently takes the atom-name branch and raises `KeyError`, although these are valid connections. The same limitation affects custom connection names outside the conjugation pipeline.

The lookup predates this PR; general chemistry and connection-aware hydrogen construction expose its restriction. A controlled replay of the pre-fix function fails for `conj_NZ`. The branch uses the declared connection index and retains the absent `up`/`down` fallback. The coupled biotin and N-glycan tests exercise actual hydrogen construction using the remote partner, rather than only testing the name lookup.

### 51. P1 — chemical connectivity does not establish conformer correspondence

[tmol/pose/_conjugated_groups.py:143](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/pose/_conjugated_groups.py#L143)

> Could the sampler declare which outputs form a joint conformer, and could merging validate that declaration? OptH independently samples protons on connected residues. Equal counts such as 3/3 do not mean rotamer 0 on one block corresponds to rotamer 0 on the other; these blocks need all nine pairings. Unequal counts must also remain valid for independent sampling. Conversely, a later independent sampler must not append extra states to one member of a declared joint group.

Controlled replay of the pre-fix branch marks independent 3/3 samples as correlated and rejects 3/2 samples. Upstream lacks even that count check. The branch now carries producer-declared considered-block groups through merging, rejects overlapping/additional ownership before coordinate allocation, and uses the validated groups for both pair enumeration and energy collapse. It stores one mask per rotamer set, shared by score terms. Actual joint-sampler masks match the previous implementation on all three fixtures; the lookup profiler reports only this setup stage, not an end-to-end scoring speedup.

### 52. P1 — OptH selects an attachment torsion as an amide flip

[tmol/pack/rotamer/_opth_sampler.py:314](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/pack/rotamer/_opth_sampler.py#L314)

> Can NHQ flips use their actual amide/ring axis and check which atoms it moves? Conjugation appends chi torsions, so the lexicographically last chi can be a cross-block linkage rather than ASN chi2. A terminal amide flip also cannot independently move an atom bonded to a glycan. Eligibility, sidechain roots and sample counts need the same annotation so disabling a flip allows the fallback sampler to supply the current conformation.

The attached ASN fixture fails the controlled pre-fix annotation check. The branch preserves the canonical ASN/GLN/HIS axes and disables a flip when its downstream atoms include a declared connection. During development, inconsistent eligibility after disabling the flip suppressed fallback and exposed a native bounds exit in backbone scoring; that intermediate failure is retained in the validation log and is not counted as a passing test. Eligibility and sidechain roots now use the same annotation.

## Validation record

See [VALIDATION.md](VALIDATION.md) for the initial review commands, counts, environment, examples, and remaining failures; [FOLLOWUP.md](FOLLOWUP.md) records subsequent fixes and validation. All raw run logs were retained separately under `/mnt/home/kdidi/tmol-pr503-results`. No upstream golden score files were regenerated to make tests pass.
