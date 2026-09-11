# Review of tmol PR #503

Reviewed PR: https://github.com/uw-ipd/tmol/pull/503  
Author branch: `dimaio/noncanonicals_through_ligand_pipeline`  
Pinned head: `c03c1e745f3bc655948ea12dac44d6c74620358f`  
Diff merge base: `08d82941b6b5bfcd405303f8730b36b54dcfd28a`  
Improvement branch: `review/pr503-chemistry-efficiency`  
Review date: 2026-09-11

**Recommendation: request changes.** The chemistry expansion is substantial, but the exact submitted tree cannot import its I/O package. After supplying the missing module, targeted checks expose batch identity errors, lost/misassigned group torsions, residue identity errors, and a mixed D/L disulfide score that depends on residue ordering. The passing chemistry fixtures do not cover these cases.

This document and the comments below are drafts for the user; no review or comments were posted to Frank's PR. Links in the comments point to the pinned upstream commit, so their line numbers remain valid after changes to this branch.

## What the PR implements

- **Input and chemical identity:** a CIF reader adds unresolved atoms at NaN; preparation classifies polymer residues from declared entities, connections, and chemistry. Noncanonical backbones are capped for parameter generation, then reconstructed as polymer types with generated terminal patches. Conjugation patches represent attached glycans and ligands as connected blocks.
- **D amino acids:** generated chemical/charge/cartbonded/reference records, mirrored Dunbrack libraries, mirrored backbone grids, and D-aware disulfide parameters. Optional symmetric glycine tables make full mirror-image comparisons possible.
- **Scoring:** hybrid generic/Rosetta typing partitions torsion ownership; cartbonded enumerates connection-spanning impropers instead of treating them as bonded paths. Nucleic-acid references can be borrowed from similar bases.
- **Packing and kinematics:** chemical fold-tree edges, more general single-residue trees, chemistry-derived sampling references, group conformer enumeration and energy collapse onto one representative.

The core design is useful: explicit reference fields are more extensible than residue-name inference; preserving unresolved atoms prevents accidental truncation of chemical identities; and a covalent group needs correlated conformers. The main weaknesses are identity/scope assumptions that hold for one fixture but fail across poses, repeated residues, or reused samplers, and incomplete enforcement of the advertised sampling budget.

The complete 213-file upstream inventory is in [upstream-files.tsv](upstream-files.tsv). Static review concentrated on the new preparation, CIF completion, database mirroring/caching, group-packing, fold-forest, and scoring changes. Generated databases and fixture coordinates were assessed through their generators, schema, provenance notes, and executable checks; this is not an independent refit or scientific validation of those parameters.

## Branch improvements

1. Supply the absent `_cyclic_search.py` using the documented API: preserve explicit closures, infer first/last connections within each contiguous polymer run, exclude missing atoms/nonpolymers/padding/single residues, and keep all operations on the tensor device. **This is a new implementation inferred from the committed callers and documentation, not Frank's missing original file.** It is independently tested and must be reconciled if he supplies that file.
2. Replace whole-structure masks with residue boundaries, including insertion codes; select representative residues lazily; prevent conjugation preparation from merging equally numbered residues across chains.
3. Cache CCD templates once per component **within each completion call**, concatenate arrays once, and construct/remap bond tables in batches. No global mutable CCD cache is introduced.
4. Identify hydrogen atoms by element when checking whether a template covers the observed heavy atoms; names such as `1H` no longer block completion.
5. Retain original chi indices through budgeting, separating the anchor library's multiplicity from the child's chi numbering. Update the conformer count by division while freezing chi instead of recomputing the product each iteration.
6. Key anchor library entries by `(pose, block)` and scope group kinforest caches to their `PackedBlockTypes`, so equal numeric type indices in another database cannot reuse a different chemistry's tree.
7. Reuse invariant node/scan/generation tensors across group conformers. This removes repeated CPU-to-device allocations; the much larger per-conformer kinematics loop remains a future batching opportunity.
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

## Suggested inline comments

Each item gives an upstream location, suggested comment, and what this branch does about it. “Reproduced” means exercised against PR code with only the missing import dependency supplied, not merely inferred from source.

### 1. P0 — missing module prevents test collection

[tmol/io/details/__init__.py:10](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/io/details/__init__.py#L10)

> Could you commit `_cyclic_search.py`? This import points to a file absent from the PR tree. Both CPU and H200 pytest runs fail while loading conftest with `ModuleNotFoundError`, before collecting any tests. A clean-checkout import/test smoke check would catch this.

Reproduced on the exact head. Branch supplies a separately tested implementation; original baseline failure is retained.

### 2. P1 — mixed D/L disulfide energy depends on residue ordering

[tmol/score/disulfide/potentials/potentials.hh:122](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/score/disulfide/potentials/potentials.hh#L122) and the derivative path at line 312.

> Could the S–S potential use an explicitly defined pair-chirality model? It currently reads only `params1.dss_*`. Reversing the same mixed D/L pair changes the first parameter row but not the physical structure. In a two-cysteine reproduction using 6DMZ coordinates, reordering blocks changes the disulfide-only score from −0.0337973 to 0.5349466. Please add permutation and mixed-chirality gradient tests; whole-L versus whole-D mirror tests do not detect this.

Reproduced. [reproduce_disulfide_order.py](reproduce_disulfide_order.py) included. The geometry is deliberately held fixed while only block order changes; it is an invariance test, not a claim that the relabeled pair is a relaxed D/L structure. Scientific fix left open.

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

Invariant allocations hoisted in this branch. No claim of measured end-to-end group-packing speedup.

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

Reproduced on CPU and H200; baseline also fails, including with cyclic inference disabled. Left open: the cap representation and intrinsic-terminus classification need to agree. The example runner retains both failures.

### 23. P1 — reference score assertions do not reproduce

[tmol/tests/score/test_noncanonical_scoring.py:78](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/score/test_noncanonical_scoring.py#L78).

> Could you document the environment and preparation state behind these reference scores and make them reproducible before relaxing tolerances? All four classes currently fail on both CPU and CUDA in the H200 environment. For the beta-peptide fixture, `fa_ljrep` is approximately 121.26 versus the committed 662.72, which is far beyond a precision discrepancy. Please distinguish changes in generated coordinates/parameters from changes in the scoring kernels.

Recorded without updating goldens. Environment and baseline/candidate comparison are in the validation record; numerical discrepancies alone do not identify which implementation or reference is scientifically correct.

### 24. P2 — the D repacking test checks labels, not geometry

[tmol/tests/score/test_mirror_image_scoring.py:120](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/score/test_mirror_image_scoring.py#L120).

> Could `_chirality()` also check the signed tetrahedral volume around stereocentres using the output coordinates? Reading `properties.polymer.sidechain_chirality` verifies that the type label stays D, but a D-labeled block built with inverted geometry would still pass. The current test also would not establish that the mirrored library was actually used.

Coverage suggestion; no unvalidated stereochemistry convention added to the branch.

### 25. P2 — fold-tree tests contradict branch-point splitting

[tmol/tests/kinematics/test_fold_forest.py:182](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/kinematics/test_fold_forest.py#L182), also line 234.

> Could the expected polymer edges be split at the conjugation parent? The builder now emits `0→2` and `2→3` so the chemical edge leaving block 2 has a parent edge ending there. This matches the new validator's documented invariant, but the assertion still expects `0→3`. The mid-chain conjugation test has the same stale expectation.

Both failures reproduced on the baseline. Updated these two expectations; all seven executable fold-forest tests then pass on CPU, with the CUDA-only smoke case skipped locally.

### 26. P1 — conjugation selection excludes every ligand fragment

[tmol/io/details/_select_from_canonical.py:1110](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/io/details/_select_from_canonical.py#L1110), and the intermediate array at [tmol/ligand/_fragmentation.py:742](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_fragmentation.py#L742).

> Could explicitly prepared ligand fragments be exempt from this candidate filter, and their cut bonds remain owned by the fragment mapping? Every fragment has a non-polymer connection, so this filter leaves no candidate for its residue class. Merely allowing the candidate then routes its original cut bonds through conjugation-patch selection, although `apply_fragment_connections()` separately restores those bonds. The baseline fails 35 fragmentation tests before their restoration/scoring assertions can execute.

Reproduced. The branch permits fragment candidates and removes only bonds crossing newly split pieces of the same original residue from the intermediate bond table. The prepared fragment mapping remains authoritative for cut-bond restoration. All 35 previously failing fragment cases pass in the final 133-case fragment/conjugation run (129 passed, four existing skips).

### 27. P2 — HYP rotamer count disagrees with the stated sampling model

[tmol/tests/pack/rotamer/test_noncanonical_rotamers.py:266](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/tests/pack/rotamer/test_noncanonical_rotamers.py#L266).

> Could this distinguish borrowed-library cardinality from the additional chi sampling/expansion multiplier? HYP currently produces 18 rotamers on both CPU and CUDA, versus the asserted six. The comment describes two library rotamers times three hydroxyl samples. Please pin whether expanded samples are intended here, then test the library and extra-chi counts separately so a factor-of-three change has a clear diagnosis.

Reproduced on the baseline and initial candidate. Follow-up checks independently identify two unique library states and nine hydroxyl angles (three means with ±20° expansions) on CPU/CUDA. The expected count is now 18 with that separate check. Explicit task overrides and actual group-library bounds now have CPU/CUDA checks. Aggregate/default budget policy remains open.

### 28. P1 — terminal proton sampling rotates an entire generated nucleotide

[tmol/ligand/_polymer_builder.py:893](https://github.com/uw-ipd/tmol/blob/c03c1e745f3bc655948ea12dac44d6c74620358f/tmol/ligand/_polymer_builder.py#L893).

> Could the nucleotide jump root use the sugar side of the glycosidic torsion, as canonical nucleotides do, instead of the second mainchain atom? For generated 5CM that atom is O5′, which carries a proton chi at a free 5′ terminus. OptH writes that chi into the jump degree of freedom and moves heavy atoms by up to 2.44 Å. Please check heavy-coordinate preservation for every offered proton rotamer, not only finite scores or the selected residue label.

Reproduced in the follow-up branch before the root correction. After correction, the maximum heavy displacement in that diagnostic is 0.0000049 Å. Five modified DNA/RNA fixtures preserve heavy atoms in all offered proton rotamers and actual packing on CPU/CUDA (job 237980). Old score references may encode the damaged geometry and must be reconciled independently.

## Validation record

See [VALIDATION.md](VALIDATION.md) for the initial review commands, counts, environment, examples, and remaining failures; [FOLLOWUP.md](FOLLOWUP.md) records subsequent fixes and validation. All raw run logs were retained separately under `/mnt/home/kdidi/tmol-pr503-results`. No upstream golden score files were regenerated to make tests pass.
