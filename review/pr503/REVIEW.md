# Review of tmol PR #503

The shared protonation/dependency audit adds findings 101–108 in
[PROTONATION_AUDIT.md](PROTONATION_AUDIT.md), including six direct comparisons
against the existing native histidine fixtures. The complex-input audit adds findings 92–100 in
[CORPUS_FINDINGS.md](CORPUS_FINDINGS.md). The proposed shared file/tensor input
contract and PR #380 review are in [INPUT_CONTRACT.md](INPUT_CONTRACT.md).

Reviewed PR: https://github.com/uw-ipd/tmol/pull/503  
Author branch: `dimaio/noncanonicals_through_ligand_pipeline`  
Initial pinned head: `c03c1e745f3bc655948ea12dac44d6c74620358f`\
Previous updated head: `0f4c3bc426bca78e8681f0b730fa23c3e26ef261`\
Latest reviewed head: `0593a93b07d80b0302383163d2d98c78e315ab98`\
Diff merge base: `08d82941b6b5bfcd405303f8730b36b54dcfd28a`  
Improvement branch: `review/pr503-chemistry-efficiency`  
Review date: 2026-09-12

Both subsequent six-file updates have been reviewed separately. Comments 1–46
retain their original `c03c1e745` anchors; comments 47–56 address `0f4c3bc42`,
and comments 57–91 address `0593a93b0`. The
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

The initial 213-file inventory is in [upstream-files.tsv](upstream-files.tsv); the intermediate 214-file inventory is in [upstream-files-0f4c3bc42.tsv](upstream-files-0f4c3bc42.tsv), and the latest 215-file inventory is in [upstream-files-0593a93b0.tsv](upstream-files-0593a93b0.tsv). Static review concentrated on the new preparation, CIF completion, database mirroring/caching, group-packing, fold-forest, and scoring changes. Generated databases and fixture coordinates were assessed through their generators, schema, provenance notes, and executable checks; this is not an independent refit or scientific validation of those parameters.

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
4. **Budget semantics:** Is `set_chi_sample_budget()` meant to bound library rotamers, sampled heavy chi, proton combinations, group conformers, total block rotamers, or pairwise memory? What happens when the anchor library alone exceeds the limit, when all child chi freeze, or when required proton samples exceed it? Does a frozen chi retain its input angle or use ideal residue geometry, and is that policy consistent for upstream and downstream axes?
5. **Group constraints:** What is the contract for a partially disabled group, multiple polymer anchors joined by a conjugate, free oligosaccharides, and groups with cycles? Should unsupported topologies fail clearly or remain frozen? Which bond closes a non-tree cycle during sampling/minimization?
6. **Chemical authority:** How should custom residue names with declared bonds/stereochemistry but missing coordinates work under `use_ccd=False`? Currently coordinate completion can still require a CCD component. What explicit input describes ambiguous polymer ends instead of selecting the conventional backbone heuristically?
7. **Sampling versus scoring:** The graph matcher borrows sidechain references for sampling, while scoring ownership is independently inferred. Which tests guarantee that changing a sampling reference cannot silently change the scored potential or leave a rotatable bond unconstrained? How were graph-match thresholds and unknown-base averages chosen?
8. **Lifecycle and reproducibility:** Are samplers intended to be reusable across databases/tasks? Are generated chemistry and cache entries bounded in a long-running service? Can users persist the seed, generated parameters, chosen references, warnings, and exact input authority in a build report?
9. **API and migration:** The PR body says `process_ligands=True`, but the implementation uses `prepare_ligands=True`. Is removing `sample_proton_chi` intentional? Can public docs show the supported CIF, AtomArray, SMILES/mol2, cyclic, and group-packing entry points with executable examples?
10. **Scale and release:** What canonical-protein load/score/pack baseline is acceptable after eagerly adding mirrored tables and invoking group discovery in scoring setup? Can the PR be split along its existing commit sequence into import/preparation, score corrections, and group packing?
11. **Shared AtomWorks chemistry:** Can AtomWorks own CIF completion and chemical annotations, with tmol consuming a completed AtomArray? Before replacing the default reader, which author/label identifiers, insertion codes, alternate locations, unresolved residues and CCD authority must be preserved? Can the copied Dimorphite and pre-protonation rules use one versioned shared API, while tmol keeps parameter generation and numerical scoring?

12. **Shared rule provenance:** Tmol adds an enamine SMARTS rule with pKa 1 ± 1 that AtomWorks does not contain. What evidence supports its scope and values, and should it be a shared default or an explicit preparation profile? Direct replacement currently changes charge states for an enamine and a vinylogous amide at pH 2 and 7.4. Which complete rule inventory and model version should exported parameters record?

13. **Attachment charge policy:** Should local charge changes preserve the curated residue baseline and add a connected-versus-disconnected MMFF correction, or replace the whole capped group's charges? The former preserves remote backbone parameters but is a model choice requiring validation. Complete capped biotin changes formal charge by −1, with heavy-atom-plus-hydrogen deltas on both LYS and BTN, including LYS CE. Applying only a hydrogen-count patch cannot represent this. Which atom-type, torsion-ownership and proton-construction changes must accompany the charge model?

14. **Junction parameter scope:** Can renamed junctions carry explicit chemical roles, including the partner frame, rather than borrowing rows through atom names alone? How should both ends being renamed, carbonyl oxygen versus hydroxyl, N-substitution, retained hydrogens, and non-peptide polymers be covered? The new `0593a93b0` helper maps nucleotide phosphate to peptide nitrogen; single-frame substitution also leaves the remote atom names canonical. A connection-specific parameter contract would make the supported scope explicit.

15. **Mirror sampling contract:** Should paired L/D structures receive one-to-one reflected conformer sets, including terminal defaults, probability truncation and extra-chi sampling? Which equivalent-atom permutations are allowed, and what grid-boundary convention makes the probability-ordering tables agree with their reflected counterparts? Matching scores and preserving D chirality do not establish this.

16. **Achiral glycine model:** Is `with_symmetric_gly()` intended to symmetrize the complete model or only select backbone tables? The two alpha-hydrogen ideal lengths and bonded targets differ, which breaks exact reflection after rebuilding equivalent hydrogens. Also, a reflection-invariant omega potential can retain backbone dependence; what supports replacing it with a uniformly trans table? The follow-up preserves the default parameter files and makes the hydrogen averaging opt-in.

17. **Private library grids:** Are custom backbone origins, anisotropic spacing and smaller periodic tables supported? Can that contract be checked through both scoring and sampling, including serialized reflected libraries and mixtures of differently sized tables? These cases expose inherited layout assumptions that default 36×36 grids conceal.

## Suggested inline comments

Each item gives an upstream location, suggested comment, and what this branch does about it. For the initial review, “reproduced” means exercised against PR code with only the missing import dependency supplied. Later follow-up comments identify their tested branch/stage in the linked source manifests; they are not claims that an untouched upstream checkout ran successfully.

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

Reproduced and addressed: explicit limits survive task conversion; private NA/OptH sampler views avoid mutating reusable caller objects; actual group-library cardinality and native Dunbrack count checks enforce bounds before final sample allocation. CPU/CUDA tests cover setting limits before/after adding samplers and repeated reuse. The shared merge path also checks the combined count across all samplers and allowed types at each physical residue, including current/fallback rows. This closes per-residue aggregate enforcement for explicit limits; default policy, adaptive source-library workspace and whole-task/pair-memory limits remain open. See [BUDGETS.md](BUDGETS.md).

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

The later work in comment 35 preserves existing cyclic/external constraints by restricting independent sampling axes. Alternative ring conformations and a general closure sampler remain outside the implemented contract; no unvalidated closure algorithm was added.

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

Initially a static lifecycle concern; subsequently fixed with the shared bounded weak-identity cache in comment 29. The sampler group-tree cache is separately scoped to its packed-type owner.

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

Latest reconciliation at `a237928b0`: the later upstream YAML lowers beta-peptide LJ repulsion to approximately 122; the original quoted 663 discrepancy is historical. The current standalone CPU run still fails DNA and beta-peptide references; the container fails all four classes on CPU/CUDA. Frozen-input replay passes all 192 term comparisons on each backend, with exact atom identities/types and coordinates. CPU has 190 exact terms and a maximum difference of 3.81e-6; CUDA's maximum is 0.00263548. This separates the numerical replay from fresh preparation and the DNA geometry correction; it does not establish the missing historical parameter provenance or justify a blind YAML refresh. The executable verifier rejects missing cases/terms, duplicate cases, changed coordinates/types, changed scores and corrupted reference inputs. See [results/noncanonical-reference-status.json](results/noncanonical-reference-status.json).

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

The follow-up adds isolated, exact-variant `CartRes` replacements and their `.tmol` persistence as the mechanism for local corrections. Eight before-fix CPU checks show that exact variant rows were ignored; independent whole-pose, weighted block-pair, rotamer and biotin bundle score/gradient checks now pass on CPU/CUDA. The later private coupled generator supplies connected-versus-disconnected corrections for local types, charges, bonded terms, hydrogen construction and torsion ownership, with explicit provisional MMFF provenance. Guarded `.tmol` bundles now deliver those corrections through the ordinary loader (comment 86). **Default model selection, three-block angles and broader chemical-context coverage remain unresolved.** These checks do not establish a scientifically fitted amide model.

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

### 53. P1 — generic bonded annotations retain another database's parameters

[tmol/score/genbonded/_genbonded_energy_term.py:307](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/score/genbonded/_genbonded_energy_term.py#L307)

> Could both block and packed-block annotations be scoped to the generic database and chemical element mapping? Reusing the same pose with another database currently skips setup because the attributes already exist. Changed Fourier coefficients, improper strengths, type hierarchies or Rosetta ownership can therefore retain the first term's data. Rendering an older term again after another setup also needs to recover its own parameters, while existing modules keep their original tensors.

Four CPU reproductions reuse a jagged KK/KKK pose with synthetic generic ownership and two parameter databases whose strengths differ by exactly two. Both setup orders fail in whole-pose and block-pair scoring before the fix. This is a controlled scaling test of cache identity, not a proposed physical model. The follow-up keeps one latest annotation per owner, weakly identifies its database, and captures the returned tensors when rendering. Rotamer energies/weighted gradients, ownership changes, invalid element mappings and database collection are covered. The database-wide inter-block tables are now shared through a bounded weak cache instead of rebuilt for each packed set. Slurm 249000 passes 119 CPU/CUDA cases with four fixture-specific skips; exact-tensor setup profiles and limits are recorded in [FOLLOWUP.md](FOLLOWUP.md).

### 54. P1 — Cartbonded ownership outlives the configuration that defined it

[tmol/score/cartbonded/_cartbonded_energy_term.py:235](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/score/cartbonded/_cartbonded_energy_term.py#L235)

> Could the Rosetta ownership mask be part of the term's parameter snapshot? This `hasattr` guard ignores changes to `genbonded.rosetta_typed`, and the Cartbonded content hash does not contain that setting. Reusing a pose with another ownership configuration therefore scores connection impropers using whichever configuration ran first. Existing rendered modules must also retain their own mask after a new term is set up.

Both setup orders fail in a perturbed AA connection reproduction with unchanged Cartbonded parameters and different ownership sets. The follow-up records the ownership setting/mask in the packed annotation, refreshes it on configuration changes, and captures it during rendering. Identical ownership masks are shared across parameter fits. The independent peptide/proline improper reference and the new reuse tests pass on CPU/CUDA.

### 55. P2 — bonded annotation dictionaries retain every fitted parameter set

[tmol/score/cartbonded/_cartbonded_energy_term.py:370](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/score/cartbonded/_cartbonded_energy_term.py#L370), also [block annotations at line 228](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/score/cartbonded/_cartbonded_energy_term.py#L228).

> Can these caches have an explicit lifetime bound? Each changed bonded database adds another set of subgraphs, parameter dictionaries and packed tensors to every reused owner. Retaining a pose or its block types therefore retains all prior fits even when their databases are no longer needed. Evicting cached annotations must leave existing modules valid and allow an older term to rebuild its own snapshot.

The branch now retains the two most recently used parameter configurations per block and packed set. Six fitted databases previously retained six entries; the cache now remains at two. On 230 default block types, cache-reachable packed tensor storage falls from 9,379,104 to 3,170,528 bytes, and block NumPy arrays from 5,974,848 to 1,991,616 bytes on both CPU/CUDA. This excludes Python dictionaries, rendered-module snapshots and process/allocator memory. Reusing an evicted configuration rebuilds it; this is a memory/performance tradeoff, not a faster fit claim. Existing and newly rendered modules are checked after repeated evictions.

### 56. P2 — common atom setup synchronizes once per atom

[tmol/score/_atom_type_dependent_term.py:154](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/score/_atom_type_dependent_term.py#L154), with [the per-block CUDA slice bound at line 141](https://github.com/uw-ipd/tmol/blob/0f4c3bc426bca78e8681f0b730fa23c3e26ef261/tmol/score/_atom_type_dependent_term.py#L141).

> Could heavy-atom selection reuse the host indices and host hydrogen flags already available here? This repeats type-name lookup and converts one device boolean to Python per atom. The slice bound also reads a device scalar once per block. Expanded residue databases multiply this common setup cost across thousands of atoms before scoring begins.

This path predates the PR. The optimization uses the current resolver's freshly computed host indices and static residue lengths, preserving behavior even for a fresh packed set built from shared blocks with another atom-type ordering. A separate profiler counts 4,968 native scalar reads before and zero after on 230 default types / 4,738 atoms. Seven paired warm sets show 19.863→7.646 ms CPU and 72.613→10.255 ms CUDA, with every annotation exactly equal. These are common setup measurements, excluding term construction and scoring; they do not resolve the separate stale-cache identity issue when reusing an already annotated packed set.

### 57. P1 — peptide junction substitutions do not validate chemical roles

[tmol/ligand/_polymer_profile.py:1018](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_polymer_profile.py#L1018), and the lower-side mapping at line 1025.

> Could these mappings require a carbonyl carbon or an amine nitrogen, with the expected bond orders? The upper side chooses the first oxygen, including a single-bonded leaving hydroxyl. The lower side maps every non-`N` connection atom to peptide nitrogen: real 5CM, 8OG and PSU profiles produce `{"N": "P", "CA": "O5'"}`. Could this consume the final prepared graph? A removed hydrogen or hydroxyl can discard an otherwise valid frame, while a heavy-atom-only input omits the subsequently generated amide hydrogen and its angle row.

Reproduced against the new helper with reordered synthetic atom arrays and three real nucleotide CCD components. Ordinary nucleotide links do not necessarily match these extraneous peptide rows; this is a parameter-role error, not a claim that their standard phosphodiester energies changed. The branch checks elements and bond orders on the final reconstructed residue graph, using its connection and mainchain fields directly. This eliminates separate input-profile, connection and retained-name arguments. Tests preserve canonical phosphate handling, verify prepared FGA carbonyl rows, and retain the generated `HN1–NX–+C` angle when a renamed nitrogen arrives without hydrogens.

### 58. P1 — a canonical connection name can hide a noncanonical junction frame

[tmol/ligand/_polymer_profile.py:1018](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_polymer_profile.py#L1018), also line 1025.

> Could the condition compare the entire junction frame rather than only `upper != "C"` or `lower != "N"`? A beta/gamma backbone can retain `C` or `N` while its directly bonded carbon has another name. The current early exclusion leaves the corresponding connection angle without its substituted row. Please test both sides and then a connection where both partner frames use noncanonical names; the latter still needs a solution for the canonical `+` names retained in these rows.

Reproduced for independently renamed upper and lower neighbors. The branch maps changed bonded frames even when the connection atom retains its canonical name. Independent Cartesian harmonic energy and gradient checks exercise the real FGA-to-peptide connection. Simultaneously renamed partners remain a broader connection-parameter coverage requirement; this single-frame change does not claim to solve them.

### 59. P2 — cap completion adds a second coordinate builder and dense tables

[tmol/io/details/_build_missing_leaf_atoms.py:860](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/io/details/_build_missing_leaf_atoms.py#L860), with the packed fields at line 550.

> Could the cap's third reference use the existing native unresolved-atom connection representation? This branch adds four packed fields (24 bytes per padded atom), then resolves each hydrogen's connection and frame through Python scalar reads. The existing native builder can follow one bond into the partner, which supplies the required plane without another coordinate-placement pass. Please retain the cap geometry and packing tests when consolidating these paths.

The improvement branch uses the native connection ancestor and preserves relative generated hydrogen dihedrals, setting the first cap hydrogen trans to the partner reference. Before reconciliation, construction and packing passed but the two equivalent NH2 hydrogen names were reversed relative to the new upstream convention. After alignment, all new upstream cap checks pass on CPU and CUDA. The omitted dense fields avoid the stated storage by construction; no end-to-end latency claim is made from this source-level comparison. CPU/CUDA evidence is recorded in the follow-up validation artifact.

### 60. P1 — nonbonded indices and pair exclusions retain a different configuration

[tmol/score/_atom_type_dependent_term.py:90](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/_atom_type_dependent_term.py#L90), with LJ/LK's corresponding guard at [line 65](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/ljlk/_ljlk_energy_term.py#L65).

> Could these annotations identify the chemical catalog and scoring ownership that produced them? Reordering identical atom-type definitions leaves the force field unchanged, but a reused packed set keeps the previous integer indices and indexes different LJ/LK parameters. Changing `rosetta_typed` similarly retains the previous pair exclusions. Please test both configuration orders, re-rendering an earlier term, and old modules after another configuration is prepared.

Reproduced with fixed coordinates and fresh-annotation controls, for whole-pose and weighted block-pair energies/gradients. The branch uses one current annotation per object, weak source identities and immutable configuration settings; renderers take returned snapshots. A separate test checks reordered catalogs and changed element flags on reused blocks and packed sets. Atom identity strings remain independent of the chemical catalog.

### 61. P1 — hydrogen-bond and LK-ball scorers read mutable shared annotations

[tmol/score/hbond/_hbond_energy_term.py:208](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/hbond/_hbond_energy_term.py#L208), and [LK-ball's block cache at line 81](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/lk_ball/_lk_ball_energy_term.py#L81).

> Could the donor/acceptor and LK-ball annotations follow their source databases and be captured when rendering? Disabling donor mappings on a reused pose leaves the first donor inventory active; changing LK solvation coefficients retains the first tiled parameters. Updating cache guards alone is insufficient: these wrappers read tables through `pose_stack.packed_block_types` on every forward, so updating that object would also change an older rendered scorer's parameters.

Reproduced in both configuration orders. The branch snapshots the relevant annotation records alongside the existing pose reference, without copying the pose or all its unrelated caches. Tests compare fresh versus reused whole-pose, block-pair and jagged rotamer scoring, including gradients and a new render after another configuration. Removing donors supplies an independent zero-energy/zero-gradient check for hydrogen bonds.

### 62. P2 — hydrogen-bond resolver caches are unbounded and cannot verify their owners

[tmol/score/hbond/_params.py:79](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/hbond/_params.py#L79), with a second cache at line 179.

> Could both resolver caches be bounded and retain weak references to both input databases? Their keys contain integer IDs only. Tables remain cached after sources disappear, and an eventual ID reuse cannot be distinguished from a valid hit. Both the chemical catalog and hydrogen-bond database determine the result; expiring either should remove the entry.

The shared weak-identity LRU now supports multiple owners. Each resolver retains at most 32 entries, validates both referents, and expires an entry when either source is collected. Tests exercise independent source changes, LRU eviction, retained output tensors after collection, and equivalent indexed/unindexed device spellings. Removing the memoizer also exposed an obsolete type annotation hidden from argument validation; both raw and patched chemical databases remain accepted.

### 63. P2 — residue annotation repeats catalog lookups and full parameter transfers

[tmol/score/hbond/_hbond_dependent_term.py:134](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/hbond/_hbond_dependent_term.py#L134), and [tmol/score/lk_ball/_lk_ball_energy_term.py:152](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/lk_ball/_lk_ball_energy_term.py#L152).

> Could donor/acceptor names be mapped once for the chemical catalog, then gathered for each residue? The two pandas lookups repeat across every block type. LK-ball also transfers its full type table to CPU once per residue. A host catalog prepared once per term would remove this repeated work while retaining configuration-specific tables.

The branch prepares the donor/acceptor catalog once, retains a name-based fallback for unregistered types, and gathers host arrays during annotation. LK-ball transfers its host type table once per term. The paired profiler checks every public annotation array for exact equality against the preceding implementation and reports constructor, annotation and warm-hit costs separately. These standalone-term timings include inherited setup and must not be added together to estimate whole-score-function performance; see the follow-up artifact for measurements and retained-array tradeoffs.

### 64. P2 — native rotamer counts can wrap before allocation checks

[tmol/pack/rotamer/dunbrack/dispatch.impl.hh:604](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/dispatch.impl.hh#L604), with the total scan at line 625 and library-size narrowing at [line 364](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/dispatch.impl.hh#L364).

> Could products, prefix sums and library-size conversion check the native index range? The count-only helper turns `65536 * 65536` into zero, `3 * 2**30` into a negative count, and four counts of `2**30` into a zero total with negative offsets. A later per-residue budget check cannot recover the true count after it has wrapped. Please cover overflow across scan blocks and reject it before allocating or indexing rotamer arrays.

This arithmetic risk is inherited, but the expanded chemistry and sampling paths still depend on it. The branch propagates an error marker through checked products and both native count scans, then raises before large allocation. It also validates 64-bit library sizes before narrowing. The checks reuse existing buffers and synchronization points. Small-table regressions cover product and total overflow, repeated overflow that would become positive again, negative inputs, empty inputs, exact capacity boundaries, and the public sampler's possible-library stage. Ordinary counts/offsets are compared exactly; the isolated native profiler measures validation overhead rather than claiming a speedup.

## 65. Validate nucleotide counts before narrowing or allocation — P2

Location: [`tmol/pack/rotamer/_na_chi_sampler.py:268`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_na_chi_sampler.py#L268).

> Could the nucleotide sampler retain wide counts until both each product and the total fit the native index range? Casting here turns a count of `2**32` into zero, which the following empty-result branch accepts. A wrapped positive total can also hide invalid negative counts. Please check the product, total and narrowing before allocating rotamer rows.

The branch keeps counts in int64, bounds invalid combination inputs with an out-of-range sentinel before multiplication, and narrows only after checking index capacity. The shared Python allocation helper now rejects negative counts, oversized individual counts and an oversized total, with one host transfer. Tests inject small count metadata into a real RNA sampler while blocking row allocation; they do not construct gigantic libraries. Existing budget behavior remains independently tested. The realistic default nucleotide products are small; this is an allocation-boundary guard, not evidence that ordinary RNA fixtures need billions of states.

## 66. Key Dunbrack sampler annotations by their parameter source — P1

Locations: [`tmol/pack/rotamer/dunbrack/_dunbrack_chi_sampler.py:102`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/_dunbrack_chi_sampler.py#L102) and [`:241`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/_dunbrack_chi_sampler.py#L241).

> Could both sampler annotation levels check which parameter resolver produced them? Two resolvers with different residue-to-library mappings currently reuse the first `dun_sampler_cache` on shared chemical types. Repeated sampling then depends on which resolver was used first. Please compare shared-type sampling against fresh annotations in both resolver orders, including returning to an earlier sampler.

This includes an inherited cache assumption; the PR's expanded reference/library machinery makes it particularly relevant. The regression changes ILE's mapping to the two-chi LEU library without mutating the default resolver. Fresh samplers produce distinct results, while stale shared annotations incorrectly preserve the earlier mapping. The branch stores one annotation per RT/PBT with a weak resolver-identity key, and sampling refreshes its own annotations even without a separate setup call. Resolver data remain immutable inputs; publish a new resolver after changing tables.

Small name/index/chi metadata are read on the host once per sampler/resolver, replacing per-residue device-tensor lookups and scalar synchronization. Table IDs in RT metadata now have their declared Python-integer representation. A redundant probability-selection branch is reduced to its common 0.98 value. Sampler equality compares resolver identity only with another sampler, so an integer equal to the object's hash cannot compare equal. The scoring-term annotations and global resolver-cache lifetime are separate audit items; this fix does not claim to resolve them.

## 67. Include polymer types that sample their own heavy chi — P1

Location: [`tmol/pack/rotamer/dunbrack/_dunbrack_chi_sampler.py:447`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/_dunbrack_chi_sampler.py#L447).

> Could this selection use the sampler's buildability predicate instead of requiring a library index? `defines_rotamers_for_rt` explicitly accepts amino-acid polymers with their own heavy-chi samples and no Dunbrack library, and the native code has a no-library path. This filter removes those types before the native sampler can enumerate their samples.

A small probe gives an allowed polymer type three explicit chi1 means, clears its library reference, and verifies that the sampler advertises it as buildable. It receives zero rotamers while ten other allowed types receive samples. The concrete task's target mask is checked explicitly. The branch now uses its buildability predicate, retains gaps before later chi slots, and derives no-library sidechain roots from the chi actually sampled. Tests check explicit Cartesian products, expansion offsets, two-pose masks, budget failure, requested coordinate torsions and an unchanged input chi1 when only chi2 is sampled. The fixtures use private copies of ILE's valid topology, not newly fitted chemical parameters. The slot-count issue is at [line 540](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/_dunbrack_chi_sampler.py#L540); counting defined atoms cannot represent a hole before a later chi.

## 68. Refresh mainchain-copy fingerprints when sampler ownership changes — P1

Location: [`tmol/pack/rotamer/_mainchain_fingerprint.py:359`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_mainchain_fingerprint.py#L359).

> Could fingerprint reuse check the sampler's actual sidechain roots and chemical element definitions, rather than only its class name? Changing which chi a sampler owns changes which input degrees of freedom must be copied. A cached fingerprint can therefore corrupt an unsampled torsion even after the rotamer counts and library lookup are correct.

An independent probe first builds a private polymer type through a library, then reuses the same pose with a resolver that supplies only explicit chi2 samples. Both shared and fresh runs return two states, but their supposedly frozen chi1 is −0.08948 versus 1.05183 radians. The shared type retains the earlier six-atom mainchain fingerprint; the fresh one has thirteen copied atoms. The follow-up now keys retained regions by actual roots and weak chemical-owner identity, represents individual sampler instances, and uses the union of source regions so nonnested selections can both preserve their input geometry. Current annotations replace old entries instead of retaining every past sampler.

The PBT early-return check at [line 410](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_mainchain_fingerprint.py#L410) looks for `mc_atom_mapping`, while this function stores `mc_fingerprints`. A corrected cache must validate the underlying fingerprints as well as avoiding redundant reconstruction.

## Validation record

See [VALIDATION.md](VALIDATION.md) for the initial review commands, counts, environment, examples, and remaining failures; [FOLLOWUP.md](FOLLOWUP.md) records subsequent fixes and validation. All raw run logs were retained separately under `/mnt/home/kdidi/tmol-pr503-results`. No upstream golden score files were regenerated to make tests pass.

Comment 68 is fixed and checked with both sampler orders, two differently configured instances in one two-pose task, chemical-owner changes, zero-state samplers, nonnested regions, and released sampler/database owners. The initial broad CPU/CUDA run passes 426 cases; the final optimized source passes 75 focused CPU/CUDA cases. Across 230 types, all 121,440 default three-sampler source/target transfer maps match the prior implementation exactly where a source exists. Default fingerprint setup improves 1.45× on CPU and 1.40× on CUDA, with 4.1% fewer packed tensor bytes. The two-sampler configuration has 4.8–7.2% slower cold setup and 2.8% more packed tensor bytes, while repeated setup improves. See [results/fingerprint-validation.json](results/fingerprint-validation.json) for separate source stages and measurement limits.

## 69. Bound the Dunbrack resolver cache and release private databases — P2

Location: [`tmol/score/dunbrack/_params.py:150`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_params.py#L150).

> Could this resolver cache use a bounded weak-owner policy? Its memoizer holds the whole database as a key, so every independently generated library and its derived CPU/CUDA tables remain alive after the preparation/scoring task ends. Please also normalize equivalent device spellings and include the resolver class in the key.

This cache dates to 2019; it is an inherited assumption made more consequential by per-input chemical databases. The follow-up reuses the shared weak-identity LRU with four entries. The compiled-table construction is unchanged. Five pre-fix regressions cover source/output retention, live-owner eviction, device aliases and subclass identity in both orders. All 49 tensor fields and three lookup DataFrames match exactly. The measured private default-sized resolver retains **67,494,320 bytes** of derived tensor storage before the fix and **zero** after its callers release the database and resolver. Active consumers remain valid. Four entries bound cached resolver count, not arbitrary library sizes or total process/GPU memory.

The CPU resolver/scoring/sampler suite passes 60 tests (58 CUDA skips); Slurm 249841 passes 210 tests without skips on CPU/CUDA. See [results/dun-resolver-validation.json](results/dun-resolver-validation.json) for table equality, lifetime evidence and terminal accounting. Scoring-annotation identity is a separate defect and is not corrected by changing the resolver cache.

## 70. Keep Dunbrack scoring annotations tied to their resolver — P1

Location: [`tmol/score/dunbrack/_dunbrack_energy_term.py:91`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_dunbrack_energy_term.py#L91), packed guard at [line 212](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_dunbrack_energy_term.py#L212), and renderer at [line 297](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_dunbrack_energy_term.py#L297).

> Could both annotation levels validate the resolver that produced their values, and could rendering refresh the calling term's own packed annotations? A second parameter set can remap or remove a residue's library while the shared chemical types still carry the first term's table IDs and offsets. Rendered modules also need to retain their own tensors after another term updates the shared packed set.

This is an inherited scoring-cache assumption, separate from sampler and global resolver caching. Nine regressions reproduce wrong energies on shared types or stale RT annotations after direct PBT setup. Remapping ILE to the two-chi LEU library or removing its lookup is compared with independently annotated fresh jagged poses, in both orders and whole-pose/block-pair modes. The follow-up keeps one weak resolver-keyed annotation per RT/PBT, refreshes the caller at render time and retains each rendered module's tensor arguments. Tests also compare coordinate gradients, permit owner expiry without invalidating a rendered scorer, reject duplicate lookup names and prohibit per-residue device-scalar reads.

Small lookup metadata are copied to the host once per term; whole annotation tables are assembled there before one transfer per field. All 2,760 RT fields and 12 packed tensor fields match exactly over 230 default types; packed storage remains 76,360 bytes. The complete scoring/resolver CPU suite passes 25 tests (25 CUDA skips). Slurm 249931 passes 119 CPU/CUDA cases, including mirror-image scoring, D repacking and noncanonical/conjugated-group packing. See [results/dun-scoring-validation.json](results/dun-scoring-validation.json) for source stages, scheduler preemption/restart accounting and setup performance limits.

## 71. Keep DOF-copy indexing on device and repair its source oracle — P2

Location: [`tmol/pack/rotamer/_chi_sampler.py:155`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_chi_sampler.py#L155), repeated host lookups at [line 237](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_chi_sampler.py#L237) and [line 295](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_chi_sampler.py#L295), source test at [line 974](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/tests/pack/rotamer/test_build_rotamers.py#L974).

> Could this copy plan use one shared device copy of the residue-to-kinforest atom map? It currently transfers per-conformer index arrays to the CPU for NumPy lookup, then copies the result back. The source assertion also compares `src_gold` with itself; please compare it with actual source indices and build considered-type IDs from the concrete task.

The follow-up uses direct device gathers, masks missing atoms, releases unused atom-index arrays before the next gather and adds offsets in place to newly gathered buffers. Its KFO table is shared with existing chi correction. The production code is 123 lines shorter. Both old fixtures used invalid assumptions about considered-type IDs; correcting those inputs and the source oracle makes named source atoms and offsets independently checked. The old implementation and the new one both pass those corrected oracles before benchmarking. New checks cover selected/reversed conformer order, empty states without annotations, no retained regions, absence of explicit GPU-to-CPU index transfer and shared table identity.

The final CPU suite passes 44 tests (39 CUDA skips). Slurm 250036 passes 436 broad CPU/CUDA cases, and 250127 passes all five mirror-image/D-repacking cases. Final paired latency and allocated-memory measurements use the corrected fixtures; the earlier prototype measurements use different fixture mappings and are kept separate. See [results/dof-copy-validation.json](results/dof-copy-validation.json). This is an index-construction optimization, not a whole-packer speed claim.


## 72. Keep chi assignment and ring-offset lookup on the tensor device — P2

Location: [`tmol/pack/rotamer/_chi_sampler.py:367`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_chi_sampler.py#L367), repeated chi-index copies at [line 373](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_chi_sampler.py#L373) and [line 392](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_chi_sampler.py#L392).

> Could this assignment reuse the shared device KFO map and cache one device copy of the static ring offsets? It currently copies the selected block types and the chi-atom indices to the host three times per call, then transfers the gathered results back. A single sparse selection also removes the full arange/division temporary and supports noncontiguous chi columns.

The follow-up preserves exact old/new DOF outputs on the paired profile inputs, with independent PRO ring-offset and untouched-DOF checks. It reduces the two production files by 19 lines. CPU passes 33 tests (32 CUDA skips); Slurm 250227 passes 446 broad CPU/CUDA cases. Paired latency, additional CUDA allocation peaks, persistent-table tradeoffs and stage limits are recorded in [FOLLOWUP.md](FOLLOWUP.md) and [results/chi-assignment-validation.json](results/chi-assignment-validation.json). This is an assignment-stage optimization, not a whole-packer speed claim.

## 73. Validate reflected conformer sets, terminal defaults and probability ordering — P1

Location: [`tmol/pack/rotamer/dunbrack/dispatch.impl.hh:210`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/dispatch.impl.hh#L210), bin selection at [line 463](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/dispatch.impl.hh#L463) and [line 736](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/dispatch.impl.hh#L736), coverage at [`test_mirror_image_scoring.py:83`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/tests/score/test_mirror_image_scoring.py#L83).

> Could the mirror test compare the offered conformer sets and packing energy tables, with chirality-aware missing-torsion defaults and consistent reflected bin boundaries? On the follow-up branch, the exact L/D fixture pair receives 1,308 versus 1,265 conformers on both CPU and CUDA. Fifteen residue/type groups differ in count; another ten equal-count groups have nonmatching named heavy-atom conformers. Keeping the final structure D does not detect these omissions.

These counts use the follow-up branch’s 0.98 coverage for both library classes; they are not a rerun of the untouched upstream defaults. The anchored bin/default logic is unchanged. The native sampler uses the same −60°/+60° fallback for missing L and D backbone angles. Its floor-based ordering-cell lookup also conflicts with reflecting the sorted tables as grid points. An isolated shift of only the D ordering cells removes 14 of 15 count mismatches, leaving terminal ARG at 51 L / 17 D states; this diagnostic does not settle exact boundary conventions and is not integrated as a fix. New per-term whole-pose/block-pair energy and reflected-gradient tests pass on CPU/CUDA (Slurm 250251: nine cases), but the full packing gate remains false. [check_mirror_packing.py](check_mirror_packing.py) uses one-to-one assignment, retains unmatched counts, and distinguishes named heavy-atom geometry from unclassified hydrogen permutations. The structured CUDA diagnostic is Slurm 250256; see [results/chi-assignment-validation.json](results/chi-assignment-validation.json). This defect was open at that stage; the follow-up validation below supersedes that status.

## 74. Wrap the positive periodic endpoint before sorted-table indexing — P1

Location: [`tmol/pack/rotamer/dunbrack/dispatch.impl.hh:457`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/dispatch.impl.hh#L457) and the repeated logic at [line 730](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/dispatch.impl.hh#L730).

> Could both sampling paths share a periodic-coordinate helper that maps the upper endpoint back into the valid bin range? With the actual float32 starts, steps and periods, phi or psi at +π produces `wrap == period`, bypasses this strict `>` loop, and selects bin 36 in a 36-bin table. The probability interpolator is periodic, but the sorted-index lookup happens first and is not wrapped.

[reproduce_dun_periodic_boundary.py](reproduce_dun_periodic_boundary.py) proves the wrong index safely on CPU and CUDA (Slurm 250349): it pads the lookup to 37×37 and puts a different valid PHE rotamer in the extra row/column. Equivalent −π/+π phi inputs then return approximately 0.752777/0.081837 instead of the same probability; psi returns 0.220622/0.024137. The normal database has only 36×36 bins. No actual out-of-bounds read or crash is executed by this guarded diagnostic. The same unchecked index arithmetic appears in chi reconstruction. This inherited endpoint defect is open and should be covered together with, but separately from, D/L cell-reflection semantics. Evidence is retained in [results/chi-assignment-validation.json](results/chi-assignment-validation.json).


Comment 74 follow-up: both native paths now share a helper with an exclusive
upper endpoint and a guard against division rounding up to the bin count.
Four probability/chi × phi/psi regressions fail before and pass after the fix.
The CPU native suite passes 51 tests (49 CUDA skips), and Slurm 250352 passes
188 CPU/CUDA cases. See [results/periodic-lookup-validation.json](results/periodic-lookup-validation.json).
Comment 73's reflected-cell and terminal-default behavior is corrected in the subsequent mirror-packing follow-up below.


## 75. Reflect the chi-mean branch before spline fitting — P1

Location: [`tmol/score/dunbrack/_params.py:374`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_params.py#L374), and scoring at [`potentials/potentials.hh:415`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/potentials/potentials.hh#L415).

> Could mirrored means use the reflected angular branch before fitting, with scoring measuring a periodic chi-minus-mean difference? Moving every mean below −120° toward +180° adds a full turn to only some grid points of a reflected library. Interpolation then produces a different conformation, even after the probability-ordering cells and terminal defaults agree. A periodic difference in the scorer is needed when the mean uses the reflected branch.

After correcting only the ordering cells and defaults, the mirror fixture has equal conformer counts but PHE heavy atoms still differ by 0.657 Å (0.943 Å with expanded chi). A synthetic 110°/130° mean grid reproduces the branch defect against the exact previous fitting method. The follow-up carries explicit reflection metadata with each library, preserves it through renaming/serialization, and unwraps D means toward −180°. Both scoring paths now wrap the actual chi-minus-mean difference. Adding −2, +1 or +3 full turns to private mean coefficients preserves whole-pose and block-pair energies and weighted coordinate gradients. CPU numerical scoring references and gradient checks pass; the 289-case CPU/CUDA run in Slurm 250427 covers these fixes before the subsequent glycine changes.

## 76. Rebuild both glycine alpha hydrogens and define the symmetry option completely — P1

Location: [`tmol/pack/rotamer/_fixed_aa_chi_sampler.py:55`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_fixed_aa_chi_sampler.py#L55); symmetry option at [`tmol/database/__init__.py:64`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/database/__init__.py#L64).

> Could fixed glycine sampling rebuild HA2 and HA3 together? Preserving HA2 while rebuilding HA3 in the fixed ideal convention puts the two hydrogens only 0.081 Å apart in the reflected fixture. Once both are rebuilt, exact mirror packing also exposes unequal ideal C–H lengths (1.090168/1.089353 Å) and bonded targets (1.09017/1.08935 Å). Should the existing symmetric-glycine option average those equivalent hydrogen parameters as well?

The follow-up rebuilds both alpha hydrogens, preventing the overlap and arbitrary inheritance of a glycine hydrogen during design. In the opt-in symmetric database, it averages their ideal lengths and harmonic targets/force constants, including terminal forms, without editing the default YAML parameters. Other residue objects remain shared, and repeated symmetrization preserves values and cache content IDs. The pre-fix L fixture passes the hydrogen geometry check while D fails. Full-atom one-to-one matching permits only hydrogen exchanges with identical chemical types and named neighbors; with the opt-in correction, every per-term packing interaction table agrees on CPU. Slurm 250496 passes 311 CPU/CUDA cases (one skip for CPU-only invocation of the CUDA annealer), including per-term packing matrices and an actual single-position packing result compared with exhaustive whole-pose scores. Slurm 250506 passes four exhaustive grid-sweep cases. See [results/mirror-packing-validation.json](results/mirror-packing-validation.json). This does not validate the underlying statistical or bonded parameters scientifically.


Comment 73 follow-up: reflected ordering cells, per-library missing-torsion
defaults, mean interpolation, and the glycine hydrogen defects are now fixed.
The fixture receives **1,308 conformers on each side**, with one-to-one full-atom
matching and all 24 configured score components' interaction matrices agreeing
on CPU/CUDA (components for absent chemistry remain zero). Expanded-chi counts
and heavy geometry pass separately. Actual CUDA packing of one movable PHE
reaches the exhaustively enumerated minimum for both inputs and produces
reflected final coordinates. This validates that controlled packing task,
not identical random trajectories for arbitrary multi-position packing.
See [FOLLOWUP.md](FOLLOWUP.md) and the stage-separated
[manifest](results/mirror-packing-validation.json).


## 77. Generate missing D libraries from requested mappings, not a name-prefix guard — P1

Location: [`tmol/database/scoring/_mirrored_dunbrack.py:177`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/database/scoring/_mirrored_dunbrack.py#L177), unconditional duplication at [line 196](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/database/scoring/_mirrored_dunbrack.py#L196).

> Could this check which requested residue mappings are missing and mirror only the libraries those mappings need? Any covered table whose name begins with `d` currently disables all generation, including an ordinary source called `different_original` or a database with only one D type already added. Conversely, requesting no D types or only DALA still duplicates every library. Existing explicit target mappings should remain authoritative, and generated-name collisions should fail clearly.

Eight regressions fail before the fix. The follow-up supports incremental requests, retains existing library objects/mappings, returns the original database when no new library is required, and rejects missing source tables, competing target requests and ambiguous generated names. Selective ARG/DARG/PHE native sampling matches the full database exactly on CPU/CUDA even though PHE's table index moves. All default generated values remain exactly equal to the previous function. Final Slurm 250526 passes 54 CPU/CUDA cases with one intentional CPU annealer skip, including full-atom mirror energies and actual single-position packing.

Seven alternating warm timing rounds show DARG-only generation **17.28 → 1.65 ms**, with additional owned tensor storage **30,927,608 → 4,279,200 bytes**. DSER-only is **17.29 → 0.089 ms** and **30,927,608 → 77,784 bytes**. The full default request is essentially unchanged (**15.79 → 15.86 ms**, identical tensor storage); a no-D request now allocates no new tensor storage. These measurements exclude the source L tables, resolver fitting, transient peaks and whole-program work. See [profile_mirrored_library_generation.py](profile_mirrored_library_generation.py) and [results/mirrored-library-generation.json](results/mirrored-library-generation.json).


## 78. Support rotamer-only and empty library databases through the native boundary — P2

Location: [`tmol/score/dunbrack/_params.py:605`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_params.py#L605), empty lookup construction at [line 279](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_params.py#L279).

> Could absent statistical-library families retain their actual zero length with the correct tensor ranks? A valid LEU-only database fails while packing the empty semirotameric coefficient list. An entirely empty database also fails when constructing lookup columns and offsets. Once those are corrected, an entirely unmapped scorer reaches the native interface with zero-stride dihedral metadata. Explicit ligand/polymer chi sampling should not require unrelated amino-acid tables to be installed.

These are inherited full-database assumptions exposed by private/minimal chemistry. Five pre-fix cases fail; semirotameric-only controls pass. The follow-up shares lookup construction, creates rank-correct empty tensors and true empty offsets, handles empty prefix sums, and constructs empty dihedral tensors with native-compatible strides. It introduces no dummy tables or dihedrals. Whole-pose/block-pair energies and weighted coordinate gradients match the full database restricted to the same mappings, including exactly zero outputs for an unmapped term. Public explicit-chi coordinate construction also works with no statistical library, including gapped chi numbering.

All 50 default resolver tensor fields and three lookup DataFrames match the preceding implementation on CPU/CUDA. Derived tensor storage is 374,345 bytes for LEU-only, 1,308,573 for PHE-only, and zero for no libraries; these are deliberately different reference sets, not automatic pruning of a full model. The full default remains 67,494,212 bytes, 144 fewer because two views can share integer offsets. Slurm 250598 passes 297 CPU/CUDA cases (one intentional CPU annealer skip); the separately added public-construction cases pass eight CPU/CUDA tests in 250599. See [check_empty_dunbrack_database.py](check_empty_dunbrack_database.py) and [results/empty-dunbrack-libraries.json](results/empty-dunbrack-libraries.json).


## 79. Register reflected grids at their actual coordinates — P2

Location: [`tmol/database/scoring/_mirrored_dunbrack.py:84`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/database/scoring/_mirrored_dunbrack.py#L84), related semirotameric origins at [`_params.py:614`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_params.py#L614).

> Could the reflected table carry the reflected grid origin, and could semirotameric scoring use its declared backbone origins and spacing? Rounding `-2*start/step` while retaining `start` relabels points when reflection does not land on the original grid. Independently, a circular reindexing of an equivalent PHE table changes the score because its backbone grid is hardcoded to −180°/10°.

The follow-up permutes the existing data and sets the reflected origin to `-start-c*step`, where `c` is the integer permutation shift. It retains the source origin for discrete sampling-cell selection, including missing-angle defaults, and persists that metadata through serialization. Default aligned grids reuse their existing origin storage. Reindexing preserves whole-pose/block-pair scores and weighted coordinate gradients; custom origins and rectangular 20°/30° grids satisfy mirror checks. This is a metadata correction without resampling or a scientific refit. See [results/grid-registration-validation.json](results/grid-registration-validation.json).


## 80. Read each table's metadata and respect padded spline strides — P2

Location: [`tmol/pack/rotamer/dunbrack/dispatch.impl.hh:484`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/dunbrack/dispatch.impl.hh#L484). Related inherited interpolation code: [`tmol/numeric/bspline_compiled/bspline.hh:392`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/numeric/bspline_compiled/bspline.hh#L392).

> Could these access the selected metadata row directly, and could interpolation use the supplied tensor strides? A vector view's stride is measured in vectors, but the current pointer arithmetic applies it after converting to a scalar pointer. Square grids conceal the wrong row/axis selection. Interpolation then derives contiguous strides from logical sizes even though differently sized tables are packed with padding.

These are inherited assumptions exposed by generalized libraries. The follow-up uses `table_metadata[index].data()` at all Dunbrack scoring/sampling call sites and uses actual strides in spline interpolation. Fitting creates a contiguous owned coefficient buffer, as its in-place filter requires. Independent 2D/3D/4D tests compare padded and transposed coefficients against contiguous values and derivatives exactly. Mixed default/custom score tables and rectangular native sampling exercise the integration. The stride and metadata fixes must be applied together: making interpolation respect previously misaddressed strides exposes failures even with default tables. No score golden was refreshed for these fixes.


## 81. Correct periodic fitting for small spline grids — P2

Related inherited location, outside the PR's changed lines: [`tmol/numeric/bspline_compiled/bspline.hh:98`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/numeric/bspline_compiled/bspline.hh#L98). This is a general review comment or a separate numeric fix, not an inline comment on the PR diff.

> Can the short-period anticausal initialization stop before re-reading its own accumulator? The last entry is already included as the initial term. Iterating through it again both uses a partially modified value and advances the denominator to the wrong pole power. On small grids, interpolation consequently misses the supplied grid values.

After correcting layout handling, three independent 2D/3D/4D tests still fail on 5–8-point axes. Iterating over the other `N-1` entries gives the periodic denominator `1-pole**N` and restores the grid-point interpolation checks. The large-grid truncation branch is unchanged. All three issues and their separately failing intermediate stages are recorded in [results/grid-registration-validation.json](results/grid-registration-validation.json); default parameter tensors are independently checked for exact equality.


## 82. Reuse periodic indices inside spline interpolation — P2 performance suggestion

Related inherited location, outside the PR's changed lines: [`tmol/numeric/bspline_compiled/bspline.hh:392`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/numeric/bspline_compiled/bspline.hh#L392). This is a general review suggestion for the shared numerical primitive.

> Could interpolation reuse each axis's wrapped indices across the tensor-product contributions? The same few index remainders are recomputed for every contribution. A small offset cache helps CPU and 2D CUDA; larger CUDA stencils can retain only wrapped base indices to avoid an extra local-memory array. The choice should be checked in full scoring kernels, including register/local-memory costs, rather than inferred from a standalone kernel.

The follow-up preserves the floating-point accumulation order and adds independent periodic linear-system checks, including one- and two-point axes. The integrated GPU suite passes 388 CPU/CUDA cases with one intentional CPU annealer skip. In three alternating-checkout trials with 64 ubiquitin poses, CPU Dunbrack forward/forward-plus-gradient improves about 22%/19%; backbone forward improves about 17%. GPU Dunbrack forward improves about 11%, while its gradient workload and backbone scoring are essentially unchanged. These are individual term timings with construction excluded, not full application speedups.

Managed GPU allocation peaks are identical. The selected strategy avoids the first prototype's doubled local memory in the isolated 3D kernel. Driver-compiled copies of the actual H200 scoring PTX mostly use fewer registers, but one float32 Dunbrack pose-forward kernel uses 112 rather than 96 bytes of local memory per thread (142 → 127 registers). This small tradeoff is retained and reported; no universal reduction in GPU memory is claimed. Full CPU outputs are exact; CUDA differences are within tolerance and no larger than repeated baseline-process variation. See [results/spline-index-performance.json](results/spline-index-performance.json).


## 83. Register each sampler object once and combine its enabled regions — P2

Related inherited registration location: [`tmol/pack/_packer_task.py:424`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/_packer_task.py#L424). The new mask-disabling API relies on this registry at [`tmol/pack/_packer_task.py:447`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/_packer_task.py#L447).

> Can registration preserve one entry per sampler object and combine its enabled regions? Adding the same object for masks A and B appends it twice but overwrites its identity-to-column lookup. Both sampling calls then read B: A's residues disappear and B's residues receive duplicate conformers. Disabling that object also addresses only its latest column. A palette that returns a reusable sampler list exposes another alias: adding a sampler mutates the palette's list.

The follow-up owns the task's list, deduplicates defaults by object identity, and unions masks when an existing sampler is enabled again. Enabling it everywhere fills its existing mask column. Equal but distinct sampler objects remain independent. Repeated registration does not grow the mask tensor or duplicate downstream sampling/allocation. Five CPU regressions fail before the change: four expose wrong sampling multiplicities and the distinct-instance control exposes palette-list mutation. Tests use actual IncludeCurrent sampling on a ragged two-pose batch and check the expected per-residue counts, then exercise disabling/re-enabling without mask reallocation. This is an inherited API defect relevant to reusable group-packing configuration, not a claim that the PR introduced registration itself. See [results/sampler-registration-validation.json](results/sampler-registration-validation.json).


## 84. Reject unresolved or ambiguous library mappings before fitting — P2

Related inherited location: [`tmol/score/dunbrack/_params.py:283`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/dunbrack/_params.py#L283). This is a general review comment on a lookup contract used by the expanded private/mirrored libraries.

> Can the resolver validate declared targets and name uniqueness before preparing tensors? `get_indexer` turns a misspelled or omitted target table into `-1`, which scoring interprets as an unsupported residue and silently omits. Duplicate residue keys are rejected only by later consumers, after expensive table preparation, while duplicate table names produce a pandas-specific error. Deliberately omitting a residue's lookup row is a different case and should remain supported.

The follow-up validates residue-key uniqueness, table-name uniqueness across both families and every declared target before the first derived tensor helper. It builds one mapping and derives each family's local indices from that mapping, eliminating repeated conversion and indexing. Per-family `-1` entries, empty families and many residue aliases sharing one table remain valid. Six new invalid-input cases fail before the fix; an independent reordered-alias oracle preserves the intended family indices.

The CPU scoring suite passes 50 cases / 49 CUDA skips. Slurm 250855 passes 247 CPU/CUDA cases / one intentional CPU annealer skip, including sampling, mirrored libraries and packing. All 51 default tensor fields and three lookup DataFrames are exact on CPU and CUDA; derived tensor storage is unchanged. Default lookup construction improves 0.664 → 0.288 ms; a synthetic 10,040-row alias case improves 8.365 → 1.898 ms. These are metadata timings, excluding fitting and scoring. Traced peak memory falls in both cases, but retained Python memory rises slightly for the large case because uniqueness validation caches the index's lookup engine. See [results/dun-lookup-validation.json](results/dun-lookup-validation.json).


## 85. Share current-conformation copying and avoid padded index work — P2 performance suggestion

Related inherited locations, outside the PR's changed lines: [`tmol/pack/rotamer/_include_current_sampler.py:102`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_include_current_sampler.py#L102), [`tmol/pack/rotamer/_include_current_sampler.py:123`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_include_current_sampler.py#L123), and [`tmol/pack/rotamer/_fallback_sampler.py:150`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/pack/rotamer/_fallback_sampler.py#L150). This is a general suggestion for shared packing code affected by larger and more heterogeneous chemical databases.

> Can the two samplers share their identical copy method, remove the unconditional global CUDA barriers and enumerate only the atoms being copied? The existing planner builds padded index matrices using the database-wide maximum residue size. A large unrelated ligand type consequently increases temporary work for small protein residues. The CUDA barriers also run for a CPU task whenever any GPU is available.

The follow-up derives source offsets without compacting padded pose rows and enumerates copied atoms from their counts. It reuses a fresh index buffer for source atom indices and preserves both copy order and virtual-root indexing. The fallback sampler shares the include-current fill method while retaining its own selection policy and class identity. This removes 76 net production lines. Empty inputs return before accessing annotations. Independent tests cover explicit source/destination indices, ragged pose layouts, reordered selections, unchanged input metadata, exact DOF copying and non-default CUDA streams. Real protein rotamer coordinates match the old implementation exactly.

In the final synthetic 12,000-conformer test with an unrelated 1,024-atom type, GPU planner time falls 0.498 → 0.185 ms and extra allocated peak falls 118,974,464 → 6,471,680 bytes. With a 40-atom database maximum, the peak falls 12,695,040 → 6,471,680 bytes. Real 16-pose ubiquitin DOF filling improves 1.060 → 0.671 ms on CPU and 0.381 → 0.220 ms on GPU. Complete include-current-only construction changes much less (about 1–4% in these warm rounds), and its GPU allocation peak is unchanged. These are not full sampling/packing/scoring speedups. The first whole-build peak measurement was invalidated by garbage collection; the corrected measurements collect garbage before recording each allocation baseline. See [results/current-copy-validation.json](results/current-copy-validation.json).


## 86. Define guarded replacement semantics for reusable parameter bundles — P2 design question

[`tmol/ligand/_registry.py:399`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_registry.py#L399).

> Should a parameter bundle be able to explicitly replace an existing exact residue definition? The current name filter silently skips it. A coupled attachment correction needs its atom types, charges, bonded terms and hydrogen construction to arrive together. An opt-in baseline digest would let the loader distinguish a compatible update, an identical reload and an incompatible chemical baseline. Combined bundles also need to preserve their original patch parameters without applying those values over an already corrected residue.

This is an API/design extension, not a claim that ordinary duplicate definitions should overwrite existing types. The follow-up adds explicit guarded replacements in `.tmol` version 4; older readers reject that version. Versions 1–3 retain their existing addition semantics. The private coupled generator and ordinary bundle injector now share one installation path. The guard normalizes declared scalar types and NumPy/Python strings so serialization does not change baseline identity. Patch order is preserved when loading shared metadata.

Biotin, N-glycan and O-glycan tests cover fresh/prepared/corrected databases, reversed bundle order, changed baselines, missing records, conflicting connections, exact built coordinates, and native scores/gradients. The bonded database is built and hashed once instead of twice. [PARAMETER_BUNDLES.md](PARAMETER_BUNDLES.md) explains the API, format and fingerprint scope; [results/parameter-replacement-validation.json](results/parameter-replacement-validation.json) records validation and installation-only performance. This does not choose a default attachment force field or validate the provisional MMFF charge model.


## 87. Coalesce repeated batch definitions and reject contradictory sources — P1

[`tmol/ligand/_registry.py:399`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_registry.py#L399), with charge collection at [`tmol/ligand/_registry.py:452`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_registry.py#L452).

> Can a batch establish one definition per residue name before patching and injection? `existing_names` excludes names newly added by the same batch, so overlapping files append the same residue repeatedly. Conflicting core definitions can then leave duplicate chemical rows but last-source charge/bonded records. Shared charge maps also replace one another by residue name, and atom-type element selection depends on source order. Identical definitions should coalesce; conflicting definitions or atom-charge assignments should raise; compatible disjoint metadata should merge.

Fifteen regressions fail on parent `c8efb34ca`: repeated preparations/files, conflicting core definitions, disjoint/conflicting shared charges, conflicting YAML charge rows and complementary/conflicting element maps. The follow-up checks complete definitions once per name while retaining all sources' shared metadata in order. Repeated paths are parsed once within a batch; distinct files are still read and checked. Export emits one matching definition and combines disjoint atom-charge entries. Conflicts fail even when the target already exists.

The audit also found a **follow-up regression**, not an upstream claim: merging additional charge deltas could mutate a frozen preparation's nested dictionary. A separately failing input-immutability test now passes after copying only the map being updated. Guarded replacements validate contradictory original metadata before filtering values unnecessary for an already corrected target.

The final CPU suite passes 124 cases / 11 CUDA skips. Slurm 250976 passes 233 CPU/CUDA cases / 8 skips; an additional 12-case H200 run checks distinct overlapping files through public preparation, exact named inventories, pose construction, scoring and gradients for all three attachment fixtures. In a repeated-file benchmark, 100 copies previously added 100 definitions and took 277.772 ms; the corrected batch adds one and takes 9.750 ms. A single file is essentially unchanged (9.351 versus 9.426 ms). This measures duplicate-input parsing/installation, not general preparation or scoring performance. See [results/batch-identity-validation.json](results/batch-identity-validation.json).


## 88. Preserve declared elements for custom atom types through parameter export — P1

[`tmol/ligand/_params_file.py:238`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_params_file.py#L238), with the fallback at [`tmol/ligand/_registry.py:212`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_registry.py#L212).

> Can `.tmol` preserve `atom_type_elements` instead of reloading it as `None`? For an unfamiliar type, the lenient fallback assumes carbon unless the name starts with H or the lookup marks it polar H. A declared nitrogen, sulfur, chlorine or hydrogen type with an unrelated name consequently becomes carbon after export/reload; strict registration instead fails. The element should come from the declared chemistry, and an incompatible declaration for an existing type should raise rather than be ignored. New types introduced only by a patch also need registration before patch construction.

The old-writer diagnostic reproduces all four element changes. Version 5 now carries the shared declaration map, preserves it through ordinary and guarded bundles, and is rejected by the version-4 reader. Older formats remain supported when they omit this metadata. Type collection is shared across new residues, patch atoms and explicit replacements; a separately failing patch test previously raised `KeyError` despite its supplied element mapping.

The final H200 suite passes 273 CPU/CUDA cases / 11 skips, and the current AtomWorks parser path passes all 19 fixtures through preparation, construction, scoring, gradients and rotamers. These checks do not supply force-field rows for arbitrary new atom types. A metadata-only benchmark over 1,000 sources improves known-type collection from 10.951 to 0.538 ms and a repeated custom type from 12.095 to 0.568 ms; one-source cost and traced allocation peak are essentially unchanged. This is type collection, not a whole-preparation or GPU speedup. See [results/element-metadata-validation.json](results/element-metadata-validation.json).


## 89. Reject missing parameters for used atom types before scoring — P1

[`tmol/ligand/_registry.py:212`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/ligand/_registry.py#L212), with the inherited table reindexing at [`tmol/score/ljlk/_params.py:124`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/ljlk/_params.py#L124). The scoring fallback predates this PR; accepting additional chemical types makes its contract relevant to the new preparation paths.

> Can scoring distinguish an unused database entry from a used atom type with no force-field row? Registering an element does not supply LJ/solvation parameters. Reindexing the scoring table inserts NaNs for missing rows, and setup currently accepts them. Removing the `CH3` row produces NaN energies and gradients for a two-alanine pose in both LJ/solvation and LK-ball. An actionable error should identify the residue, atom, type and missing fields. Real atoms also should not carry the unknown-type index intended for padded low-level lookups.

Fourteen regressions initially fail on parent `9ceaeeb1f`: unknown chemical types, missing/non-finite LJLK rows, float32 overflow and reuse of previously valid packed annotations. The follow-up shares real-atom index validation, checks all five LJLK fields in their kernel representation on the host, and validates used types before scoring. Unused incomplete types, finite zero-valued virtual types and the low-level NaN sentinel remain supported. The expanded tests cover each float field and an empty parameter table. Reused LJLK annotations include the scoring database identity, so a changed table cannot bypass validation.

Packed setup also reuses the validated block indices and heavy-atom lists instead of resolving them again. On CPU, warm packed annotation for the 230-type default catalog changes from 7.55 to 2.79 ms, with traced Python peak allocation changing from 382,047 to 312,444 bytes. Resolver construction increases from 2.228 to 2.326 ms because it now checks coverage; its traced allocation peak is essentially unchanged. These are setup measurements, not scoring or full-workflow speedups. All default LJLK tensor fields match the parent exactly. See [results/parameter-coverage-validation.json](results/parameter-coverage-validation.json) for final CPU/CUDA and workflow results. Coverage and finiteness do not establish parameter provenance, valid physical ranges or scientific accuracy.


## 90. Require explicit charges instead of silently zeroing an unknown residue — P1

[`tmol/score/elec/_params.py:53`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/elec/_params.py#L53). This inherited fallback is relevant to the PR's new generated and reusable charge bundles.

> Why does a missing atom charge raise while an entirely missing residue charge table returns zero? Removing every alanine charge record changes the native electrostatic score of a two-alanine pose from −0.6708 to zero and removes every coordinate derivative without reporting incomplete parameters. A real residue should require an applicable finite charge for each atom; an intentionally uncharged residue should carry explicit zeros. Patch lookup precedence should remain unchanged.

Six behavioral regressions fail on parent `b1c4834d8`, covering a missing residue table, non-finite/float32-overflow charges and reused packed annotations. A seventh initially failing test requires explicit water records. HOH is the only default residue whose charges previously came from the missing-table fallback. The follow-up stores its same three zeros explicitly; this preserves existing behavior and does not introduce a fitted water model. All charges across the 230 default types match the preceding implementation exactly.

The corrected lookup raises the existing missing-atom `KeyError` for absent residue tables and gives a residue/atom error for non-finite used charges. Unused invalid rows and explicitly zeroed residues remain supported. Native complete-table CPU energies and every gradient component are exact. Charge lookup adds about 2.6 microseconds for alanine and 0.62 ms for all 230 types in the focused benchmark; the check runs during annotation, with no added forward/backward kernel work. See [results/charge-coverage-validation.json](results/charge-coverage-validation.json) for the final CPU/CUDA checks. This establishes required charge coverage, not the scientific validity or neutrality of a charge model.


## 91. Build complete hydrogen-bond pair tables in bulk — P1 correctness / P2 performance

Inherited code at [`tmol/score/hbond/_params.py:118`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/hbond/_params.py#L118) and [`tmol/score/hbond/_params.py:163`](https://github.com/uw-ipd/tmol/blob/0593a93b07d80b0302383163d2d98c78e315ab98/tmol/score/hbond/_params.py#L163). The chemistry expansion makes repeated construction for distinct chemical databases more relevant.

> Can construction enforce the complete declared-family table required by the existing coverage test, then gather its polynomial fields in bulk? Omitting the backbone donor/acceptor pair currently produces NaN native ubiquitin energies and gradients. The loop also performs repeated one-row Pandas lookups and many small TensorGroup slice assignments. Host assembly can validate the family references, selected polynomial values and float32 weights before copying each final tensor. Later-definition precedence must remain explicit when pair or polynomial records repeat.

The follow-up requires a pair for every declared donor/acceptor family combination. It does not infer missing hydrogen-bond mappings from chemical donor flags: those flags and the mapper have distinct existing contracts. Unknown references and non-finite selected parameters raise actionable errors. Empty donor or acceptor families may omit their polynomial library; their native score and gradients are checked separately. Last pair/polynomial definitions retain precedence.

Fifteen initial failures comprise seven silent-acceptance cases, five existing errors made actionable, and three newly supported empty-family cases. Default and reversed catalogs have exactly matching tensor values, strides and logical storage across all 15 resolved/compacted tensor fields. Added whole-pose, weighted block-pair and ragged rotamer tests check that family reordering and reuse preserve scores and derivatives.

The final CPU benchmark reduces cold resolver construction from 64.37 to 0.796 ms and cold compact construction, including its resolver, from 62.79 to 0.990 ms. Traced Python peaks decrease by about 1.8 KB; logical tensor storage is unchanged at 129,048 bytes for the resolved plus compact tables. These are cold table-construction measurements in a warm process, excluding cache hits, fitting and scoring. See [results/hbond-table-validation.json](results/hbond-table-validation.json) for final CPU/CUDA evidence and the native missing-pair reproduction. These checks establish table integrity, not scientific validity of the fitted hydrogen-bond model.
