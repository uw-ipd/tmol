# Task index

Find a guide, runnable notebook, or API for a specific operation.

## Fundamentals, input, and output

| Task | Guide or tutorial | API and notes |
| --- | --- | --- |
| Build a default `ParameterDatabase`, `PackedBlockTypes`, or `CanonicalOrdering` | {doc}`Structure I/O guide </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | {doc}`Database API </api/database>`; {doc}`I/O API </api/io>`. Most callers receive packed types and ordering through a build context. |
| Build a pose from PDB | {doc}`Structure I/O guide </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | `pose_stack_from_pdb()` is a compatibility path. Prefer CIF/Biotite when metadata or ligand bonds matter. |
| Select a residue range from PDB | {doc}`Tutorial 01 <01_working_with_tmol>` | `residue_start`/`residue_end` are zero-based, half-open parsed positions, not author residue numbers. |
| Build a pose from Biotite or mmCIF | {doc}`Structure I/O guide </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | Preferred general input path; see the {doc}`I/O API </api/io>`. |
| Build from OpenFold, RoseTTAFold2, or AtomWorks tensors | {doc}`Integrations </user_guide/integrations>`; {doc}`Tutorial 01 <01_working_with_tmol>` | Use Atom37 or map another named layout to canonical tensors; see the {doc}`model input tutorial </model_inputs>`. |
| Preserve chain gaps and disconnected regions | {doc}`Tutorial 01 <01_working_with_tmol>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | Keep internal gaps disconnected rather than silently turning them into chemical termini. |
| Batch heterogeneous poses | {doc}`GPU batching guide </workflows/gpu_batching>`; {doc}`Tutorial 02 <02_gpu_batching>` | Use `PoseStackBuilder.from_poses()` for compatible chemistry. |
| Export Biotite, one PDB, or multiple models | {doc}`Structure I/O guide </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | PDB is not a lossless replacement for CIF plus authoritative ligand chemistry. |
| Build a stable chemistry subset or inspect packed block-type metadata | {doc}`Tutorial 11 <11_extending_chemistry_and_scoring>` | Keep the source `ParameterDatabase` immutable; a subset is only safe when the complete input alphabet is known. |

## Kinematics and minimization

| Task | Guide or tutorial | API and notes |
| --- | --- | --- |
| Build an automatic multi-chain, gap-aware forest | {doc}`Optimization guide </user_guide/optimization>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | `FoldForest.reasonable_fold_forest()` follows polymer connectivity and ignores non-polymer connections such as disulfides. |
| Construct explicit or per-residue-root forests | {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | `FoldForest.from_edges()` uses `(edge_type, start_block, end_block, jump_index)`. Validate root coverage and sentinel padding. |
| Select named torsions and jumps | {doc}`Optimization guide </user_guide/optimization>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | Configure a `MoveMap`; see the {doc}`kinematics API </api/kinematics>`. |
| Run Cartesian or kinematic minimization | {doc}`Optimization guide </user_guide/optimization>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | The coordinate models differ. Compare only with matched masks, weights, budgets, and stopping checks. |
| Run Cartesian, kinematic, or batched FastRelax | {doc}`Optimization guide </user_guide/optimization>`; {doc}`Tutorial 06 <06_fast_relax>` | `fast_relax()` defaults to Cartesian minimization and accepts a compatible kinematic minimizer. |
| Couple chains with an explicit jump or construct a dandelion forest | {doc}`Tutorial 12 <12_explicit_foldforests_and_torsions>` | Validate edge coverage and contiguous ordinary-jump indices before coordinate operations. |
| Perturb rigid-body jump DOFs or assign named torsions | {doc}`Tutorial 12 <12_explicit_foldforests_and_torsions>` | These are coordinate transformations, not docking, minimization, or idealization protocols. |

## Scoring and constraints

| Task | Guide or tutorial | API and notes |
| --- | --- | --- |
| Build default, empty, or focused score functions | {doc}`Scoring guide </user_guide/scoring>`; {doc}`Tutorial 03 <03_scoring_and_analysis>` | See the {doc}`score API </api/score>` and {doc}`term map </api/score_terms>`. |
| Score a pose or backpropagate through coordinates | {doc}`Scoring guide </user_guide/scoring>`; {doc}`Tutorial 03 <03_scoring_and_analysis>` | Render a module for the current pose layout and call it with coordinates. |
| Analyze weighted or unweighted block pairs | {doc}`Scoring guide </user_guide/scoring>`; {doc}`Tutorial 03 <03_scoring_and_analysis>` | Directed accounting can require both matrix orientations for an unordered pair. |
| Map a protein interface and test selected alanine substitutions | {doc}`Protein-interface guide </workflows/protein_interfaces>`; {doc}`Tutorial 09 <09_protein_interface_hotspot_scan>` | Compose author-label masks, both block-pair orientations, and matched local-repacking tasks. |
| Reweight an interface differentiably | {doc}`Tutorial 03 <03_scoring_and_analysis>` | Apply an explicit analytical weight tensor before summing and backpropagating. |
| Add distance, coordinate, or torsion constraints | {doc}`Optimization guide </user_guide/optimization>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | See the {doc}`constraint API </api/score_terms>`. `constrain_all_ca()` is protein-specific; main-chain restraints follow block declarations. |
| Construct a focused score function or derive modified score parameters | {doc}`Tutorial 11 <11_extending_chemistry_and_scoring>` | Change weights to alter term contributions; derive a new immutable database only when changing the underlying parameter model. |

## Packing, design, and preparation

| Task | Guide or tutorial | API and notes |
| --- | --- | --- |
| Construct samplers and repack a fixed sequence | {doc}`Packing guide </workflows/packing>`; {doc}`Tutorial 04 <04_packing_and_mutation_scan>` | `IncludeCurrentSampler` deliberately keeps the input conformation as a candidate. |
| Optimize polar-hydrogen chis or build supported side chains | {doc}`Structure I/O guide </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | Use normal preparation or explicitly configure the relevant sampler. |
| Add extra χ sampling | {doc}`Packing guide </workflows/packing>`; {doc}`Tutorial 04 <04_packing_and_mutation_scan>` | TMol χ indices are zero-based: `0` is χ1 and `1` is χ2. |
| Run regional design or a small mutation-score experiment | {doc}`Packing guide </workflows/packing>`; {doc}`Tutorial 04 <04_packing_and_mutation_scan>` | Compose explicit task masks. |
| Subclass `PackerPalette`, audit rotamer candidates, or export a rotamer ensemble | {doc}`Tutorial 13 <13_extending_the_packer>` | Candidate enumeration is deterministic for a fixed task; annealing is a separate stochastic assignment search. |
| Prepare and inject ligand parameters | {doc}`Ligand guide </user_guide/ligands>`; {doc}`Tutorial 07 <07_ligand_and_params>` | Start from authoritative CIF/MOL2 chemistry. `.tmol` is the only parameter format. |
| Score controlled ligand-pose decoys and locally refine diagnostic states | {doc}`Tutorial 10 <10_ligand_pose_sensitivity>` | Reuse one ligand-aware context, batch matched rigid-body decoys, and compare scores with ligand displacement. |
| Score or pack DNA/RNA | {doc}`Nucleic-acid guide </workflows/nucleic_acids>`; {doc}`Tutorial 08 <08_nucleic_acids>` | The packer samples glycosidic and hydroxyl-proton chi; sugar pucker stays fixed. |
