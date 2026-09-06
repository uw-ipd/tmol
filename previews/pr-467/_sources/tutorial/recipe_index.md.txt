# Find a TMol task

Use this page to find a concise workflow, deep Tutorial, or public API for a
common task. Browse the {doc}`interactive examples </examples_index>` when you
want a complete walkthrough rather than a lookup table.

## Fundamentals, input, and output

| Task | Workflow or Tutorial | API and notes |
| --- | --- | --- |
| Build a default `ParameterDatabase`, `PackedBlockTypes`, or `CanonicalOrdering` | {doc}`Structure I/O workflow </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | {doc}`Database API </api/database>`; {doc}`I/O API </api/io>`. Most callers receive packed types and ordering through a build context. |
| Build a pose from PDB | {doc}`Structure I/O workflow </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | `pose_stack_from_pdb()` is a compatibility path. Prefer CIF/Biotite when metadata or ligand bonds matter. |
| Select a residue range from PDB | {doc}`Tutorial 01 <01_working_with_tmol>` | `residue_start`/`residue_end` are zero-based, half-open parsed positions, not author residue numbers. |
| Build a pose from Biotite or mmCIF | {doc}`Structure I/O workflow </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | Preferred general input path; see the {doc}`I/O API </api/io>`. |
| Build from OpenFold, RosettaFold2, or AtomWorks tensors | {doc}`Integrations </user_guide/integrations>`; {doc}`Tutorial 01 <01_working_with_tmol>` | Supported adapters have distinct tensor contracts; see the {doc}`I/O API </api/io>`. |
| Preserve chain gaps and disconnected regions | {doc}`Tutorial 01 <01_working_with_tmol>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | Keep internal gaps disconnected rather than silently turning them into chemical termini. |
| Batch heterogeneous poses | {doc}`GPU batching workflow </workflows/gpu_batching>`; {doc}`Tutorial 02 <02_gpu_batching>` | Use `PoseStackBuilder.from_poses()` for compatible chemistry. |
| Export Biotite, one PDB, or multiple models | {doc}`Structure I/O workflow </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | PDB is not a lossless replacement for CIF plus authoritative ligand chemistry. |

## Kinematics and minimization

| Task | Workflow or Tutorial | API and notes |
| --- | --- | --- |
| Build an automatic multi-chain, gap-aware forest | {doc}`Optimization workflow </user_guide/optimization>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | `FoldForest.reasonable_fold_forest()` follows polymer connectivity and ignores non-polymer connections such as disulfides. |
| Construct explicit or per-residue-root forests | {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | `FoldForest.from_edges()` uses `(edge_type, start_block, end_block, jump_index)`. Validate root coverage and sentinel padding. |
| Select named torsions and jumps | {doc}`Optimization workflow </user_guide/optimization>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | Configure a `MoveMap`; see the {doc}`kinematics API </api/kinematics>`. |
| Run Cartesian or kinematic minimization | {doc}`Optimization workflow </user_guide/optimization>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | The coordinate models differ. Compare only with matched masks, weights, budgets, and stopping checks. |
| Run Cartesian, kinematic, or batched FastRelax | {doc}`Optimization workflow </user_guide/optimization>`; {doc}`Tutorial 06 <06_fast_relax>` | `fast_relax()` defaults to Cartesian minimization and accepts a compatible kinematic minimizer. It is a smaller Rosetta-inspired routine, not protocol parity. |

## Scoring and constraints

| Task | Workflow or Tutorial | API and notes |
| --- | --- | --- |
| Build default, empty, or focused score functions | {doc}`Scoring workflow </user_guide/scoring>`; {doc}`Tutorial 03 <03_scoring_and_analysis>` | See the {doc}`score API </api/score>` and {doc}`term map </api/score_terms>`. |
| Score a pose or backpropagate through coordinates | {doc}`Scoring workflow </user_guide/scoring>`; {doc}`Tutorial 03 <03_scoring_and_analysis>` | Render a module for the current pose layout and call it with coordinates. |
| Analyze weighted or unweighted block pairs | {doc}`Scoring workflow </user_guide/scoring>`; {doc}`Tutorial 03 <03_scoring_and_analysis>` | Directed accounting can require both matrix orientations for an unordered pair. |
| Map a protein interface and test selected alanine substitutions | {doc}`Protein-interface workflow </workflows/protein_interfaces>`; {doc}`Case Study 09 <09_protein_interface_hotspot_scan>` | Compose author-label masks, both block-pair orientations, and matched local-repacking tasks. Report one-complex score changes, not thermodynamic ΔΔG. |
| Reweight an interface differentiably | {doc}`Tutorial 03 <03_scoring_and_analysis>` | Apply an explicit analytical weight tensor before summing and backpropagating. |
| Add distance, coordinate, or torsion constraints | {doc}`Optimization workflow </user_guide/optimization>`; {doc}`Tutorial 05 <05_minimization_constraints_kinematics>` | See the {doc}`constraint API </api/score_terms>`. `constrain_all_ca()` is protein-specific; main-chain restraints follow block declarations. |

## Packing, design, and preparation

| Task | Workflow or Tutorial | API and notes |
| --- | --- | --- |
| Construct samplers and repack a fixed sequence | {doc}`Packing workflow </workflows/packing>`; {doc}`Tutorial 04 <04_packing_and_mutation_scan>` | `IncludeCurrentSampler` deliberately keeps the input conformation as a candidate. |
| Optimize polar-hydrogen chis or build supported side chains | {doc}`Structure I/O workflow </workflows/structure_io>`; {doc}`Tutorial 01 <01_working_with_tmol>` | Use normal preparation or explicitly configure the relevant sampler. |
| Add extra χ sampling | {doc}`Packing workflow </workflows/packing>`; {doc}`Tutorial 04 <04_packing_and_mutation_scan>` | TMol χ indices are zero-based: `0` is χ1 and `1` is χ2. |
| Run regional design or a small mutation-score experiment | {doc}`Packing workflow </workflows/packing>`; {doc}`Tutorial 04 <04_packing_and_mutation_scan>` | Compose explicit task masks. TMol has no built-in Rosetta resfile, selector, or mutation-scan protocol layer. |
| Prepare and inject ligand parameters | {doc}`Ligand workflow </user_guide/ligands>`; {doc}`Tutorial 07 <07_ligand_and_params>` | Start from authoritative CIF/MOL2 chemistry. The Rosetta `.params` writer is syntactic and experimental. |
| Score controlled ligand-pose decoys and locally refine diagnostic states | {doc}`Case Study 10 <10_ligand_pose_sensitivity>` | Reuse one ligand-aware context, batch matched rigid-body decoys, and report pose sensitivity rather than docking or binding affinity. |
| Score or pack DNA/RNA | {doc}`Nucleic-acid workflow </workflows/nucleic_acids>`; {doc}`Tutorial 08 <08_nucleic_acids>` | Sugar-pucker sampling and full RosettaDNA/RNA protocols are not implemented. |

## Availability labels

- **Library API** means TMol provides a reusable public function or class.
- **Workflow recipe** means the documentation composes lower-level APIs for the
  task; it is not a built-in protocol.
- **Not available** means the Rosetta workflow has no supported TMol
  implementation.

Readers translating Rosetta workflows should also keep the
{doc}`Rosetta-to-TMol crosswalk <rosetta_crosswalk>` open alongside this index.
