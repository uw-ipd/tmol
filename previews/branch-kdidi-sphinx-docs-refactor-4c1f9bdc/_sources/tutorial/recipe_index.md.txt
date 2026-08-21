# Find a TMol task

Use this page to find the tutorial or API reference for a common task. Start
with the numbered tutorials when learning a workflow; use the API pages when
you already know which object or function you need.

## Fundamentals, input, and output

| Task | Maintained location | Status and notes |
| --- | --- | --- |
| Default `ParameterDatabase` | [01 — Working with TMol](01_working_with_tmol.ipynb), [database API](../api/database.rst) | Supported. |
| Default `PackedBlockTypes` and `CanonicalOrdering` | [01 — Working with TMol](01_working_with_tmol.ipynb), [I/O API](../api/io.rst) | Supported. Most users receive these through a build context rather than constructing them manually. |
| Pose from PDB | [01 — Working with TMol](01_working_with_tmol.ipynb), [`tmol.io`](../api/io.rst) | Supported by the direct `tmol.pose_stack_from_pdb()` path. Prefer CIF/Biotite when richer metadata or ligand bonds matter. |
| Residue-range PDB input | [01 — Working with TMol](01_working_with_tmol.ipynb) | Supported. `residue_start`/`residue_end` are zero-based, half-open positions in parsed residue order, not author residue numbers. `res_not_connected` distinguishes an internal cut from a true terminus. |
| Pose from Biotite or mmCIF | [01 — Working with TMol](01_working_with_tmol.ipynb), [I/O API](../api/io.rst) | Supported and preferred for general structure input. |
| Pose from OpenFold tensors and differentiable missing-atom construction | [01 — Working with TMol](01_working_with_tmol.ipynb), [`pose_stack_from_openfold`](../api/io.rst) | Supported for canonical proteins. The API consumes `aatype`, final `positions`, and `chain_index`; gradients can flow from built atoms to supplied coordinates. No separate tutorial 09 is included because the repository has no checked-in, non-binary OpenFold prediction fixture suitable for a fully executable end-to-end notebook. |
| Input with missing residues or chain gaps | [01 — Working with TMol](01_working_with_tmol.ipynb), [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported. Preserve disconnection semantics; do not silently turn an internal gap into a chemical terminus. |
| Concatenate heterogeneous poses or copy one pose into a batch | [02 — GPU batching](02_gpu_batching.ipynb) | Supported with `PoseStackBuilder.from_poses()`. |
| Write one pose, a multi-model PDB, or separate per-pose PDBs | [01 — Working with TMol](01_working_with_tmol.ipynb), [I/O API](../api/io.rst) | Supported. PDB is a compatibility format and is not a lossless replacement for CIF plus authoritative ligand chemistry. |

## Kinematics and minimization

| Task | Maintained location | Status and notes |
| --- | --- | --- |
| Automatic N→C, multi-chain, and gap-aware forest | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported by `FoldForest.reasonable_fold_forest()`. It follows polymer connectivity and intentionally ignores non-polymer connections such as disulfides. |
| Explicit `FoldForest.from_edges()` | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported. Current edges have four fields: `(edge_type, start_block, end_block, jump_index)`. The PR 399 three-field examples are stale. |
| Per-residue-root (“dandelion”) forest | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported low-level construction for NN-like per-residue frames. Validate one root edge per real block and sentinel padding. |
| Enable all or selected named torsions | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported with `MoveMap`. |
| Cartesian and kinematic minimization | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb), [06 — FastRelax](06_fast_relax.ipynb) | Supported through `tmol.run_cart_min()` and `tmol.run_kin_min()`. These optimize different coordinate models. Compare them only with masks, weights, iteration budgets, and stopping checks chosen for your system—not the tutorials' 10-iteration smoke tests. |
| Cartesian and kinematic FastRelax | [06 — FastRelax](06_fast_relax.ipynb) | `fast_relax()` uses Cartesian minimization by default and accepts a compatible kinematic minimizer. It is a Rosetta-inspired tensor-native subset; schedules, trajectories, acceptance, and scores are not interchangeable with Rosetta FastRelax. |
| Batched relax | [06 — FastRelax](06_fast_relax.ipynb) | Supported; the larger demonstration is GPU-only while the small local smoke path runs on CPU. |

## Scoring and constraints

| Task | Maintained location | Status and notes |
| --- | --- | --- |
| Default, empty, and focused score functions | [03 — Scoring and analysis](03_scoring_and_analysis.ipynb) | Supported. |
| Activate and deactivate score terms | [03 — Scoring and analysis](03_scoring_and_analysis.ipynb) | Supported. A zero weight removes an energy-term implementation only when none of that implementation's sibling score types remain active. |
| Whole-pose scoring and coordinate backpropagation | [03 — Scoring and analysis](03_scoring_and_analysis.ipynb) | Supported. |
| Weighted and unweighted block-pair tensors | [03 — Scoring and analysis](03_scoring_and_analysis.ipynb) | Supported. The matrix uses directed accounting; combine both orientations for an unordered off-diagonal pair. |
| Differentiable interface/block-pair reweighting | [03 — Scoring and analysis](03_scoring_and_analysis.ipynb) | Supported. Apply an explicit tensor of analytical weights to the block-pair result before summing and backpropagating. |
| Low-level harmonic distance constraint | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported with `ConstraintSet` and `ConstraintEnergyTerm.harmonic`. |
| Propagate one constraint definition across a pose batch | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported with `add_constraints_to_all_poses()`. |
| Cα and declared-main-chain coordinate restraints | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported. `constrain_all_ca()` is protein/Cα-specific; `create_mainchain_coordinate_constraints()` follows each block type's main-chain declaration. |

## Packing, design, and preparation

| Task | Maintained location | Status and notes |
| --- | --- | --- |
| Construct and reuse a Dunbrack sampler | [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported. |
| Fixed-sequence repacking | [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported. `IncludeCurrentSampler` deliberately retains the input conformation as a candidate. |
| Optimize hydroxyl/proton chis | [01 — Working with TMol](01_working_with_tmol.ipynb), [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported through normal preparation (`no_optH=False`) or an explicitly configured `OptHSampler`. PR 399 constructed `optH_task` but accidentally passed `task`; that cell did not test the intended operation. |
| Build supported missing side chains during input | [01 — Working with TMol](01_working_with_tmol.ipynb) | Supported by structure conversion when sufficient chemistry and anchors are present. |
| Extra χ sampling | [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported. TMol χ indices are zero-based (`0` for χ1, `1` for χ2); the PR 399 `1`/`2` calls were off by one. |
| Regional design and small mutation scans | [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported through explicit low-level task masks. TMol has no built-in Rosetta resfile, selector, or mutation-scan protocol layer. |
| Ligand parameter preparation and injection | [07 — Ligands and parameter files](07_ligand_and_params.ipynb), [ligand guide](../user_guide/ligands.md) | Supported from authoritative CIF/MOL2 chemistry. The Rosetta `.params` writer is syntactic and experimental, not a validated Rosetta parameterization workflow. |
| DNA/RNA chi packing | [08 — DNA and RNA](08_nucleic_acids.ipynb) | Supported for the documented local primitives. Sugar-pucker sampling and full RosettaDNA/RNA protocols are not implemented. |

## Availability labels

- **Library API** means TMol provides a reusable public function or class.
- **Example recipe** means the documentation composes lower-level APIs for the
  task; it is not a built-in protocol.
- **Not available** means the Rosetta workflow has no supported TMol
  implementation.

Readers translating Rosetta workflows should also keep the
[Rosetta-to-TMol crosswalk](rosetta_crosswalk.md) open alongside this index.
