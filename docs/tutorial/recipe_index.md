# TMol recipe index

This index audits the implemented examples in the historical
[PR #399](https://github.com/uw-ipd/tmol/pull/399) notebook
(`tmol_how_to_guide.ipynb`, revision `b45ba1c48`) against the maintained
tutorials and API. It is a migration map, not an endorsement of every old
cell. The numbered tutorials are the maintained learning path; API pages are
the concise reference.

## Fundamentals, input, and output

| PR 399 recipe | Maintained location | Status and notes |
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

| PR 399 recipe | Maintained location | Status and notes |
| --- | --- | --- |
| Automatic N→C, multi-chain, and gap-aware forest | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported by `FoldForest.reasonable_fold_forest()`. It follows polymer connectivity and intentionally ignores non-polymer connections such as disulfides. |
| Explicit `FoldForest.from_edges()` | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported. Current edges have four fields: `(edge_type, start_block, end_block, jump_index)`. The PR 399 three-field examples are stale. |
| Per-residue-root (“dandelion”) forest | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported low-level construction for NN-like per-residue frames. Validate one root edge per real block and sentinel padding. |
| Enable all or selected named torsions | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb) | Supported with `MoveMap`. |
| Cartesian and kinematic minimization | [05 — Minimization, constraints, and kinematics](05_minimization_constraints_kinematics.ipynb), [06 — FastRelax](06_fast_relax.ipynb) | Supported through the top-level `tmol.run_cart_min()` and `tmol.run_kin_min()` convenience APIs. Cartesian coordinates and kinematic DOFs are different models; compare them only with explicit masks, weights, and convergence criteria. |
| Cartesian and kinematic FastRelax | [06 — FastRelax](06_fast_relax.ipynb) | Supported as a tensor-native subset, not exact Rosetta protocol parity. |
| Batched relax | [06 — FastRelax](06_fast_relax.ipynb) | Supported; the larger demonstration is GPU-only while the small local smoke path runs on CPU. |

## Scoring and constraints

| PR 399 recipe | Maintained location | Status and notes |
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

| PR 399 recipe | Maintained location | Status and notes |
| --- | --- | --- |
| Construct and reuse a Dunbrack sampler | [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported. |
| Fixed-sequence repacking | [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported. `IncludeCurrentSampler` deliberately retains the input conformation as a candidate. |
| Optimize hydroxyl/proton chis | [01 — Working with TMol](01_working_with_tmol.ipynb), [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported through normal preparation (`no_optH=False`) or an explicitly configured `OptHSampler`. PR 399 constructed `optH_task` but accidentally passed `task`; that cell did not test the intended operation. |
| Build supported missing side chains during input | [01 — Working with TMol](01_working_with_tmol.ipynb) | Supported by structure conversion when sufficient chemistry and anchors are present. |
| Extra χ sampling | [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported. TMol χ indices are zero-based (`0` for χ1, `1` for χ2); the PR 399 `1`/`2` calls were off by one. |
| Regional design and small mutation scans | [04 — Packing and mutation scan](04_packing_and_mutation_scan.ipynb) | Supported through explicit low-level task masks. TMol has no built-in Rosetta resfile, selector, or mutation-scan protocol layer. |
| Ligand parameter preparation and injection | [07 — Ligands and parameter files](07_ligand_and_params.ipynb), [ligand guide](../user_guide/ligands.md) | Supported from authoritative CIF/MOL2 chemistry. The Rosetta `.params` writer is syntactic and experimental, not a validated Rosetta parameterization workflow. |
| DNA/RNA chi packing | [08 — DNA and RNA](08_nucleic_acids.ipynb) | Supported for the documented local primitives. Sugar-pucker sampling and full RosettaDNA/RNA protocols are not implemented. |

## Rejected, obsolete, defective, or unsupported PR 399 items

- **Obsolete setup:** the `0.1.36` wheel and `TMOL_USE_JIT=1` setup no longer
  describe the supported installation/runtime path. The tutorials use the
  current wheel compatibility checks and repository fixtures.
- **Defective OptH example:** it configured `optH_task` but passed the unrelated
  `task` to `pack_rotamers()`.
- **Defective extra-chi example:** it treated χ indices as one-based; TMol uses
  zero-based indices.
- **Defective PR Cartesian wrapper:** the historical implementation selected
  the kinematic minimizer as its fallback. The current
  `tmol.cartesian_fast_relax()` wrapper uses the Cartesian minimizer and is
  demonstrated in tutorial 06.
- **Defective toy restraint system:** constraining every Cα pair creates
  \(O(N^2)\) restraints and obscures the scientific restraint being imposed.
  Tutorial 05 instead demonstrates one interpretable distance restraint,
  batch propagation, and coordinate-restraint utilities.
- **Defective custom-cartbonded assertion:** arbitrarily doubling one PRO
  spring constant does not guarantee a larger total score for an arbitrary
  structure. Custom parameter databases are an advanced parameterization task,
  not a monotonic-score recipe.
- **TODO-only stubs:** alternate block-type loading, custom databases and
  canonical orderings, database subsetting, custom residue/scoring parameter
  creation, sequence-to-pose construction, mmCIF output, rotamer-set output,
  rigid-body perturbation, direct dihedral assignment, custom score databases,
  soft-rep score creation, custom `PackerPalette`, and separate output stubs
  contained no implemented recipe to preserve.
- **Currently unsupported stubs:** idealization from PDB, idealization from a
  dandelion forest, backbone-only idealization, and an end-to-end
  sequence/output protocol are not presented as supported APIs.
- **OpenFold tutorial boundary:** tensor input itself is supported and indexed
  above, but there is no checked-in prediction fixture that supports a
  standalone, fully executable prediction-to-refinement notebook without
  adding a fragile binary or network download. The maintained coverage stays
  in tutorial 01, tutorial 05, and the I/O API.

Readers translating Rosetta workflows should also keep the
[Rosetta-to-TMol crosswalk](rosetta_crosswalk.md) open alongside this index.
