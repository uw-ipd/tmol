# Protein-interface analysis

Use this recipe to score a declared protein–protein interface and prepare an
explicit batch of local mutation experiments. The case study develops the full
workflow on neighboring KcsA subunits and visualizes the native hotspots and
packed variants.

> - **Prerequisites:** A prepared multi-chain `PoseStack` and a score function
>   built from the same parameter database.
> - **Deep example:** {doc}`09 — Map and Test a Protein Interface
>   </tutorial/09_protein_interface_hotspot_scan>`.
> - **Related workflows:** {doc}`Scoring </user_guide/scoring>`,
>   {doc}`Packing </workflows/packing>`, and
>   {doc}`GPU batching </workflows/gpu_batching>`.
> - **API reference:** {doc}`Scoring </api/score>`,
>   {doc}`Packing </api/pack>`, and {doc}`Analysis helpers </api/analysis>`.
> - **Rosetta mapping:** {doc}`Packing, design, and mutation scans
>   </tutorial/rosetta_crosswalk>`.

## Define partners from metadata

Do not assume chain boundaries from block positions. Build masks from the
author labels retained in `pdb_info`:

```python
chain_labels = pose_stack.pdb_info.chain_labels
partner_a = torch.as_tensor(chain_labels == "A", device=pose_stack.device)
partner_b = torch.as_tensor(chain_labels == "B", device=pose_stack.device)
```

Each mask has shape `[n_poses, max_n_blocks]`. Verify the selected chains and
block counts before scoring; author labels, insertion codes, missing regions,
and model-specific preprocessing determine the actual layout.

## Score the interface

The convenience reduction sums both stored orientations between the masks:

```python
from tmol.ops import calculate_block_pair_ddg

interface_by_term = calculate_block_pair_ddg(
    pose_stack,
    partner_a,
    partner_b,
    sfxn=score_function,
    sum_terms=False,
    minimize=False,
    pack=False,
)
```

Despite the helper's historical name, this is a weighted cross-mask score from
one complex. It is not a binding free energy. For residue-pair analysis, render
a block-pair scorer and add `matrix[i, j] + matrix[j, i]`; one orientation alone
can miss an interaction.

## Batch explicit mutations

Construct one pose per requested site/state, then restrict task choices
monotonically:

```python
scan_batch = PoseStackBuilder.from_poses([pose_stack] * n_requests, device)
task = PackerTask(scan_batch, PackerPalette())

mutation_mask = torch.zeros_like(scan_batch.block_type_ind, dtype=torch.bool)
mutation_mask[mutant_pose_indices, target_block_indices] = True

task.restrict_to_repacking(~mutation_mask)
task.restrict_absent_name3s({"ALA"}, mutation_mask)
task.disable_packing_by_block_mask(~local_shells)
```

Add the appropriate conformer samplers and call `pack_rotamers()` once on the
batch. Include an independently repacked WT control for each site, use the same
declared shell for its WT/mutant pair, verify all non-target identities, and
record stochastic outcomes rather than presenting a single pack as converged.

For one target block per pose, pass an integer index tensor to
`calculate_block_pair_ddg()` together with the partner mask. That indexed route
avoids a dense target-mask reduction.

## Interpret the result

Report the exact computational experiment: score function and weights, input
and preparation, partner masks, packing shell, allowed identities, samplers,
device, number of outcomes, minimization settings, and comparison rule. A
mutant-minus-repacked-WT value from one complex is a **score change**, not a
thermodynamic ΔΔG. Binding claims require explicitly modeled states and a
validated protocol beyond this lower-level recipe.
