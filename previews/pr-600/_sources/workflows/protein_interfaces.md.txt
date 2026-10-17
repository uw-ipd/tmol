# Protein-interface analysis

Calculate interface contributions and compare local mutations. Use a
prepared multi-chain `PoseStack` and a score function from the same parameter database.

## Define partners from metadata

Build partner masks from the author chain labels in `pdb_info`:

```python
chain_labels = pose_stack.pdb_info.chain_labels
partner_a = torch.as_tensor(chain_labels == "A", device=pose_stack.device)
partner_b = torch.as_tensor(chain_labels == "B", device=pose_stack.device)
```

Masks have shape `[n_poses, max_n_blocks]`. Check the selected chain labels and
block counts; file residue numbers need not match block indices.

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

The result sums weighted interactions between the two masks. For residue-pair
analysis, render
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
device, number of outcomes, minimization settings, and comparison rule. Compare
each mutant with its independently repacked WT control.

## Examples and reference

{doc}`Interface example </tutorial/09_protein_interface_hotspot_scan>` · {doc}`Analysis API </api/analysis>` · {doc}`Packing </workflows/packing>`
