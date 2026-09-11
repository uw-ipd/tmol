# Sampling limits on the review branch

`task.set_chi_sample_budget(expanded_limit, limit)` accepts two positive integer
counts. It stores an immutable pair, copied into the concrete task; setting it
before or after adding a sampler has the same effect. Without an explicit
override, each sampler retains its defaults (Dunbrack remains uncapped).

For generic chi products, keep expansions if the expanded product fits
`expanded_limit`; otherwise drop expansions and freeze heavy chi from the tip
inward until the product fits `limit`. Proton chi retain at least the first mean
so their hydrogen is placed. Library multiplicity is an actual row count,
not an estimate inferred from chi count. A required library larger than the
available budget raises a diagnostic instead of silently discarding its states.

For a conjugated group, limits count the conformer rows across **all members**,
including the offered input conformation. Thus eight members and 66 anchor rows
need at least 536 member rotamers with `include_current=True`, when all library
axes remain movable and those rows are distinct. Cyclic/external constraints
project the library onto the remaining movable axes, retaining the first of
each identical projected row. All distinct projected states remain required;
child chi may freeze. An empty set of axes produces one input conformer, not
two identical empty rows. The count is checked before allocating the Cartesian
product. Subsequent task masks can freeze individual members: the total is
`active_members * conformers + fixed_members`, with one input/background
rotamer for each fixed member. Their atoms constrain the permitted axes before
enumeration, so the sampled members cannot pull away from a frozen neighbor.
Disabling the group sampler for every member skips enumeration entirely.

Fully constrained anchors skip the source-library pass. A partially constrained
anchor can need more temporary source-library rows than its final projected
budget. That private pass does not apply the final group limit to its raw rows;
the group enforces it after projection. Thus this is not a bound on temporary
library workspace. Bounding/adapting that workspace remains part of the broader
library-expansion work.

NA and OptH use private sampler/task views for explicit task settings and
configuration-aware RT/PBT caches. Their final count check covers runtime
library/proton products before sample-array allocation. Dunbrack checks its
native per-restype library/extra-chi count against the larger explicit limit
before allocating rotamer mappings and chi tensors. Required oversized products
currently raise; Dunbrack extra-chi settings and NA glycosidic/proton products
are not yet adaptively reduced together.

With an explicit task budget, the combined count at each physical residue must
also fit `max(expanded_limit, limit)`, summing all allowed residue types and all
samplers, including IncludeCurrent/Fallback. This shared check runs before
merging source rows or allocating conformer coordinates. Oversized required
unions raise with the pose/block/count; states are not silently discarded.
Separate poses and residue positions have separate limits. Sampler defaults
continue to apply when no explicit task budget is set.

This aggregate check occurs after samplers have created their private source
rows, so it is not a bound on temporary sampling workspace. It is also not a
bound on the task's total across positions or quadratic pair-energy memory.
Default policy, adaptive library expansion and extreme native integer-overflow
limits remain tracked in `FOLLOWUP.md`. The merge itself counts in int64 and
rejects totals exceeding native int32 indexing. Tests cover sequential reuse;
concurrent mutation of shared RT/PBT annotations is not supported by these
cache checks.
