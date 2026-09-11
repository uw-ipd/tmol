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
need at least 536 member rotamers with `include_current=True`. Child chi may
freeze; this does not drop the group's required library rows. The count is
checked before allocating the Cartesian product.

NA and OptH use private sampler/task views for explicit task settings and
configuration-aware RT/PBT caches. Their final count check covers runtime
library/proton products before sample-array allocation. Dunbrack checks its
native per-restype library/extra-chi count against the larger explicit limit
before allocating rotamer mappings and chi tensors. Required oversized products
currently raise; Dunbrack extra-chi settings and NA glycosidic/proton products
are not yet adaptively reduced together.

These are per-sampler/per-restype bounds (or a group's sum across members), not
a bound on a whole task's sum across residue types and samplers, including
IncludeCurrent/Fallback. They also do not directly bound quadratic pair-energy
memory. Default policy, aggregate budgeting and extreme integer-overflow limits
remain tracked in `FOLLOWUP.md`. Tests cover sequential reuse; concurrent
mutation of shared RT/PBT annotations is not supported by these cache checks.
