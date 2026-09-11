"""Bounding the rotamers a set of sampled chi enumerates.

The budget is applied when rotamers are generated, not when a residue type is
built: a block's chi may be enumerated on their own or jointly with those of the
blocks conjugated to it, and which of those it is depends on the pose. Sampled
chi are therefore stored complete and frozen here.

Freezing runs tip-first, ordered by each chi's depth in the kinforest that will
build it. For a single block that is the residue's own tree, so the ordering is
the familiar one -- a chi near the backbone swings the whole sidechain where one
at the tip moves a couple of atoms. For a group of conjugated blocks it is the
group's tree, so the outermost sugar's outermost chi freezes first. The two are
the same rule; only the tree differs.
"""

import attr
from copy import copy


def sampler_for_task(sampler, task):
    """Apply an explicit task budget to a private sampler/task view."""
    budget = getattr(task, "chi_sample_budget", None)
    if budget is None or budget == (
        sampler.chi_sample_expanded_limit,
        sampler.chi_sample_limit,
    ):
        return sampler, task
    configured = attr.evolve(
        sampler, chi_sample_expanded_limit=budget[0], chi_sample_limit=budget[1]
    )
    private_task = copy(task)
    private_task.conformer_sampler_index = dict(task.conformer_sampler_index)
    private_task.conformer_sampler_index[id(configured)] = task.conformer_sampler_index[
        id(sampler)
    ]
    return configured, private_task


def checked_sample_count(counts, expanded_limit, limit):
    """Check the bound and get the allocation size with one device-to-host copy."""
    import torch

    if counts.numel() == 0:
        return 0
    total, largest = torch.stack((counts.sum(), counts.max())).tolist()
    maximum = max(expanded_limit, limit)
    if largest > maximum:
        raise ValueError(
            f"Sampling budget {maximum} cannot fit the required library states"
        )
    return total


def n_conformers(samples, expanded: bool) -> int:
    """How many conformers a set of sampled chi enumerates."""
    total = 1
    for cs in samples:
        if cs is None:  # frozen
            continue
        total *= len(cs.samples) * (1 + 2 * len(cs.expansions) if expanded else 1)
    return total


def apply_chi_sample_budget(
    samples,
    depths,
    expanded_limit,
    limit,
    n_library_chi: int = 0,
    library_size: int = 1,
):
    """Trim sampled chi so the rotamers they enumerate stay bounded.

    ``depths`` gives each sample's depth in the kinforest that builds it, and
    ``n_library_chi`` how many chi a borrowed rotamer library already defines.
    The expansions are kept while the product stays under ``expanded_limit``
    and dropped at ``limit``; past it, chi freeze from the tip inward. A proton
    chi retains at least one mean, so its hydrogen is still placed. Library
    multiplicity is explicit; the number of chi cannot predict a library's size.
    """
    return tuple(
        cs
        for _, cs in _budgeted_chi_samples(
            samples,
            depths,
            expanded_limit,
            limit,
            n_library_chi=n_library_chi,
            library_size=library_size,
        )
    )


def _budgeted_chi_samples(
    samples, depths, expanded_limit, limit, n_library_chi=0, library_size=1
):
    """Return (input index, sample) pairs, preserving ownership across blocks.

    ``n_library_chi`` excludes chi already supplied by the same block's library.
    ``library_size`` accounts for an independent library in a group product
    without excluding identically named chi on its attached blocks.
    """
    kept = [
        (i, cs, d)
        for i, (cs, d) in enumerate(zip(samples, depths, strict=True))
        if cs.is_proton or int(cs.chi_dihedral[3:]) > n_library_chi
    ]
    indices = [i for i, _, _ in kept]
    samples = [cs for _, cs, _ in kept]
    depths = [d for _, _, d in kept]
    library = library_size
    if library * n_conformers(samples, True) <= expanded_limit:
        return tuple(zip(indices, samples))

    samples = [attr.evolve(cs, expansions=()) for cs in samples]
    order = sorted(
        (i for i, cs in enumerate(samples) if not cs.is_proton),
        key=lambda i: (depths[i], int(samples[i].chi_dihedral[3:])),
        reverse=True,
    )
    total = library * n_conformers(samples, False)
    for index in order:
        if total <= limit:
            break
        total //= len(samples[index].samples)
        samples[index] = None
    # A proton still needs a placement, not every alternative placement.
    for index in sorted(
        (i for i, cs in enumerate(samples) if cs is not None and cs.is_proton),
        key=lambda i: (depths[i], int(samples[i].chi_dihedral[3:])),
        reverse=True,
    ):
        if total <= limit:
            break
        total //= len(samples[index].samples)
        samples[index] = attr.evolve(samples[index], samples=samples[index].samples[:1])
    if total > limit:
        raise ValueError(
            f"Sampling budget {limit} cannot fit the required {library} library states"
        )
    return tuple((i, cs) for i, cs in zip(indices, samples) if cs is not None)


def chi_depths(rkd, chi_atoms):
    """Depth in a kinforest of the atom each chi turns about.

    The freeze order: a chi deep in the tree moves few atoms, one near the root
    swings everything past it. ``rkd`` is the ResidueKinforestData for the tree
    that will build these chi -- a single residue's for optH, a conjugated
    group's for full packing -- and ``chi_atoms`` gives each chi's defining
    atom in that tree's atom numbering.
    """
    depth = {}
    for atom in rkd.bfto_2_orig:
        parent = rkd.preds[atom]
        depth[int(atom)] = 0 if parent < 0 else depth[int(parent)] + 1
    return [depth.get(int(a), 0) for a in chi_atoms]
