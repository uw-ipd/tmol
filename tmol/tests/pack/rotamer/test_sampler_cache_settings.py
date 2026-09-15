"""Sequential samplers sharing chemical types must honor their own settings."""

import attr
import pytest
import torch

from tmol.io import default_packed_block_types
from tmol.pack.rotamer import NaChiRotamerSampler, OptHSampler


@pytest.mark.parametrize("kind", ["na", "opth"])
def test_changing_budget_rebuilds_both_annotation_levels(
    kind, default_database, torch_device
):
    pbt = default_packed_block_types(torch_device)
    sampler = (
        NaChiRotamerSampler.from_database(default_database, torch_device)
        if kind == "na"
        else OptHSampler()
    )
    counts = []
    for limit in (100000, 1, 100000):
        current = attr.evolve(
            sampler, chi_sample_expanded_limit=limit, chi_sample_limit=limit
        )
        if kind == "na":
            cache = current.annotate_packed_block_types(pbt)
            counts.append(cache["n_combos"].clone())
            assert current.annotate_packed_block_types(pbt) is cache
        else:
            current._annotate_packed_block_types(pbt)
            cache = pbt.opth_sample_cache
            counts.append(cache.n_proton_samples.clone())
            current._annotate_packed_block_types(pbt)
            assert pbt.opth_sample_cache is cache
    assert bool(torch.any(counts[0] > counts[1]))
    torch.testing.assert_close(counts[0], counts[2])


def test_changing_flip_setting_updates_buildable_types(torch_device):
    pbt = default_packed_block_types(torch_device)
    indices = torch.tensor(
        [i for i, bt in enumerate(pbt.active_block_types) if bt.name == "ASN"],
        device=torch_device,
    )
    assert len(indices) == 1
    for flip in (False, True, False, True):
        sampler = OptHSampler(flip_NHQ=flip)
        assert bool(sampler.defines_rotamers_for_bts(pbt, indices)[0]) == flip


def test_na_element_assignments_are_part_of_annotation_identity(
    default_database, torch_device
):
    pbt = default_packed_block_types(torch_device)
    sampler = NaChiRotamerSampler.from_database(default_database, torch_device)
    initial = sampler.annotate_packed_block_types(pbt)
    assert bool(torch.any(initial["ring"] >= 0))
    # Changing the ring oxygen's element removes the identifiable heterocycle.
    # Exercise mutation of the accepted dictionary, not just a new sampler.
    oxygen_types = {
        name: element
        for name, element in sampler.element_for_atom_type.items()
        if element == "O"
    }
    sampler.element_for_atom_type.update({name: "C" for name in oxygen_types})
    changed = sampler.annotate_packed_block_types(pbt)
    assert bool(torch.all(changed["ring"] == -1))
    sampler.element_for_atom_type.update(oxygen_types)
    restored = sampler.annotate_packed_block_types(pbt)
    torch.testing.assert_close(restored["ring"], initial["ring"])
