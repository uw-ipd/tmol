"""Hydrogen-bond tables must follow both databases without retaining either."""

import gc
import weakref

import attr
import pytest
import torch

from tmol.score.hbond._params import HBondParamResolver, CompactedHBondDatabase
from tmol.utility.weak_identity_cache import WeakIdentityLRU


@pytest.mark.parametrize("resolver", [HBondParamResolver, CompactedHBondDatabase])
@pytest.mark.parametrize("expire", ["chemical", "hbond"])
def test_parameter_tables_are_bounded_and_either_owner_can_expire(
    default_database, torch_device, monkeypatch, resolver, expire
):
    cache = WeakIdentityLRU(2)
    monkeypatch.setattr(resolver, "_from_db_cache", cache)
    chemical = attr.evolve(default_database.chemical)
    hb = attr.evolve(default_database.scoring.hbond)
    table = resolver.from_database(chemical, hb, torch_device)
    assert resolver.from_database(chemical, hb, torch_device) is table
    chemicals = [chemical, attr.evolve(chemical)]
    hbs = [hb, attr.evolve(hb)]
    other = resolver.from_database(chemicals[1], hb, torch_device)
    assert other is not table
    assert resolver.from_database(chemical, hb, torch_device) is table
    resolver.from_database(chemical, hbs[1], torch_device)
    assert len(cache) == 2
    assert resolver.from_database(chemical, hb, torch_device) is table
    assert resolver.from_database(chemicals[1], hb, torch_device) is not other
    assert len(cache) == 2
    # Refill exactly one entry so expiry of either source must remove it.
    cache = WeakIdentityLRU(2)
    monkeypatch.setattr(resolver, "_from_db_cache", cache)
    retained = resolver.from_database(chemical, hb, torch_device)
    del chemicals, hbs
    reference = weakref.ref(chemical if expire == "chemical" else hb)
    if expire == "chemical":
        del chemical
    else:
        del hb
    gc.collect()
    assert reference() is None
    assert len(cache) == 0
    tensors = (
        retained.pair_params.AHdist.coeffs
        if resolver is HBondParamResolver
        else retained.pair_poly_table
    )
    assert torch.isfinite(tensors).all()


@pytest.mark.parametrize("resolver", [HBondParamResolver, CompactedHBondDatabase])
def test_equivalent_device_spellings_reuse_parameter_tables(
    default_database, torch_device, resolver, monkeypatch
):
    cache = WeakIdentityLRU(2)
    monkeypatch.setattr(resolver, "_from_db_cache", cache)
    first = resolver.from_database(
        default_database.chemical, default_database.scoring.hbond, torch_device
    )
    alias = torch.device("cuda" if torch_device.type == "cuda" else "cpu:0")

    def unexpected_rebuild(*args, **kwargs):
        raise AssertionError("Equivalent devices must reuse the parameter table")

    monkeypatch.setattr(resolver, "_from_database", unexpected_rebuild)
    assert (
        resolver.from_database(
            default_database.chemical, default_database.scoring.hbond, alias
        )
        is first
    )
