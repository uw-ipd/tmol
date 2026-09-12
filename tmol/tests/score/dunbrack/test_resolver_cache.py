"""Derived Dunbrack tables must not keep every private parameter set alive."""

import gc
import weakref

import attr
import pytest
import torch

from tmol.score.dunbrack import DunbrackParamResolver


def private_database(database, label):
    return attr.evolve(
        database,
        dun_lookup=(
            *database.dun_lookup,
            attr.evolve(database.dun_lookup[0], residue_name=label),
        ),
    )


def test_released_database_and_resolver_are_not_retained(
    default_database, torch_device
):
    database = private_database(default_database.scoring.dun, "cache_lifetime")
    resolver = DunbrackParamResolver.from_database(database, torch_device)
    assert DunbrackParamResolver.from_database(database, torch_device) is resolver
    owner_ref, resolver_ref = weakref.ref(database), weakref.ref(resolver)
    del database
    gc.collect()
    assert owner_ref() is None
    # A live consumer may still use the compiled tables after owner expiry.
    assert bool(torch.isfinite(resolver.sampling_db.rotameric_mean_tables).all())
    del resolver
    gc.collect()
    assert resolver_ref() is None


def test_resolver_cache_is_bounded_while_owners_remain_live(
    default_database, torch_device
):
    owners = [
        private_database(default_database.scoring.dun, f"cache_capacity_{i}")
        for i in range(5)
    ]
    references = []
    for database in owners:
        resolver = DunbrackParamResolver.from_database(database, torch_device)
        references.append(weakref.ref(resolver))
    del resolver
    gc.collect()
    assert references[0]() is None
    assert all(ref() is not None for ref in references[1:])
    assert len(DunbrackParamResolver._from_dun_db_cache) <= 4
    latest = references[-1]()
    assert DunbrackParamResolver.from_database(owners[-1], torch_device) is latest


def test_resolver_device_aliases_share_tables(default_database, torch_device):
    database = private_database(default_database.scoring.dun, "cache_device_alias")
    if torch_device.type == "cuda":
        alias = torch.device("cuda")
        concrete = torch.device("cuda", torch.cuda.current_device())
    else:
        alias, concrete = torch.device("cpu", 0), torch.device("cpu")
    first = DunbrackParamResolver.from_database(database, alias)
    second = DunbrackParamResolver.from_database(database, concrete)
    assert first is second
    assert first.device == concrete


@pytest.mark.parametrize("subclass_first", [False, True])
def test_resolver_subclass_identity(default_database, torch_device, subclass_first):
    class SpecializedResolver(DunbrackParamResolver):
        pass

    database = private_database(default_database.scoring.dun, "cache_subclass")
    classes = [DunbrackParamResolver, SpecializedResolver]
    if subclass_first:
        classes.reverse()
    for cls in classes:
        resolver = cls.from_database(database, torch_device)
        assert type(resolver) is cls
        assert cls.from_database(database, torch_device) is resolver
