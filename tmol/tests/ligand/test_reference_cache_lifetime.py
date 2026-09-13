"""Reference caching must not retain generated databases or alias identities."""

from dataclasses import dataclass
import gc
import weakref

from tmol.ligand import _rotamer_reference as reference


@dataclass
class Database:
    """An unhashable, weak-referenceable database stand-in."""

    label: str


def test_cache_uses_identity_and_releases_database(monkeypatch):
    monkeypatch.setattr(reference, "_PROFILE_CACHE", reference.WeakIdentityLRU())
    calls = []

    def profiles(db, libraries):
        calls.append(db.label)
        return [object()]

    monkeypatch.setattr(reference, "reference_profiles", profiles)
    first, second = Database("same"), Database("same")
    a = reference._cached_profiles(first, ["LYS", "PRO"])
    assert reference._cached_profiles(first, ["PRO", "LYS"]) is a
    assert reference._cached_profiles(second, ["LYS", "PRO"]) is not a
    assert len(calls) == 2
    ref = weakref.ref(first)
    del first
    gc.collect()
    assert ref() is None
    assert len(reference._PROFILE_CACHE) == 1


def test_cache_is_bounded_and_evicts_least_recently_used(monkeypatch):
    monkeypatch.setattr(reference, "_PROFILE_CACHE", reference.WeakIdentityLRU(2))
    monkeypatch.setattr(reference, "reference_profiles", lambda *args: [object()])
    dbs = [Database(str(i)) for i in range(3)]
    a = reference._cached_profiles(dbs[0], ["PRO"])
    b = reference._cached_profiles(dbs[1], ["PRO"])
    assert reference._cached_profiles(dbs[0], ["PRO"]) is a
    reference._cached_profiles(dbs[2], ["PRO"])
    assert len(reference._PROFILE_CACHE) == 2
    assert reference._cached_profiles(dbs[0], ["PRO"]) is a
    assert reference._cached_profiles(dbs[1], ["PRO"]) is not b


def test_polymer_profiles_release_real_database_and_remain_bounded(monkeypatch):
    import attr
    from tmol.database import ParameterDatabase
    from tmol.ligand import _polymer_profile as polymer

    cache = reference.WeakIdentityLRU(3)
    monkeypatch.setattr(polymer, "_POLYMER_PROFILE_CACHE", cache)
    database = attr.evolve(ParameterDatabase.get_default().chemical)
    alpha = polymer.alpha_profile(database)
    dna = polymer.na_profile(database, "dna")
    rna = polymer.na_profile(database, "rna")
    assert alpha is polymer.alpha_profile(database)
    assert dna is polymer.na_profile(database, "dna")
    assert rna is polymer.na_profile(database, "rna")
    assert dna != rna
    assert len(cache) == 3
    second = attr.evolve(database)
    assert polymer.alpha_profile(second) is not alpha
    assert len(cache) == 3
    owner = weakref.ref(database)
    del database
    gc.collect()
    assert owner() is None
    assert len(cache) == 1  # Only the second database's alpha profile survives.


def test_cache_checks_referent_identity_even_when_integer_key_matches():
    cache = reference.WeakIdentityLRU()
    first, old_owner = Database("first"), Database("old")
    stale = object()
    # Simulate an id reused after collection without relying on allocator luck.
    cache._entries[((id(first),), "profile")] = ((weakref.ref(old_owner),), stale)
    actual = cache.get_or_create(first, "profile", object)
    assert actual is not stale
    assert cache.get_or_create(first, "profile", object) is actual


def test_simultaneous_misses_publish_one_cached_value():
    from concurrent.futures import ThreadPoolExecutor
    import threading

    cache = reference.WeakIdentityLRU()
    database = Database("shared")
    barrier = threading.Barrier(2)

    def factory():
        barrier.wait(timeout=10)
        return object()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(cache.get_or_create, database, "profile", factory)
            for _ in range(2)
        ]
        first, second = [future.result(timeout=15) for future in futures]
    assert first is second
    assert cache.get_or_create(database, "profile", object) is first
    assert len(cache) == 1
