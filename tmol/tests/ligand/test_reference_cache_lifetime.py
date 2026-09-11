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
    monkeypatch.setattr(reference, "_PROFILE_CACHE", reference.OrderedDict())
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
    monkeypatch.setattr(reference, "_PROFILE_CACHE", reference.OrderedDict())
    monkeypatch.setattr(reference, "_PROFILE_CACHE_LIMIT", 2)
    monkeypatch.setattr(reference, "reference_profiles", lambda *args: [object()])
    dbs = [Database(str(i)) for i in range(3)]
    a = reference._cached_profiles(dbs[0], ["PRO"])
    b = reference._cached_profiles(dbs[1], ["PRO"])
    assert reference._cached_profiles(dbs[0], ["PRO"]) is a
    reference._cached_profiles(dbs[2], ["PRO"])
    assert len(reference._PROFILE_CACHE) == 2
    assert reference._cached_profiles(dbs[0], ["PRO"]) is a
    assert reference._cached_profiles(dbs[1], ["PRO"]) is not b
