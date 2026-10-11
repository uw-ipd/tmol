"""Zero-valued patch coordinates are explicit overrides, not missing values."""

from tmol_jax.database.chemical import Icoor, IcoorVariant
from tmol_jax.chemical.patched_chemdb import update_icoor


def test_patch_distinguishes_zero_from_absent():
    source = Icoor("A", 1.5, 0.8, 1.2, "B", "C", "D")
    mapping = {"<A>": "A"}
    inherited = update_icoor((source,), (IcoorVariant("A", source="<A>"),), (), mapping)
    zeroed = update_icoor(
        (source,), (IcoorVariant("A", source="<A>", phi=0.0),), (), mapping
    )
    assert inherited[0].phi == 1.5
    assert zeroed[0].phi == 0.0
    assert zeroed[0].theta == 0.8
