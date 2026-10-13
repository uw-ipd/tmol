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


def test_ideal_coordinates_follow_atom_names_when_atom_order_changes():
    import attrs
    import numpy as np
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    database = ParameterDatabase.get_default().chemical
    raw = next(residue for residue in database.residues if residue.name == "ALA")
    residue = _refined_from_raw(attrs.evolve(raw, atoms=raw.atoms[::-1]))
    packed = PackedBlockTypes.from_restypes(database, [residue])
    nitrogen = packed.ideal_coords[0, residue.atom_to_idx["N"]]
    alpha_carbon = packed.ideal_coords[0, residue.atom_to_idx["CA"]]
    np.testing.assert_allclose(nitrogen, 0, atol=1e-7)
    np.testing.assert_allclose(
        np.linalg.norm(alpha_carbon - nitrogen),
        residue.icoors[residue.icoors_index["CA"]].d,
        atol=1e-6,
    )
