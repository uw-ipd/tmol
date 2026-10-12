import numpy as np

from tmol_jax.database import ParameterDatabase
from tmol_jax.io import default_packed_block_types
from tmol_jax.score.backbone_torsion.backbone_torsion_energy_term import (
    BackboneTorsionEnergyTerm,
)


def test_rama_reference_selects_variant_tables_and_proline_identity():
    database = ParameterDatabase.get_default()
    block_types = default_packed_block_types().active_block_types
    by_name = {block_type.name: block_type for block_type in block_types}
    term = BackboneTorsionEnergyTerm(database)

    for variant, reference in (
        ("HIS_D", "HIS_D"),
        ("HIS_POS", "HIS_POS"),
        ("CYD", "CYD"),
        ("ALA:nterm", "ALA"),
        ("ALA:cterm", "ALA"),
        ("PRO:nterm", "PRO"),
    ):
        block_type = by_name[variant]
        assert block_type.rama_reference == reference
    np.testing.assert_array_equal(
        term.rama_access.reference_to_table_idx_2["HIS_D"],
        term.rama_access.reference_to_table_idx_2["HIS"],
    )
    np.testing.assert_array_equal(
        term.omega_access.reference_to_table_idx_2["HIS_D"],
        term.omega_access.reference_to_table_idx_2["HIS"],
    )

    for reference in ("HIS_POS", "CYD", None):
        np.testing.assert_array_equal(
            term.rama_access.reference_to_table_idx_2.get(reference, (-1, -1)),
            [-1, -1],
        )
        np.testing.assert_array_equal(
            term.omega_access.reference_to_table_idx_2.get(reference, (-1, -1)),
            [-1, -1],
        )
