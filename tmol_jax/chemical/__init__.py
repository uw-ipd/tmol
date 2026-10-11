"""JAX port of ``tmol.chemical``.

Re-exports the major classes and helpers used by downstream code.
"""

from __future__ import annotations

from tmol_jax.chemical.constants import (
    MAX_PATHS_FROM_CONNECTION,
    MAX_SIG_BOND_SEPARATION,
    canonical_aa_1letter,
    canonical_aa_3letter,
    one2three,
    three2one,
)
from tmol_jax.chemical.ideal_coords import (
    build_coords_from_icoors,
    build_ideal_coords,
)
from tmol_jax.chemical.all_bonds import (
    AllBonds,
    all_bonds_for_pose_stack,
    bonds_and_bond_ranges,
)
from tmol_jax.chemical.restypes import (
    AtomIndex,
    BondCount,
    ConnectionIndex,
    IcoorIndex,
    RefinedResidueType,
    ResName3,
    ResidueTypeSet,
    UnresolvedAtomID,
    uaid_t,
)
from tmol_jax.chemical.patched_chemdb import (
    PatchedChemicalDatabase,
    default_patched_chemdb,
)

__all__ = [
    # constants
    "MAX_SIG_BOND_SEPARATION",
    "MAX_PATHS_FROM_CONNECTION",
    "one2three",
    "three2one",
    "canonical_aa_1letter",
    "canonical_aa_3letter",
    # ideal_coords
    "build_coords_from_icoors",
    "build_ideal_coords",
    # all_bonds
    "bonds_and_bond_ranges",
    "AllBonds",
    "all_bonds_for_pose_stack",
    # restypes
    "AtomIndex",
    "BondCount",
    "ConnectionIndex",
    "IcoorIndex",
    "ResName3",
    "UnresolvedAtomID",
    "uaid_t",
    "RefinedResidueType",
    "ResidueTypeSet",
    # patched_chemdb
    "PatchedChemicalDatabase",
    "default_patched_chemdb",
]
