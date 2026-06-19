"""Vendored third-party code used by the tmol ligand pipeline.

- :mod:`atomworks_rdkit`: AtomArray -> RDKit conversion ported from atomworks.
- :mod:`xyz2mol_tm`: transition-metal-complex bond perception (MIT, Jensen Group).
"""

from tmol.ligand.external.atomworks_rdkit import (
    atom_array_to_rdkit,
    ccd_code_to_rdkit,
    fix_mol,
)

__all__ = ["atom_array_to_rdkit", "ccd_code_to_rdkit", "fix_mol"]
