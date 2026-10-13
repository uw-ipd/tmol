"""Build canonical proteins from Biotite arrays using the shared PDB mapping."""

import biotite.structure as struc
import numpy as np

from .canonical_ordering import (
    _canonical_form_from_records,
    default_canonical_ordering,
    default_packed_block_types,
)
from .pdb_parsing import _records_from_biotite
from .pose_stack_construction import pose_stack_from_canonical_form


def canonical_form_from_biotite(atoms, *, canonical_ordering=None):
    """Convert an AtomArray or same-topology AtomArrayStack to canonical form."""
    co = canonical_ordering or default_canonical_ordering()
    models = list(atoms) if isinstance(atoms, struc.AtomArrayStack) else [atoms]
    if not models:
        raise ValueError("structure contains no models")
    forms = [
        _canonical_form_from_records(_records_from_biotite(model), co)
        for model in models
    ]
    return {
        key: np.concatenate([form[key] for form in forms], axis=0) for key in forms[0]
    }


def pose_stack_from_biotite(
    atoms, *, canonical_ordering=None, packed_block_types=None, **kwargs
):
    """Reconstruct and pack canonical proteins; unsupported chemistry raises."""
    canonical = canonical_form_from_biotite(
        atoms, canonical_ordering=canonical_ordering
    )
    return pose_stack_from_canonical_form(
        canonical, packed_block_types or default_packed_block_types(), **kwargs
    )
