from tmol.io.canonical_ordering import (
    aa_codes,
    ordered_canonical_aa_atoms,
    ordered_canonical_aa_atoms_v2,
)
from tmol.io.basic_resolution import pose_stack_from_canonical_form


def include_paths():
    """C++/CUDA include paths for tmol components."""

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]
