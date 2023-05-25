from tmol.io.canonical_ordering import (  # noqa: F401
    aa_codes,
    ordered_canonical_aa_atoms,
    ordered_canonical_aa_atoms_v2,
    max_n_canonical_atoms,
)
from tmol.io.basic_resolution import pose_stack_from_canonical_form  # noqa: F401
from tmol.score import beta2016_score_function  # noqa: F401


def include_paths():
    """C++/CUDA include paths for tmol components."""

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]
