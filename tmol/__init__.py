from tmol.chemical.restypes import three2one  # noqa: F401
from tmol.io.canonical_ordering import (  # noqa: F401
    # ordered_canonical_aa_atoms,
    # ordered_canonical_aa_atoms_v2,
    # max_n_canonical_atoms,
    canonical_form_from_pdb_lines,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form  # noqa: F401
from tmol.score import beta2016_score_function  # noqa: F401


def include_paths():
    """C++/CUDA include paths for tmol components."""

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]
