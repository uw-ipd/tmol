# flake8: noqa

# Load pre-compiled C++/CUDA extensions (TORCH_LIBRARY ops).
# This must happen early so that torch.ops.tmol_* namespaces are available
# before any compiled module is imported.
import contextlib
from importlib.metadata import PackageNotFoundError, version


def include_paths():
    """C++/CUDA include paths for tmol components.

    Defined before other imports because JIT extension loading
    (tmol.utility.cpp_extension) imports this during module init.
    """

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]


from tmol._load_ext import ensure_compiled_or_jit as _ensure_compiled_or_jit

# Extensions may not be built yet (e.g. during sdist creation).
# Individual compiled.py modules will raise a clear error if needed.
with contextlib.suppress(Exception):
    _ensure_compiled_or_jit()

from tmol.chemical import one2three, three2one
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.io import (
    pose_stack_to_pdb_string,
    selection_gallery,
    switchable_view,
    view,
)
from tmol.io import (
    CanonicalOrdering,
    canonical_form_from_pdb,
    default_canonical_ordering,
    default_packed_block_types,
)
from tmol.io import (
    pose_stack_from_canonical_form,
)
from tmol.io import (
    canonical_form_from_openfold,
    canonical_ordering_for_openfold,
    packed_block_types_for_openfold,
    pose_stack_from_openfold,
)
from tmol.io import (
    canonical_form_from_rosettafold2,
    canonical_ordering_for_rosettafold2,
    packed_block_types_for_rosettafold2,
    pose_stack_from_rosettafold2,
)
from tmol.io import (
    atom_records_from_pose_stack,
    write_pose_stack_pdb,
)
from tmol.kinematics import KinematicModuleData
from tmol.kinematics import EdgeType, FoldForest
from tmol.kinematics import CartesianMoveMap, MoveMap
from tmol.optimization import (
    build_kinforest_network,
    run_cart_min,
    run_kin_min,
    run_min,
)
from tmol.pose import PackedBlockTypes
from tmol.pose import PoseStack
from tmol.pose import ConstraintSet
from tmol.io import extended_pose_stack_from_sequences
from tmol.pose import get_named_torsions, get_torsion_names
from tmol.kinematics import set_named_torsions
from tmol.score import beta2016_score_function
from tmol.score import ScoreFunction
from tmol.score import ScoreType
from tmol.score.constraint import (
    ConstraintEnergyTerm,
)
from tmol.score.constraint import create_mainchain_coordinate_constraints
from tmol.relax import fast_relax

try:
    __version__ = version("tmol")
except PackageNotFoundError:
    __version__ = "unknown version"
