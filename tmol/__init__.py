# flake8: noqa

# Load pre-compiled C++/CUDA extensions (TORCH_LIBRARY ops).
# This must happen early so that torch.ops.tmol_* namespaces are available
# before any compiled module is imported.
import contextlib
from importlib.metadata import PackageNotFoundError, version

from tmol._load_ext import ensure_compiled_or_jit as _ensure_compiled_or_jit

# Extensions may not be built yet (e.g. during sdist creation).
# Individual compiled.py modules will raise a clear error if needed.
with contextlib.suppress(Exception):
    _ensure_compiled_or_jit()

from tmol.chemical.restypes import one2three, three2one
from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_pdb
from tmol.io.canonical_ordering import (
    CanonicalOrdering,
    canonical_form_from_pdb,
    default_canonical_ordering,
    default_packed_block_types,
)
from tmol.io.pose_stack_construction import (
    pose_stack_from_canonical_form,
)
from tmol.io.pose_stack_from_openfold import (
    canonical_form_from_openfold,
    canonical_ordering_for_openfold,
    packed_block_types_for_openfold,
    pose_stack_from_openfold,
)
from tmol.io.pose_stack_from_rosettafold2 import (
    canonical_form_from_rosettafold2,
    canonical_ordering_for_rosettafold2,
    packed_block_types_for_rosettafold2,
    pose_stack_from_rosettafold2,
)
from tmol.io.write_pose_stack_pdb import (
    atom_records_from_pose_stack,
    write_pose_stack_pdb,
)
from tmol.kinematics.datatypes import KinematicModuleData
from tmol.kinematics.fold_forest import EdgeType, FoldForest
from tmol.kinematics.move_map import CartesianMoveMap, MoveMap
from tmol.optimization.minimizers import (
    build_kinforest_network,
    run_cart_min,
    run_kin_min,
    run_min,
)
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.score import beta2016_score_function
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.score.constraint.utility import create_mainchain_coordinate_constraints
from tmol.relax.fast_relax import fast_relax

try:
    __version__ = version("tmol")
except PackageNotFoundError:
    __version__ = "unknown version"


def include_paths():
    """C++/CUDA include paths for tmol components."""

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]
