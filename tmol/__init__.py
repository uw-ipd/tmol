from importlib.metadata import PackageNotFoundError, version

from tmol.database import ParameterDatabase  # noqa: F401
from tmol.chemical.restypes import one2three, three2one  # noqa: F401
from tmol.pose.packed_block_types import PackedBlockTypes  # noqa: F401
from tmol.pose.pose_stack import PoseStack  # noqa: F401

from tmol.kinematics.fold_forest import FoldForest, EdgeType  # noqa: F401
from tmol.kinematics.datatypes import KinematicModuleData  # noqa: F401
from tmol.kinematics.move_map import MoveMap  # noqa: F401


from tmol.io import pose_stack_from_pdb  # noqa: F401
from tmol.io.pose_stack_construction import (  # noqa: F401
    pose_stack_from_canonical_form,
)
from tmol.io.canonical_ordering import (  # noqa: F401
    CanonicalOrdering,
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_from_openfold import (  # noqa: F401
    pose_stack_from_openfold,
    canonical_form_from_openfold,
    canonical_ordering_for_openfold,
    packed_block_types_for_openfold,
)
from tmol.io.pose_stack_from_rosettafold2 import (  # noqa: F401
    pose_stack_from_rosettafold2,
    canonical_form_from_rosettafold2,
    canonical_ordering_for_rosettafold2,
    packed_block_types_for_rosettafold2,
)
from tmol.io.write_pose_stack_pdb import (  # noqa: F401
    write_pose_stack_pdb,
    atom_records_from_pose_stack,
)
from tmol.score import beta2016_score_function  # noqa: F401
from tmol.score.score_function import ScoreFunction  # noqa: F401
from tmol.score.score_types import ScoreType  # noqa: F401


from tmol.optimization.kin_min import build_kinforest_network, run_kin_min  # noqa: F401

try:
    __version__ = version("tmol")
except PackageNotFoundError:
    __version__ = "unknown version"


def include_paths():
    """C++/CUDA include paths for tmol components."""

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]
