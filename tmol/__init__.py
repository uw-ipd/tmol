# BEFORE IMPORTING TORCH:
#
# def get_gpu_compute_capability():
#     import subprocess
#     import re
#     try:
#         # Run nvidia-smi command to get XML output
#         result = subprocess.run(
#             ['nvidia-smi', '-q', '-x'],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             check=True,
#             text=True
#         )
#         xml_output = result.stdout
#
#         # Use regex to find the compute capability (major and minor)
#         # This approach can be brittle if NVIDIA changes the output format significantly
#         major_match = re.search(r'<cuda_compute_capability><major>(\d+)</major>', xml_output)
#         minor_match = re.search(r'<cuda_compute_capability><minor>(\d+)</minor></cuda_compute_capability>', xml_output)
#         if major_match and minor_match:
#             major = major_match.group(1)
#             minor = minor_match.group(1)
#             print("major:", major, "minor:", minor, f"{major}.{minor}")
#             return f"{major}.{minor}"
#         else:
#             return "Could not parse compute capability from nvidia-smi output."
#     except FileNotFoundError:
#         return "nvidia-smi command not found. Ensure NVIDIA drivers are installed and in your PATH."
#     except subprocess.CalledProcessError as e:
#         return f"Error running nvidia-smi: {e.stderr}"
#
# import os
#
# if os.environ.get('TORCH_CUDA_ARCH_LIST') is None:
#     compute_capability = get_gpu_compute_capability()
#     os.environ['TORCH_CUDA_ARCH_LIST'] = compute_capability


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
