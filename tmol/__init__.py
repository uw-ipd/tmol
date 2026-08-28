"""TMol: GPU-accelerated molecular modeling with PyTorch."""

# flake8: noqa

# Load pre-compiled C++/CUDA extensions (TORCH_LIBRARY ops).
# This must happen early so that torch.ops.tmol_* namespaces are available
# before any compiled module is imported.
import contextlib
from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError, version


def include_paths():
    """C++/CUDA include paths for tmol components.

    Defined before other imports because JIT extension loading
    (tmol.utility._cpp_extension) imports this during module init.
    """

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]


from tmol._load_ext import ensure_compiled_or_jit as _ensure_compiled_or_jit

# Extensions may not be built yet (e.g. during sdist creation).
# Individual compiled.py modules will raise a clear error if needed.
with contextlib.suppress(Exception):
    _ensure_compiled_or_jit()

try:
    __version__ = version("tmol")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "CanonicalOrdering",
    "CartesianMoveMap",
    "ConstraintEnergyTerm",
    "ConstraintSet",
    "EdgeType",
    "FoldForest",
    "KinematicModuleData",
    "MoveMap",
    "PackedBlockTypes",
    "ParameterDatabase",
    "PoseStack",
    "ScoreFunction",
    "ScoreType",
    "atom_records_from_pose_stack",
    "beta2016_score_function",
    "build_kinforest_network",
    "canonical_form_from_openfold",
    "canonical_form_from_pdb",
    "canonical_form_from_rosettafold2",
    "canonical_ordering_for_openfold",
    "canonical_ordering_for_rosettafold2",
    "create_mainchain_coordinate_constraints",
    "default_canonical_ordering",
    "default_packed_block_types",
    "extended_pose_stack_from_sequences",
    "fast_relax",
    "get_named_torsions",
    "get_torsion_names",
    "include_paths",
    "one2three",
    "packed_block_types_for_openfold",
    "packed_block_types_for_rosettafold2",
    "pose_stack_from_canonical_form",
    "pose_stack_from_openfold",
    "pose_stack_from_pdb",
    "pose_stack_from_rosettafold2",
    "pose_stack_to_pdb_string",
    "run_cart_min",
    "run_kin_min",
    "run_min",
    "selection_gallery",
    "set_named_torsions",
    "switchable_view",
    "three2one",
    "view",
    "write_pose_stack_pdb",
]


_LAZY_ATTRS = {}
for _module, _names in (
    ("tmol.chemical", ("one2three", "three2one")),
    ("tmol.database", ("ParameterDatabase",)),
    (
        "tmol.io",
        (
            "CanonicalOrdering",
            "atom_records_from_pose_stack",
            "canonical_form_from_openfold",
            "canonical_form_from_pdb",
            "canonical_form_from_rosettafold2",
            "canonical_ordering_for_openfold",
            "canonical_ordering_for_rosettafold2",
            "default_canonical_ordering",
            "default_packed_block_types",
            "extended_pose_stack_from_sequences",
            "packed_block_types_for_openfold",
            "packed_block_types_for_rosettafold2",
            "pose_stack_from_canonical_form",
            "pose_stack_from_openfold",
            "pose_stack_from_pdb",
            "pose_stack_from_rosettafold2",
            "pose_stack_to_pdb_string",
            "selection_gallery",
            "switchable_view",
            "view",
            "write_pose_stack_pdb",
        ),
    ),
    (
        "tmol.kinematics",
        (
            "CartesianMoveMap",
            "EdgeType",
            "FoldForest",
            "KinematicModuleData",
            "MoveMap",
            "set_named_torsions",
        ),
    ),
    (
        "tmol.optimization",
        ("build_kinforest_network", "run_cart_min", "run_kin_min", "run_min"),
    ),
    (
        "tmol.pose",
        (
            "ConstraintSet",
            "PackedBlockTypes",
            "PoseStack",
            "get_named_torsions",
            "get_torsion_names",
        ),
    ),
    (
        "tmol.score",
        ("ScoreFunction", "ScoreType", "beta2016_score_function"),
    ),
    (
        "tmol.score.constraint",
        ("ConstraintEnergyTerm", "create_mainchain_coordinate_constraints"),
    ),
    ("tmol.relax", ("fast_relax",)),
):
    _LAZY_ATTRS.update(dict.fromkeys(_names, _module))
del _module, _names


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
