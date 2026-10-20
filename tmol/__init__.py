"""TMol: GPU-accelerated molecular modeling with PyTorch."""

# flake8: noqa

# Load precompiled native operators.
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


from tmol._cpp_lib import has_cuda
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
    "CartesianMinimizer",
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
    "atom_label",
    "atom_array_from_file",
    "atom_array_from_pose_stack",
    "file_from_atom_array",
    "file_from_pose_stack",
    "pose_stack_from_atom_array",
    "pose_stack_from_file",
    "beta2016_score_function",
    "block_for_residue_label",
    "blocks_for_residue_labels",
    "build_kinforest_network",
    "create_mainchain_coordinate_constraints",
    "default_canonical_ordering",
    "default_packed_block_types",
    "extended_pose_stack_from_sequences",
    "fast_relax",
    "get_ligand_mask",
    "get_named_torsions",
    "get_polymer_type_mask",
    "get_sequence",
    "get_torsion_names",
    "include_paths",
    "has_cuda",
    "metal_blocks",
    "mutate_residue",
    "mutate_to_sequence",
    "one2three",
    "overlay_view",
    "pose_stack_from_canonical_form",
    "residue_label",
    "run_cart_min",
    "run_kin_min",
    "run_min",
    "selection_gallery",
    "set_named_torsions",
    "switchable_view",
    "three2one",
    "torsion_labels",
    "view",
]


_LAZY_ATTRS = {}
for _module, _names in (
    ("tmol.chemical", ("one2three", "three2one")),
    ("tmol.database", ("ParameterDatabase",)),
    (
        "tmol.io",
        (
            "CanonicalOrdering",
            "atom_array_from_file",
            "atom_array_from_pose_stack",
            "file_from_atom_array",
            "file_from_pose_stack",
            "pose_stack_from_atom_array",
            "pose_stack_from_file",
            "default_canonical_ordering",
            "default_packed_block_types",
            "extended_pose_stack_from_sequences",
            "overlay_view",
            "pose_stack_from_canonical_form",
            "selection_gallery",
            "switchable_view",
            "view",
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
            "torsion_labels",
        ),
    ),
    (
        "tmol.optimization",
        (
            "CartesianMinimizer",
            "build_kinforest_network",
            "run_cart_min",
            "run_kin_min",
            "run_min",
        ),
    ),
    (
        "tmol.pose",
        (
            "ConstraintSet",
            "PackedBlockTypes",
            "PoseStack",
            "atom_label",
            "block_for_residue_label",
            "blocks_for_residue_labels",
            "get_ligand_mask",
            "get_named_torsions",
            "get_polymer_type_mask",
            "get_sequence",
            "get_torsion_names",
            "metal_blocks",
            "residue_label",
        ),
    ),
    (
        "tmol.pack",
        ("mutate_residue", "mutate_to_sequence"),
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
