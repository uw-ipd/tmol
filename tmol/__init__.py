"""TMol: GPU-accelerated molecular modeling with PyTorch."""

# flake8: noqa

# Load precompiled native operators.
# This must happen early so that torch.ops.tmol_* namespaces are available
# before any compiled module is imported.
import contextlib
from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError, version


def _include_paths():
    """C++/CUDA include paths for tmol components.

    Defined before other imports because JIT extension loading
    (tmol.utility._cpp_extension) imports this during module init.
    """

    import os.path

    return [os.path.abspath(os.path.dirname(__file__) + "/..")]


from tmol._cpp_lib import has_cuda
from tmol._load_ext import _ensure_compiled_or_jit

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
    "ConstraintSet",
    "EdgeType",
    "FoldForest",
    "LigandFragmentDefinition",
    "MoveMap",
    "PDBInfo",
    "PackedBlockTypes",
    "PackerPalette",
    "PackerTask",
    "ParameterDatabase",
    "PatchedChemicalDatabase",
    "PoseStack",
    "PoseStackBuilder",
    "ScoreFunction",
    "ScoreType",
    "atom_array_from_file",
    "atom_array_from_pose_stack",
    "beta2016_score_function",
    "block_for_residue_label",
    "calculate_fragment_interactions",
    "extended_pose_stack_from_sequences",
    "fast_relax",
    "file_from_atom_array",
    "file_from_pose_stack",
    "get_ligand_mask",
    "get_sequence",
    "get_torsion_names",
    "has_cuda",
    "mutate_residue",
    "mutate_to_sequence",
    "overlay_view",
    "pack_rotamers",
    "pose_stack_from_atom_array",
    "pose_stack_from_file",
    "recombine_fragmented_ligands",
    "run_cart_min",
    "run_kin_min",
    "selection_gallery",
    "switchable_view",
    "view",
]


_LAZY_ATTRS = {}
for _module, _names in (
    ("tmol.database", ("ParameterDatabase",)),
    ("tmol.database._patched_chemdb", ("PatchedChemicalDatabase",)),
    ("tmol.io._canonical_ordering", ("CanonicalOrdering",)),
    (
        "tmol.io._pose_stack_from_biotite",
        (
            "atom_array_from_pose_stack",
            "pose_stack_from_atom_array",
        ),
    ),
    ("tmol.io._pose_stack_from_sequence", ("extended_pose_stack_from_sequences",)),
    (
        "tmol.io._structure_io",
        (
            "atom_array_from_file",
            "file_from_atom_array",
            "file_from_pose_stack",
            "pose_stack_from_file",
        ),
    ),
    (
        "tmol.io._visualize",
        (
            "overlay_view",
            "selection_gallery",
            "switchable_view",
            "view",
        ),
    ),
    ("tmol.kinematics._fold_forest", ("EdgeType", "FoldForest")),
    ("tmol.kinematics._move_map", ("CartesianMoveMap", "MoveMap")),
    (
        "tmol.ligand._fragmentation",
        (
            "LigandFragmentDefinition",
            "recombine_fragmented_ligands",
        ),
    ),
    ("tmol.optimization._minimizers", ("run_cart_min", "run_kin_min")),
    ("tmol.pack._mutate", ("mutate_residue", "mutate_to_sequence")),
    ("tmol.pack._pack_rotamers", ("pack_rotamers",)),
    ("tmol.pack._packer_task", ("PackerPalette", "PackerTask")),
    ("tmol.pose._constraint_set", ("ConstraintSet",)),
    ("tmol.pose._packed_block_types", ("PackedBlockTypes",)),
    ("tmol.pose._pdb_info", ("PDBInfo",)),
    ("tmol.pose._pose_stack", ("PoseStack",)),
    ("tmol.pose._pose_stack_builder", ("PoseStackBuilder",)),
    (
        "tmol.pose._util",
        (
            "block_for_residue_label",
            "get_ligand_mask",
            "get_sequence",
            "get_torsion_names",
        ),
    ),
    ("tmol.relax._fast_relax", ("fast_relax",)),
    ("tmol.score", ("beta2016_score_function",)),
    ("tmol.score._fragment_interactions", ("calculate_fragment_interactions",)),
    ("tmol.score._score_function", ("ScoreFunction",)),
    ("tmol.score._score_types", ("ScoreType",)),
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
