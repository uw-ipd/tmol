from .disulfide_search import find_disulfides, find_disulf_numba  # noqa: F401
from .build_missing_leaf_atoms import (  # noqa: F401
    build_missing_leaf_atoms,
    BlockTypeLeafAtomsAnnotation,
    BlockTypeLeafAtomICoorAnnotation,
    PackedBlockTypesLeafAtomICoorAnnotation,
    BlockTypeHCompletionAnnotation,
    PackedBlockTypesHCompletionAnnotation,
)
from .select_from_canonical import (  # noqa: F401
    logger,
    assign_block_types,
    determine_chain_ending_status,
    select_best_block_type_candidate,
    take_block_type_atoms_from_canonical,
    CanonicalOrderingAnnotation,
)
from .his_taut_resolution import (  # noqa: F401
    his_taut_variant_NE2_protonated,
    his_taut_variant_ND1_protonated,
    his_taut_variant_both_ND1_and_NE2_protonated,
    HisTautomerResolution,
    resolve_his_tautomerization,
)
from .left_justify_canonical_form import left_justify_canonical_form  # noqa: F401

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "_annotate_packed_block_types_atom_is_leaf_atom": (
        "build_missing_leaf_atoms",
        "_annotate_packed_block_types_atom_is_leaf_atom",
    ),
    "_annotate_packed_block_types_w_canonical_res_order": (
        "select_from_canonical",
        "_annotate_packed_block_types_w_canonical_res_order",
    ),
    "CanonicalOrderingAnnotation": (
        "select_from_canonical",
        "CanonicalOrderingAnnotation",
    ),
    "HisTautomerResolution": ("his_taut_resolution", "HisTautomerResolution"),
    "assign_block_types": ("select_from_canonical", "assign_block_types"),
    "build_missing_leaf_atoms": (
        "build_missing_leaf_atoms",
        "build_missing_leaf_atoms",
    ),
    "find_disulfides": ("disulfide_search", "find_disulfides"),
    "his_taut_variant_ND1_protonated": (
        "his_taut_resolution",
        "his_taut_variant_ND1_protonated",
    ),
    "his_taut_variant_NE2_protonated": (
        "his_taut_resolution",
        "his_taut_variant_NE2_protonated",
    ),
    "left_justify_canonical_form": (
        "left_justify_canonical_form",
        "left_justify_canonical_form",
    ),
    "resolve_his_tautomerization": (
        "his_taut_resolution",
        "resolve_his_tautomerization",
    ),
    "take_block_type_atoms_from_canonical": (
        "select_from_canonical",
        "take_block_type_atoms_from_canonical",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
