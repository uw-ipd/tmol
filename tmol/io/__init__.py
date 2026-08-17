import torch
from typing import Optional, Union

from tmol.types import validate_args
from tmol.types import Tensor
from tmol.pose import PoseStack
from tmol.io.visualize import (
    pose_stack_to_pdb_string,
    selection_gallery,
    switchable_view,
    view,
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "PoseBuildContext": ("build_context", "PoseBuildContext"),
    "create_pose_stack_from_sequences": (
        "create_pose_stack_from_sequences",
        "create_pose_stack_from_sequences",
    ),
    "CanonicalForm": ("canonical_form", "CanonicalForm"),
    "ordered_set": ("canonical_ordering", "ordered_set"),
    "CysSpecialCaseIndices": ("canonical_ordering", "CysSpecialCaseIndices"),
    "HisSpecialCaseIndices": ("canonical_ordering", "HisSpecialCaseIndices"),
    "CanonicalOrdering": ("canonical_ordering", "CanonicalOrdering"),
    "default_canonical_ordering": ("canonical_ordering", "default_canonical_ordering"),
    "default_packed_block_types": ("canonical_ordering", "default_packed_block_types"),
    "canonical_form_from_pdb": ("canonical_ordering", "canonical_form_from_pdb"),
    "select_atom_records_res_subset": (
        "canonical_ordering",
        "select_atom_records_res_subset",
    ),
    "canonical_form_from_atom_records": (
        "canonical_ordering",
        "canonical_form_from_atom_records",
    ),
    "chain_inds_for_pose_stack": ("chain_deduction", "chain_inds_for_pose_stack"),
    "annotate_pbt_w_valid_connection_masks": (
        "chain_deduction",
        "annotate_pbt_w_valid_connection_masks",
    ),
    "fetch_pdb": ("extern", "fetch_pdb"),
    "to_pdb": ("pdb_parsing", "to_pdb"),
    "to_cdjson": ("generic", "to_cdjson"),
    "pack_cdjson": ("generic", "pack_cdjson"),
    "atom_record_dtype": ("pdb_parsing", "atom_record_dtype"),
    "parse_pdb": ("pdb_parsing", "parse_pdb"),
    "parse_atom_lines": ("pdb_parsing", "parse_atom_lines"),
    "format_atomn": ("pdb_parsing", "format_atomn"),
    "to_pdb_lines": ("pdb_parsing", "to_pdb_lines"),
    "to_atom_lines": ("pdb_parsing", "to_atom_lines"),
    "pose_stack_from_canonical_form": (
        "pose_stack_construction",
        "pose_stack_from_canonical_form",
    ),
    "canonical_form_from_pose_stack": (
        "pose_stack_deconstruction",
        "canonical_form_from_pose_stack",
    ),
    "determine_res_not_connected_from_pose_stack": (
        "pose_stack_deconstruction",
        "determine_res_not_connected_from_pose_stack",
    ),
    "ATOMWORKS_NAME3S": ("pose_stack_from_atomworks", "ATOMWORKS_NAME3S"),
    "ATOMWORKS_ATOM37_NAMES": ("pose_stack_from_atomworks", "ATOMWORKS_ATOM37_NAMES"),
    "pose_stack_from_atomworks": (
        "pose_stack_from_atomworks",
        "pose_stack_from_atomworks",
    ),
    "canonical_form_from_atomworks": (
        "pose_stack_from_atomworks",
        "canonical_form_from_atomworks",
    ),
    "atomworks_from_pose_stack": (
        "pose_stack_from_atomworks",
        "atomworks_from_pose_stack",
    ),
    "canonical_ordering_for_atomworks": (
        "pose_stack_from_atomworks",
        "canonical_ordering_for_atomworks",
    ),
    "packed_block_types_for_atomworks": (
        "pose_stack_from_atomworks",
        "packed_block_types_for_atomworks",
    ),
    "logger": ("pose_stack_from_biotite", "logger"),
    "build_context_from_biotite": (
        "pose_stack_from_biotite",
        "build_context_from_biotite",
    ),
    "pose_stack_from_biotite": ("pose_stack_from_biotite", "pose_stack_from_biotite"),
    "biotite_from_pose_stack": ("pose_stack_from_biotite", "biotite_from_pose_stack"),
    "canonical_form_from_biotite": (
        "pose_stack_from_biotite",
        "canonical_form_from_biotite",
    ),
    "canonical_ordering_for_biotite": (
        "pose_stack_from_biotite",
        "canonical_ordering_for_biotite",
    ),
    "packed_block_types_for_biotite": (
        "pose_stack_from_biotite",
        "packed_block_types_for_biotite",
    ),
    "get_element_from_atom_name": (
        "pose_stack_from_biotite",
        "get_element_from_atom_name",
    ),
    "biotite_from_canonical_form": (
        "pose_stack_from_biotite",
        "biotite_from_canonical_form",
    ),
    "pose_stack_from_openfold": (
        "pose_stack_from_openfold",
        "pose_stack_from_openfold",
    ),
    "canonical_form_from_openfold": (
        "pose_stack_from_openfold",
        "canonical_form_from_openfold",
    ),
    "canonical_ordering_for_openfold": (
        "pose_stack_from_openfold",
        "canonical_ordering_for_openfold",
    ),
    "packed_block_types_for_openfold": (
        "pose_stack_from_openfold",
        "packed_block_types_for_openfold",
    ),
    "pose_stack_from_rosettafold2": (
        "pose_stack_from_rosettafold2",
        "pose_stack_from_rosettafold2",
    ),
    "canonical_form_from_rosettafold2": (
        "pose_stack_from_rosettafold2",
        "canonical_form_from_rosettafold2",
    ),
    "canonical_ordering_for_rosettafold2": (
        "pose_stack_from_rosettafold2",
        "canonical_ordering_for_rosettafold2",
    ),
    "packed_block_types_for_rosettafold2": (
        "pose_stack_from_rosettafold2",
        "packed_block_types_for_rosettafold2",
    ),
    "write_pose_stack_pdb": ("write_pose_stack_pdb", "write_pose_stack_pdb"),
    "atom_records_from_pose_stack": (
        "write_pose_stack_pdb",
        "atom_records_from_pose_stack",
    ),
    "atom_records_from_coords": ("write_pose_stack_pdb", "atom_records_from_coords"),
    "_ATOMWORKS_MAX_PROTEIN_IDX": (
        "pose_stack_from_atomworks",
        "_ATOMWORKS_MAX_PROTEIN_IDX",
    ),
    "_ATOMWORKS_MIN_PROTEIN_IDX": (
        "pose_stack_from_atomworks",
        "_ATOMWORKS_MIN_PROTEIN_IDX",
    ),
    "_paramdb_for_atomworks": ("pose_stack_from_atomworks", "_paramdb_for_atomworks"),
    "_paramdb_for_openfold": ("pose_stack_from_openfold", "_paramdb_for_openfold"),
    "_paramdb_for_rosettafold2": (
        "pose_stack_from_rosettafold2",
        "_paramdb_for_rosettafold2",
    ),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "pose_stack_from_pdb",
    "pose_stack_to_pdb_string",
    "selection_gallery",
    "switchable_view",
    "view",
]


@validate_args
def pose_stack_from_pdb(
    pdb_lines_or_fname: Union[str, list],
    device: torch.device,
    *,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    **kwargs,
) -> PoseStack:
    """Construct a PoseStack given the contents of a PDB file or the name of a PDB file,
    using the full set of residue types contained in tmol's chemical.yaml file.

    Optionally, a subset of the residues in the range from residue_start to residue_end-1
    can be requested.
    Any additional keyword arguments will be passed to pose_stack_from_canonical_form
    """
    from tmol.io.canonical_ordering import (
        default_canonical_ordering,
        default_packed_block_types,
        canonical_form_from_pdb,
    )
    from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
    from tmol.utility import resolve_device

    device = resolve_device(device)
    co = default_canonical_ordering()
    pbt = default_packed_block_types(device)
    cf = canonical_form_from_pdb(
        co,
        pdb_lines_or_fname,
        device,
        residue_start=residue_start,
        residue_end=residue_end,
        res_not_connected=res_not_connected,
    )

    return pose_stack_from_canonical_form(co, pbt, *cf, **kwargs)
