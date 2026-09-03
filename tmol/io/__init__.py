"""Structure conversion between external formats and TMol poses."""

import torch
from typing import Optional, Union

from tmol.types import (
    validate_args,
    Tensor,
)
from tmol.pose import PoseStack
from tmol.io._visualize import (
    pose_stack_to_pdb_string,
    selection_gallery,
    switchable_view,
    view,
)

from ._build_context import PoseBuildContext  # noqa: F401
from ._canonical_form import CanonicalForm  # noqa: F401
from ._canonical_ordering import (  # noqa: F401
    ordered_set,
    CysSpecialCaseIndices,
    HisSpecialCaseIndices,
    CanonicalOrdering,
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
    select_atom_records_res_subset,
    canonical_form_from_atom_records,
)
from ._chain_deduction import (  # noqa: F401
    chain_inds_for_pose_stack,
    annotate_pbt_w_valid_connection_masks,
)  # noqa: F401
from ._pose_stack_from_sequence import (  # noqa: F401
    create_pose_stack_from_sequences,
    extended_pose_stack_from_sequences,
    EXTENDED_BACKBONE_TORSIONS,
)
from ._extern import fetch_pdb  # noqa: F401
from ._generic import to_cdjson, pack_cdjson  # noqa: F401
from ._pdb_parsing import (  # noqa: F401
    atom_record_dtype,
    parse_pdb,
    parse_atom_lines,
    format_atomn,
    to_pdb,
    to_pdb_lines,
    to_atom_lines,
)
from ._pose_stack_construction import pose_stack_from_canonical_form  # noqa: F401
from ._pose_stack_deconstruction import (  # noqa: F401
    canonical_form_from_pose_stack,
    determine_res_not_connected_from_pose_stack,
)
from ._pose_stack_from_atomworks import (  # noqa: F401
    ATOMWORKS_NAME3S,
    ATOMWORKS_ATOM37_NAMES,
    pose_stack_from_atomworks,
    pose_stack_from_atom37_and_biotite,
    canonical_form_from_atomworks,
    atomworks_from_pose_stack,
    canonical_ordering_for_atomworks,
    packed_block_types_for_atomworks,
    _ATOMWORKS_MAX_PROTEIN_IDX,
    _ATOMWORKS_MIN_PROTEIN_IDX,
    _paramdb_for_atomworks,
)
from tmol.chemical import get_element_from_atom_name  # noqa: F401
from ._pose_stack_from_biotite import (  # noqa: F401
    Atom37MappingError,
    PreparedAtom37PoseBuilder,
    build_context_from_biotite,
    pose_stack_from_biotite,
    biotite_from_pose_stack,
    canonical_form_from_biotite,
    canonical_ordering_for_biotite,
    packed_block_types_for_biotite,
    prepare_pose_stack_from_atom37,
    biotite_from_canonical_form,
)
from ._pose_stack_from_openfold import (  # noqa: F401
    pose_stack_from_openfold,
    canonical_form_from_openfold,
    canonical_ordering_for_openfold,
    packed_block_types_for_openfold,
    _paramdb_for_openfold,
)
from ._pose_stack_from_rosettafold2 import (  # noqa: F401
    pose_stack_from_rosettafold2,
    canonical_form_from_rosettafold2,
    canonical_ordering_for_rosettafold2,
    packed_block_types_for_rosettafold2,
    _paramdb_for_rosettafold2,
)
from ._write_pose_stack_pdb import (  # noqa: F401
    write_pose_stack_pdb,
    atom_records_from_pose_stack,
    atom_records_from_coords,
)

__all__ = [
    "Atom37MappingError",
    "CanonicalForm",
    "CanonicalOrdering",
    "PoseBuildContext",
    "PreparedAtom37PoseBuilder",
    "atom_records_from_coords",
    "atom_records_from_pose_stack",
    "atomworks_from_pose_stack",
    "biotite_from_canonical_form",
    "biotite_from_pose_stack",
    "build_context_from_biotite",
    "canonical_form_from_atomworks",
    "canonical_form_from_biotite",
    "canonical_form_from_pdb",
    "canonical_form_from_pose_stack",
    "canonical_ordering_for_atomworks",
    "canonical_ordering_for_biotite",
    "canonical_ordering_for_openfold",
    "canonical_ordering_for_rosettafold2",
    "create_pose_stack_from_sequences",
    "default_canonical_ordering",
    "default_packed_block_types",
    "extended_pose_stack_from_sequences",
    "fetch_pdb",
    "packed_block_types_for_atomworks",
    "packed_block_types_for_biotite",
    "packed_block_types_for_openfold",
    "packed_block_types_for_rosettafold2",
    "pose_stack_from_atomworks",
    "prepare_pose_stack_from_atom37",
    "pose_stack_from_atom37_and_biotite",
    "pose_stack_from_biotite",
    "pose_stack_from_openfold",
    "pose_stack_from_pdb",
    "pose_stack_from_rosettafold2",
    "pose_stack_to_pdb_string",
    "selection_gallery",
    "switchable_view",
    "to_atom_lines",
    "to_pdb",
    "to_pdb_lines",
    "view",
    "write_pose_stack_pdb",
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
