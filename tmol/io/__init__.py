"""Structure conversion between external formats and TMol poses."""

import torch
from pathlib import Path
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
from ._pose_stack_from_atom37 import (  # noqa: F401
    atom37_slot_map_for_ordering,
    canonical_form_from_atom37,
    pose_stack_from_atom37,
)
from ._pose_stack_from_atomworks import (  # noqa: F401
    ATOMWORKS_NAME3S,
    ATOMWORKS_ATOM37_NAMES,
    pose_stack_from_canonical_aa_atom37,
    pose_stack_from_atom37_and_topology,
    canonical_form_from_atomworks,
    atomworks_from_pose_stack,
    canonical_ordering_for_atomworks,
    packed_block_types_for_atomworks,
    _ATOMWORKS_MAX_PROTEIN_IDX,
    _ATOMWORKS_MIN_PROTEIN_IDX,
    _paramdb_for_atomworks,
)
from tmol.chemical import get_element_from_atom_name  # noqa: F401
from tmol.io._cif import (  # noqa: F401
    atom_array_from_cif,
    atom_array_from_file,
    component_chemistry_from_cif,
    pose_stack_from_cif,
    pose_stack_from_file,
)
from ._pose_stack_from_biotite import (  # noqa: F401
    Atom37MappingError,
    PreparedAtom37PoseBuilder,
    build_context_from_biotite,
    pose_stack_from_biotite,
    biotite_from_pose_stack,
    canonical_form_from_biotite,
    canonical_ordering_for_biotite,
    packed_block_types_for_biotite,
    pose_stack_from_canonical_form_and_context,
    prepare_atom37_pose_builder,
    biotite_from_canonical_form,
)
from ._assemble import (  # noqa: F401
    assemble_input,
    atom_array_from_mol2,
    cif_from_atom_array,
)
from ._write_pose_stack_pdb import (  # noqa: F401
    write_pose_stack_pdb,
    atom_records_from_pose_stack,
    atom_records_from_coords,
)
from ._metal_coordination import (  # noqa: F401
    add_metal_coordination,
    remove_metal_coordination,
)

__all__ = [
    "add_metal_coordination",
    "assemble_input",
    "atom_array_from_file",
    "atom_array_from_mol2",
    "cif_from_atom_array",
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
    "create_pose_stack_from_sequences",
    "default_canonical_ordering",
    "default_packed_block_types",
    "extended_pose_stack_from_sequences",
    "fetch_pdb",
    "packed_block_types_for_atomworks",
    "packed_block_types_for_biotite",
    "atom37_slot_map_for_ordering",
    "canonical_form_from_atom37",
    "pose_stack_from_atom37",
    "pose_stack_from_canonical_aa_atom37",
    "pose_stack_from_file",
    "pose_stack_from_canonical_form_and_context",
    "prepare_atom37_pose_builder",
    "remove_metal_coordination",
    "pose_stack_from_atom37_and_topology",
    "pose_stack_from_biotite",
    "pose_stack_from_cif",
    "atom_array_from_cif",
    "component_chemistry_from_cif",
    "pose_stack_from_pdb",
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
    pdb_lines_or_fname: Union[str, list, Path],
    device: torch.device,
    *,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
    res_not_connected: Optional[Tensor[torch.bool][:, :, 2]] = None,
    **kwargs,
) -> PoseStack | tuple[PoseStack, dict] | tuple[PoseStack, PoseBuildContext]:
    """Read PDB through AtomWorks and the shared annotated-array constructor.

    Accept a path, PDB text or a list of lines. Residue slicing uses a half-open
    index range. Additional keywords follow :func:`pose_stack_from_file`, including
    ``prepare_ligands`` and ``return_context``. Supplied hydrogens are retained
    and hydrogen optimization is disabled unless explicitly requested.
    """
    import io

    source = pdb_lines_or_fname
    if isinstance(source, list):
        source = io.StringIO("\n".join(line.rstrip("\n") for line in source))
    elif isinstance(source, str) and (
        "\n" in source or source.startswith(("ATOM  ", "HETATM", "MODEL ", "HEADER"))
    ):
        source = io.StringIO(source)
    kwargs.setdefault("no_optH", True)
    kwargs.setdefault("trust_hydrogen_names", True)
    kwargs.setdefault("missing_density_distance_threshold", 0.0)
    if res_not_connected is not None:
        kwargs["res_not_connected"] = res_not_connected
    return pose_stack_from_file(
        source,
        device,
        residue_start=residue_start,
        residue_end=residue_end,
        **kwargs,
    )
