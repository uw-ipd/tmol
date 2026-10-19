"""Structure conversion between external formats and TMol poses."""

from tmol.io._visualize import (
    overlay_view,
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
from ._extern import fetch_pdb, fetch_cif  # noqa: F401
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
from tmol.io._cif import component_chemistry_from_cif  # noqa: F401
from ._pose_stack_from_biotite import (  # noqa: F401
    Atom37MappingError,
    PreparedAtom37PoseBuilder,
    build_context_from_atom_array,
    pose_stack_from_atom_array,
    atom_array_from_pose_stack,
    canonical_form_from_biotite,
    canonical_ordering_for_biotite,
    packed_block_types_for_biotite,
    pose_stack_from_canonical_form_and_context,
    prepare_atom37_pose_builder,
    biotite_from_canonical_form,
)
from ._structure_io import (  # noqa: F401
    atom_array_from_file,
    file_from_atom_array,
    file_from_pose_stack,
    pose_stack_from_file,
)
from ._assemble import assemble_input  # noqa: F401
from ._metal_coordination import (  # noqa: F401
    add_metal_coordination,
    remove_metal_coordination,
)

__all__ = [
    "Atom37MappingError",
    "CanonicalForm",
    "CanonicalOrdering",
    "PoseBuildContext",
    "PreparedAtom37PoseBuilder",
    "add_metal_coordination",
    "assemble_input",
    "atom37_slot_map_for_ordering",
    "atom_array_from_file",
    "atom_array_from_pose_stack",
    "atomworks_from_pose_stack",
    "biotite_from_canonical_form",
    "build_context_from_atom_array",
    "canonical_form_from_atom37",
    "canonical_form_from_atomworks",
    "canonical_form_from_biotite",
    "canonical_form_from_pose_stack",
    "canonical_ordering_for_atomworks",
    "canonical_ordering_for_biotite",
    "component_chemistry_from_cif",
    "create_pose_stack_from_sequences",
    "default_canonical_ordering",
    "default_packed_block_types",
    "extended_pose_stack_from_sequences",
    "fetch_cif",
    "fetch_pdb",
    "file_from_atom_array",
    "file_from_pose_stack",
    "packed_block_types_for_atomworks",
    "packed_block_types_for_biotite",
    "pose_stack_from_atom37",
    "pose_stack_from_atom37_and_topology",
    "pose_stack_from_atom_array",
    "pose_stack_from_canonical_aa_atom37",
    "pose_stack_from_canonical_form_and_context",
    "pose_stack_from_file",
    "prepare_atom37_pose_builder",
    "remove_metal_coordination",
    "overlay_view",
    "selection_gallery",
    "switchable_view",
    "view",
]
