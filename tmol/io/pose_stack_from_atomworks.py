import torch
import numpy
import toolz

from tmol.types.functional import validate_args
from tmol.chemical.restypes import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.io.canonical_form import CanonicalForm
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

# ---------------------------------------------------------------------------
# Atomworks UNIFIED_ATOM37_ENCODING constants (protein subset).
# Mirrored verbatim from atomworks so that tmol has no runtime dependency on
# the atomworks package.
#
# Index 0  : <M>  (mask token – all-empty atoms)
# Index 1-20: standard amino acids
# Index 21 : UNK  (unknown amino acid – all-empty atoms)
# ---------------------------------------------------------------------------

# fmt: off
ATOMWORKS_NAME3S = [
    "<M>",                                          # 0: mask
    "ALA", "ARG", "ASN", "ASP", "CYS",             # 1-5
    "GLN", "GLU", "GLY", "HIS", "ILE",             # 6-10
    "LEU", "LYS", "MET", "PHE", "PRO",             # 11-15
    "SER", "THR", "TRP", "TYR", "VAL",             # 16-20
    "UNK",                                          # 21
]

# Per-token atom names in the 37-slot layout.
# Each value is a list of exactly 37 stripped atom-name strings;
# "" means no atom occupies that slot.
ATOMWORKS_ATOM37_NAMES = {
    #                0     1     2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20    21    22    23    24    25    26    27    28    29    30    31    32    33    34    35    36
    "<M>": [       "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   ""],
    "ALA": [      "N", "CA",  "C", "CB",  "O",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "ARG": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "", "CD",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "", "NE",   "",   "",   "",   "",   "","NH1","NH2",   "", "CZ",   "",   "",   "","OXT"],
    "ASN": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "",   "",   "",   "",   "","ND2","OD1",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "ASP": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OD1","OD2",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "CYS": [      "N", "CA",  "C", "CB",  "O",   "",   "",   "",   "",   "", "SG",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "GLN": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "", "CD",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","NE2","OE1",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "GLU": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "", "CD",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OE1","OE2",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "GLY": [      "N", "CA",  "C",   "",  "O",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "HIS": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "",   "",   "","CD2","ND1",   "",   "",   "",   "",   "","CE1",   "",   "",   "",   "","NE2",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "ILE": [      "N", "CA",  "C", "CB",  "O",   "","CG1","CG2",   "",   "",   "",   "","CD1",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "LEU": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "",   "","CD1","CD2",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "LYS": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "", "CD",   "",   "",   "",   "",   "",   "",   "", "CE",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "", "NZ","OXT"],
    "MET": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "", "SD", "CE",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "PHE": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "",   "","CD1","CD2",   "",   "",   "",   "",   "",   "","CE1","CE2",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "", "CZ",   "",   "",   "","OXT"],
    "PRO": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "", "CD",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "SER": [      "N", "CA",  "C", "CB",  "O",   "",   "",   "", "OG",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "THR": [      "N", "CA",  "C", "CB",  "O",   "",   "","CG2",   "","OG1",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "TRP": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "",   "","CD1","CD2",   "",   "",   "",   "",   "",   "",   "","CE2","CE3",   "","NE1",   "",   "",   "","CH2",   "",   "",   "",   "","CZ2","CZ3",   "","OXT"],
    "TYR": [      "N", "CA",  "C", "CB",  "O", "CG",   "",   "",   "",   "",   "",   "","CD1","CD2",   "",   "",   "",   "",   "",   "","CE1","CE2",   "",   "",   "",   "",   "",   "",   "",   "",   "", "OH", "CZ",   "",   "",   "","OXT"],
    "VAL": [      "N", "CA",  "C", "CB",  "O",   "","CG1","CG2",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "","OXT"],
    "UNK": [       "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   "",   ""],
}

# Protein token index range in the atomworks encoding
_ATOMWORKS_MIN_PROTEIN_IDX = 1
_ATOMWORKS_MAX_PROTEIN_IDX = 20
# fmt: on


# ---------------------------------------------------------------------------
# Forward: atomworks tensors -> PoseStack
# ---------------------------------------------------------------------------


@validate_args
def pose_stack_from_atomworks(
    coords: torch.Tensor,
    residue_type: torch.Tensor,
    chain_iid: torch.Tensor,
    **kwargs,
) -> PoseStack:
    """Build a PoseStack from atomworks UNIFIED_ATOM37_ENCODING tensors.

    This function will build a PoseStack using a limited set of residue types:
    only the canonical amino acids with the canonical n- and c-termini patches.
    It begins by constructing a "canonical form" and then passes that canonical
    form to the pose_stack_from_canonical_form function.

    Parameters
    ----------
    coords : Tensor, shape [batch, n_res, 37, 3]
        Atom coordinates in the atomworks atom37 layout.
    residue_type : Tensor[int64], shape [batch, n_res]
        Atomworks token indices. Must be in 1..20 (standard protein only).
    chain_iid : Tensor[int64], shape [batch, n_res]
        Chain identifiers (integer IDs, not string labels).
    **kwargs
        Additional arguments passed to ``pose_stack_from_canonical_form``.

    Returns
    -------
    PoseStack

    Raises
    ------
    ValueError
        If any ``residue_type`` value is outside 1..20 (protein-only).
    """
    from tmol.io.pose_stack_construction import pose_stack_from_canonical_form

    cf = canonical_form_from_atomworks(coords, residue_type, chain_iid)

    co = canonical_ordering_for_atomworks()
    pbt = packed_block_types_for_atomworks(cf.coords.device)

    return pose_stack_from_canonical_form(co, pbt, *cf, **kwargs)


@validate_args
def canonical_form_from_atomworks(
    coords: torch.Tensor,
    residue_type: torch.Tensor,
    chain_iid: torch.Tensor,
) -> CanonicalForm:
    """Build a CanonicalForm from atomworks UNIFIED_ATOM37_ENCODING tensors.

    Parameters
    ----------
    coords : Tensor, shape [batch, n_res, 37, 3]
        Atom coordinates in the atomworks atom37 layout.
    residue_type : Tensor[int64], shape [batch, n_res]
        Atomworks token indices. Must be in 1..20 (standard protein only).
    chain_iid : Tensor[int64], shape [batch, n_res]
        Chain identifiers.

    Returns
    -------
    CanonicalForm
    """

    # Validate protein-only
    if (residue_type < _ATOMWORKS_MIN_PROTEIN_IDX).any() or (
        residue_type > _ATOMWORKS_MAX_PROTEIN_IDX
    ).any():
        bad = residue_type[
            (residue_type < _ATOMWORKS_MIN_PROTEIN_IDX)
            | (residue_type > _ATOMWORKS_MAX_PROTEIN_IDX)
        ]
        raise ValueError(
            f"residue_type must be in range [{_ATOMWORKS_MIN_PROTEIN_IDX}, "
            f"{_ATOMWORKS_MAX_PROTEIN_IDX}] (protein only). "
            f"Got out-of-range values: {bad.unique().tolist()}"
        )

    assert len(coords.shape) == 4, "coords must be 4D [batch, n_res, 37, 3]"
    assert len(residue_type.shape) == 2, "residue_type must be 2D [batch, n_res]"
    assert len(chain_iid.shape) == 2, "chain_iid must be 2D [batch, n_res]"

    device = coords.device
    n_poses = coords.shape[0]
    max_n_res = coords.shape[1]
    max_n_ats = coords.shape[2]  # 37

    aw_pose_ind_for_atom = (
        torch.arange(n_poses, dtype=torch.int64, device=device)
        .reshape(-1, 1, 1)
        .expand(-1, max_n_res, max_n_ats)
    )
    aw_res_ind_for_atom = (
        torch.arange(max_n_res, dtype=torch.int64, device=device)
        .reshape(1, -1, 1)
        .expand(n_poses, -1, max_n_ats)
    )

    assert device == residue_type.device
    assert device == chain_iid.device

    co = canonical_ordering_for_atomworks()
    aw2t_rtmap, aw2t_atmap, aw_at_is_real_map = _get_aw_2_tmol_mappings(device)

    tmol_restypes = aw2t_rtmap[residue_type]
    atom_mapping = aw2t_atmap[residue_type]
    aw_at_is_real = aw_at_is_real_map[residue_type]

    tmol_coords = torch.full(
        (n_poses, max_n_res, co.max_n_canonical_atoms, 3),
        numpy.nan,
        dtype=torch.float32,
        device=device,
    )
    tmol_coords[
        aw_pose_ind_for_atom[aw_at_is_real],
        aw_res_ind_for_atom[aw_at_is_real],
        atom_mapping[aw_at_is_real],
    ] = coords[aw_at_is_real]

    return CanonicalForm(
        chain_id=chain_iid.to(torch.int32),
        res_types=tmol_restypes.to(torch.int32),
        coords=tmol_coords,
        chain_labels=None,
        res_labels=None,
        residue_insertion_codes=None,
        atom_occupancy=None,
        atom_b_factor=None,
        disulfides=None,
        res_not_connected=None,
    )


# ---------------------------------------------------------------------------
# Reverse: PoseStack -> atomworks tensors
# ---------------------------------------------------------------------------


def atomworks_from_pose_stack(
    pose_stack: PoseStack,
) -> tuple:
    """Convert a PoseStack back to atomworks UNIFIED_ATOM37_ENCODING tensors.

    Parameters
    ----------
    pose_stack : PoseStack
        The PoseStack to convert.  Must contain only standard amino acids.

    Returns
    -------
    coords : Tensor, shape [n_poses, max_n_res, 37, 3]
        Atom coordinates in the atomworks atom37 layout.  Absent atoms are 0.
    residue_type : Tensor[int64], shape [n_poses, max_n_res]
        Atomworks token indices (1..20 for real residues, 0 for padding).
    chain_iid : Tensor[int64], shape [n_poses, max_n_res]
        Chain identifiers.
    """
    from tmol.io.pose_stack_deconstruction import canonical_form_from_pose_stack

    co = canonical_ordering_for_atomworks()
    cf = canonical_form_from_pose_stack(co, pose_stack)

    device = cf.coords.device
    tmol_2_aw_rtmap, tmol_2_aw_atmap, tmol_at_is_real = _get_tmol_2_aw_mappings(device)

    n_poses, max_n_res = cf.res_types.shape
    res_types_i64 = cf.res_types.to(torch.int64)
    is_real_res = res_types_i64 >= 0

    # Map residue types: tmol restype index -> atomworks index
    aw_res_types = torch.zeros((n_poses, max_n_res), dtype=torch.int64, device=device)
    aw_res_types[is_real_res] = tmol_2_aw_rtmap[res_types_i64[is_real_res]]

    # Build per-residue atom mapping using the restype of each position
    # Clamp to 0 for padding positions (they won't be used because of
    # coords_present masking below)
    rt_clamped = res_types_i64.clamp(min=0)
    per_res_at_map = tmol_2_aw_atmap[rt_clamped]  # [n_poses, n_res, max_canon_ats]
    per_res_at_real = tmol_at_is_real[rt_clamped]  # [n_poses, n_res, max_canon_ats]

    # Only scatter where there is a valid mapping AND a non-NaN coord
    coords_present = per_res_at_real & ~torch.isnan(cf.coords[:, :, :, 0])
    # Also mask out padding positions
    coords_present &= is_real_res.unsqueeze(2)

    pose_ind, res_ind, canon_at_ind = torch.nonzero(coords_present, as_tuple=True)
    aw_at_ind = per_res_at_map[pose_ind, res_ind, canon_at_ind]

    aw_coords = torch.zeros(
        (n_poses, max_n_res, 37, 3), dtype=torch.float32, device=device
    )
    aw_coords[pose_ind, res_ind, aw_at_ind] = cf.coords[pose_ind, res_ind, canon_at_ind]

    # Chain IDs
    aw_chain_iid = cf.chain_id.to(torch.int64)

    return aw_coords, aw_res_types, aw_chain_iid


# ---------------------------------------------------------------------------
# Memoized helpers (following the OpenFold / RoseTTAFold2 pattern)
# ---------------------------------------------------------------------------


@toolz.functoolz.memoize
def _paramdb_for_atomworks() -> ParameterDatabase:
    """Construct the ParameterDatabase for the subset of residue types
    that the atomworks atom37 protein encoding covers: the 20 canonical
    amino acids (plus HIS_D and CYD tautomers) and the canonical n-
    and c-termini patches.
    """
    desired_rt_names = sorted(
        [n for n in ATOMWORKS_NAME3S if n not in ("<M>", "UNK")] + ["HIS_D", "CYD"]
    )
    desired_variants_display_names = ["nterm", "cterm"]

    return ParameterDatabase.get_default().create_stable_subset(
        desired_rt_names, desired_variants_display_names
    )


@toolz.functoolz.memoize
def _restype_set_for_atomworks() -> ResidueTypeSet:
    paramdb = _paramdb_for_atomworks()
    return ResidueTypeSet.from_database(paramdb.chemical)


@validate_args
@toolz.functoolz.memoize
def canonical_ordering_for_atomworks() -> CanonicalOrdering:
    """Construct the CanonicalOrdering for the protein subset used
    by the atomworks UNIFIED_ATOM37_ENCODING."""
    paramdb = _paramdb_for_atomworks()
    return CanonicalOrdering.from_chemdb(paramdb.chemical)


@validate_args
@toolz.functoolz.memoize
def packed_block_types_for_atomworks(device: torch.device) -> PackedBlockTypes:
    """Construct the PackedBlockTypes for the protein subset used
    by the atomworks UNIFIED_ATOM37_ENCODING."""
    restype_set = _restype_set_for_atomworks()
    return PackedBlockTypes.from_restype_list(
        restype_set.chem_db, restype_set, restype_set.residue_types, device
    )


@toolz.functoolz.memoize
def _get_aw_2_tmol_mappings(device: torch.device):
    """Build forward mapping tensors: atomworks index -> tmol canonical.

    OXT is deliberately excluded: it only exists on C-terminal residue
    variants in tmol, and its presence on non-terminal residues would
    prevent block-type resolution.  tmol determines termini
    automatically from the chain_iid boundaries.
    """
    co = canonical_ordering_for_atomworks()

    # Strip OXT from the atom-name lists so that it is never mapped
    # into the canonical form (tmol handles termini via patches).
    aw_atom_names_no_oxt = {
        name3: [at if at != "OXT" else "" for at in atoms]
        for name3, atoms in ATOMWORKS_ATOM37_NAMES.items()
    }

    return co.create_src_2_tmol_mappings(ATOMWORKS_NAME3S, aw_atom_names_no_oxt, device)


@toolz.functoolz.memoize
def _get_tmol_2_aw_mappings(device: torch.device):
    """Build reverse mapping tensors: tmol canonical -> atomworks atom37 slot."""
    co = canonical_ordering_for_atomworks()

    n_co_restypes = len(co.restype_io_equiv_classes)
    max_n_canonical_atoms = co.max_n_canonical_atoms

    tmol_2_aw_rtmap = torch.full((n_co_restypes,), 0, dtype=torch.int64)
    tmol_2_aw_atmap = torch.full(
        (n_co_restypes, max_n_canonical_atoms), -1, dtype=torch.int64
    )
    tmol_at_is_real = torch.zeros(
        (n_co_restypes, max_n_canonical_atoms), dtype=torch.bool
    )

    for aw_idx, name3 in enumerate(ATOMWORKS_NAME3S):
        if name3 not in co.restype_io_equiv_classes:
            continue
        tmol_rt_idx = co.restype_io_equiv_classes.index(name3)
        tmol_2_aw_rtmap[tmol_rt_idx] = aw_idx

        aw_atoms = ATOMWORKS_ATOM37_NAMES[name3]
        tmol_atom_mapping = co.restypes_atom_index_mapping[name3]
        for at37_slot, at_name in enumerate(aw_atoms):
            if at_name == "":
                continue
            if at_name in tmol_atom_mapping:
                canon_at_idx = tmol_atom_mapping[at_name]
                tmol_2_aw_atmap[tmol_rt_idx, canon_at_idx] = at37_slot
                tmol_at_is_real[tmol_rt_idx, canon_at_idx] = True

    def _d(x):
        return x.to(device=device)

    return _d(tmol_2_aw_rtmap), _d(tmol_2_aw_atmap), _d(tmol_at_is_real)
