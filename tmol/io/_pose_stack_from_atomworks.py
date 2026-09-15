import torch
import numpy
import toolz
import biotite.structure

from tmol.types import validate_args
from tmol.chemical import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.io._build_context import PoseBuildContext
from tmol.io import (
    CanonicalForm,
    CanonicalOrdering,
)
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)

# Use AtomWorks' protein token order and atom slots as the source of truth.
# This module therefore has a runtime AtomWorks dependency. Copy the public
# lists so callers cannot mutate AtomWorks' shared encoding.
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING

_ATOMWORKS_MIN_PROTEIN_IDX = UNIFIED_ATOM37_ENCODING.token_to_idx["ALA"]
_ATOMWORKS_MAX_PROTEIN_IDX = UNIFIED_ATOM37_ENCODING.token_to_idx["VAL"]
ATOMWORKS_NAME3S = UNIFIED_ATOM37_ENCODING.tokens[
    : UNIFIED_ATOM37_ENCODING.token_to_idx["UNK"] + 1
].tolist()
ATOMWORKS_ATOM37_NAMES = {
    name: UNIFIED_ATOM37_ENCODING.token_atoms[name].tolist()
    for name in ATOMWORKS_NAME3S
}


# ---------------------------------------------------------------------------
# Forward: atomworks tensors -> PoseStack
# ---------------------------------------------------------------------------


@validate_args
def pose_stack_from_canonical_aa_atom37(
    coords: torch.Tensor,
    residue_type: torch.Tensor,
    chain_iid: torch.Tensor,
    **kwargs,
) -> PoseStack:
    """Build a PoseStack from atomworks UNIFIED_ATOM37_ENCODING tensors.

    This is the **tensor-only** Atom37 entry point. Residue identity comes from
    ``residue_type`` alone, so nothing but tensors is required -- but for the
    same reason it is limited to the canonical amino acids with the canonical
    n- and c-termini patches.

    For anything else -- noncanonical residues, ligands, nucleic acids, or
    covalent links -- residue identity cannot be read off a 1..20 token, and
    :py:func:`~tmol.io.pose_stack_from_atom37_and_topology` is the entry point;
    it takes the chemistry from a supplied topology instead.

    Both build a :py:class:`~tmol.io.CanonicalForm` and finish in the same
    constructor, so the choice here is only about how residue identity is
    supplied, not about how the pose is built.

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
    from tmol.io import pose_stack_from_canonical_form

    cf = canonical_form_from_atomworks(coords, residue_type, chain_iid)

    co = canonical_ordering_for_atomworks()
    pbt = packed_block_types_for_atomworks(cf.coords.device)

    return pose_stack_from_canonical_form(co, pbt, *cf, **kwargs)


@validate_args
def pose_stack_from_atom37_and_topology(
    atom37_coords: torch.Tensor,
    biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack,
    context: PoseBuildContext,
    no_optH: bool = True,
    **kwargs,
) -> PoseStack | tuple[PoseStack, dict] | tuple[PoseStack, PoseBuildContext]:
    """Build a differentiable PoseStack from atom37 coordinates and a topology.

    This is the **topology-supplied** Atom37 entry point. Unlike
    :func:`pose_stack_from_canonical_aa_atom37`, which reads residue identity
    from a 1..20 token tensor and so covers only canonical amino acids, this
    supports any chemistry shared by AtomWorks and the supplied TMol context
    (including ordinary ligands and nucleic acids): the *chemical topology* is
    taken from ``biotite_structure`` while mapped coordinates come from the
    autograd-tracked ``atom37_coords`` tensor.

    The topology is only read to derive residue identity and the Atom37 slot
    map. Once derived, construction is pure tensor work in the shared
    :py:func:`~tmol.io.pose_stack_from_canonical_form_and_context`. When the
    same topology is scored repeatedly, use
    :py:func:`~tmol.io.prepare_atom37_pose_builder` to pay that derivation once;
    the returned builder never touches the ``AtomArray`` again. Unmapped finite reference atoms
    remain context. This is the entry point for differentiable scoring/guidance
    over atomized inputs, where the same fixed topology is scored repeatedly as
    coordinates move.

    Build ``context`` once with :func:`build_context_from_biotite` (with
    ``prepare_ligands=True`` when ligands are present). For repeated diffusion
    or search steps, bind the topology once with
    :func:`prepare_atom37_pose_builder` and call the returned builder with
    each coordinate batch. Mapped reference coordinates are ignored, so one
    reference topology can be reused as those coordinates change. It must carry
    two integer annotations that map each atom into the atom37 tensor:
    ``token_id`` (the token axis) and ``atom37_slot`` (the 0..36 slot).

    Topology is derived from chemical identity alone -- ``missing_density`` breaks
    and automatic disulfide detection (both coordinate-dependent) are disabled --
    so the block types, termini, and atom count stay fixed as coordinates change.
    This adapter does not classify or allowlist residue types or elements: newly
    supported PTMs, ions, and metals work through the same API once they are
    represented by the supplied context and canonical ordering.

    Parameters
    ----------
    atom37_coords : Tensor, shape [n_poses, n_tokens, 37, 3]
        Autograd-tracked coordinates in the atomworks atom37 layout.
    biotite_structure : biotite AtomArray
        Reference topology carrying ``token_id`` and ``atom37_slot`` annotations.
    context : PoseBuildContext
        Structure-independent context from :func:`build_context_from_biotite`.
    no_optH : bool
        Preserve finite input hydrogens and leave newly built hydrogens at ideal
        positions when True (default). Pass False to run TMol's hydrogen
        optimization pipeline.
    **kwargs
        Additional arguments forwarded to ``pose_stack_from_biotite``.

    Returns
    -------
    PoseStack
        Whose ``coords`` carry gradients back to ``atom37_coords``.
    """
    from tmol.io import pose_stack_from_biotite

    return pose_stack_from_biotite(
        biotite_structure,
        atom37_coords.device,
        context=context,
        missing_density_distance_threshold=0.0,
        atom37_coords=atom37_coords,
        no_optH=no_optH,
        **kwargs,
    )


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
    from tmol.io import canonical_form_from_pose_stack

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
# Shared protein chemistry and device mappings
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
