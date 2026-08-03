"""Build a PoseStack directly from backbone coordinates without PDB file I/O.

Analogous to pose_stack_from_openfold / pose_stack_from_rosettafold2 but takes
the (N, CA, C, O) backbone coordinates and integer residue type indices that
protein structure prediction models and optimization loops (e.g. mosaic,
AlphaFold2, RFDiffusion) produce natively.
"""
import torch
import numpy
import toolz

from tmol.types.functional import validate_args
from tmol.chemical.restypes import ResidueTypeSet, one2three
from tmol.database import ParameterDatabase
from tmol.io.canonical_form import CanonicalForm
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


# Default residue ordering used by AlphaFold2, mosaic, and many ML frameworks.
_DEFAULT_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
# Backbone atom names, in the order used by backbone_coordinates in mosaic:
# N=0, CA=1, C=2, O=3
_BACKBONE_ATOM_NAMES = ["N", "CA", "C", "O"]


def pose_stack_from_backbone_coords(
    coords: torch.Tensor,
    res_types: torch.Tensor,
    chain_id: torch.Tensor,
    device: torch.device,
    *,
    aa_order: str = _DEFAULT_AA_ORDER,
    **kwargs,
) -> PoseStack:
    """Build a PoseStack from backbone-only coordinates, with no PDB file I/O.

    Accepts backbone N/CA/C/O coordinates and integer residue type indices,
    converts them in-memory to a canonical form, and passes that to
    pose_stack_from_canonical_form.  Missing side-chain leaf atoms are
    reconstructed from ideal internal-coordinate geometry.

    Args:
        coords: Backbone atom coordinates.  Shape ``(max_n_res, 4, 3)`` for a
            single pose or ``(n_poses, max_n_res, 4, 3)`` for a batch.
            Atom order: N=0, CA=1, C=2, O=3.
        res_types: Integer residue type indices (into ``aa_order``).  Shape
            ``(max_n_res,)`` or ``(n_poses, max_n_res)``.  Use ``-1`` for
            padding positions.
        chain_id: Integer chain identifiers.  Shape ``(max_n_res,)`` or
            ``(n_poses, max_n_res)``, dtype int32.  Residues with different
            values are placed on separate polymer chains.
        device: Target ``torch.device`` for the returned PoseStack.
        aa_order: 20-character string of one-letter codes defining the
            ``res_types`` index mapping.  Defaults to the AlphaFold2/mosaic
            ordering ``"ARNDCQEGHILKMFPSTWYV"``.
        **kwargs: Passed through to ``pose_stack_from_canonical_form``
            (e.g. ``find_additional_disulfides``, ``return_chain_ind``).

    Returns:
        PoseStack with ideal side-chain atoms filled in by tmol.

    Example::

        import torch, numpy as np
        import tmol

        # Single pose from a mosaic StructureModelOutput
        coords_np  = output.backbone_coordinates  # (L, 4, 3) numpy
        seq_np     = output.full_sequence.argmax(-1)  # (L,) int
        asym_np    = output.asym_id               # (L,) int

        dev = torch.device("cpu")
        ps = tmol.pose_stack_from_backbone_coords(
            torch.from_numpy(coords_np).float(),
            torch.from_numpy(seq_np).long(),
            torch.from_numpy(asym_np).int(),
            dev,
        )
    """
    from tmol.io.pose_stack_construction import pose_stack_from_canonical_form

    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
        res_types = res_types.unsqueeze(0)
        chain_id = chain_id.unsqueeze(0)

    cf = canonical_form_from_backbone_coords(coords, res_types, chain_id, aa_order)

    co = canonical_ordering_for_backbone_coords()
    pbt = packed_block_types_for_backbone_coords(device)

    return pose_stack_from_canonical_form(
        co,
        pbt,
        cf.chain_id.to(device),
        cf.res_types.to(device),
        cf.coords.to(device),
        cf.res_labels,
        cf.residue_insertion_codes,
        cf.chain_labels,
        cf.atom_occupancy,
        cf.atom_b_factor,
        cf.disulfides,
        cf.res_not_connected,
        **kwargs,
    )


def canonical_form_from_backbone_coords(
    coords: torch.Tensor,
    res_types: torch.Tensor,
    chain_id: torch.Tensor,
    aa_order: str = _DEFAULT_AA_ORDER,
) -> CanonicalForm:
    """Construct a CanonicalForm from backbone coordinates.

    This is the stable intermediate representation — suitable for serialization
    and later reconstruction via ``canonical_ordering_for_backbone_coords`` /
    ``packed_block_types_for_backbone_coords``.

    Args:
        coords: ``(n_poses, max_n_res, 4, 3)`` backbone coordinates (N/CA/C/O).
        res_types: ``(n_poses, max_n_res)`` int64 residue type indices.
        chain_id: ``(n_poses, max_n_res)`` int32 chain identifiers.
        aa_order: 20-char one-letter code ordering for ``res_types``.

    Returns:
        CanonicalForm with NaN for all non-backbone atoms.
    """
    assert coords.dim() == 4, "coords must be 4-D: (n_poses, max_n_res, 4, 3)"
    assert coords.shape[2] == 4, "atom dimension must be 4 (N, CA, C, O)"

    device = coords.device
    n_poses, max_n_res, n_src_ats, _ = coords.shape

    pose_ind = (
        torch.arange(n_poses, dtype=torch.int64, device=device)
        .reshape(-1, 1, 1)
        .expand(n_poses, max_n_res, n_src_ats)
    )
    res_ind = (
        torch.arange(max_n_res, dtype=torch.int64, device=device)
        .reshape(1, -1, 1)
        .expand(n_poses, max_n_res, n_src_ats)
    )

    co = canonical_ordering_for_backbone_coords()
    rt_map, at_map, at_is_real = _get_backbone_2_tmol_mappings(device, aa_order)

    # -1 entries are padding; clamp to 0 for safe indexing then restore -1 after
    padding_mask = res_types.long() < 0
    src_rt_safe = res_types.long().clamp(min=0)

    tmol_restypes = rt_map[src_rt_safe]
    tmol_restypes[padding_mask] = -1

    atom_mapping = at_map[src_rt_safe]     # (n_poses, max_n_res, 4)
    bb_at_is_real = at_is_real[src_rt_safe]  # (n_poses, max_n_res, 4)

    tmol_coords = torch.full(
        (n_poses, max_n_res, co.max_n_canonical_atoms, 3),
        numpy.nan,
        dtype=torch.float32,
        device=device,
    )

    valid = bb_at_is_real & (~padding_mask).unsqueeze(-1).expand_as(bb_at_is_real)
    tmol_coords[
        pose_ind[valid],
        res_ind[valid],
        atom_mapping[valid],
    ] = coords[valid]

    return CanonicalForm(
        chain_id=chain_id.to(torch.int32),
        res_types=tmol_restypes.to(torch.int32),
        coords=tmol_coords,
        res_labels=None,
        residue_insertion_codes=None,
        chain_labels=None,
        atom_occupancy=None,
        atom_b_factor=None,
        disulfides=None,
        res_not_connected=None,
    )


@toolz.functoolz.memoize
def _paramdb_for_backbone_coords() -> ParameterDatabase:
    desired_rt_names = sorted(
        [one2three(aa) for aa in _DEFAULT_AA_ORDER] + ["HIS_D", "CYD"]
    )
    return ParameterDatabase.get_default().create_stable_subset(
        desired_rt_names, ["nterm", "cterm"]
    )


@toolz.functoolz.memoize
def _restype_set_for_backbone_coords() -> ResidueTypeSet:
    return ResidueTypeSet.from_database(_paramdb_for_backbone_coords().chemical)


@validate_args
@toolz.functoolz.memoize
def canonical_ordering_for_backbone_coords() -> CanonicalOrdering:
    """Stable CanonicalOrdering for the 20 standard amino acids + termini.

    Use this together with ``packed_block_types_for_backbone_coords`` when
    deserializing a CanonicalForm produced by
    ``canonical_form_from_backbone_coords``.
    """
    return CanonicalOrdering.from_chemdb(_paramdb_for_backbone_coords().chemical)


@validate_args
@toolz.functoolz.memoize
def packed_block_types_for_backbone_coords(device: torch.device) -> PackedBlockTypes:
    """PackedBlockTypes for the 20 standard amino acids + termini."""
    rs = _restype_set_for_backbone_coords()
    return PackedBlockTypes.from_restype_list(
        rs.chem_db, rs, rs.residue_types, device
    )


@toolz.functoolz.memoize
def _get_backbone_2_tmol_mappings(device: torch.device, aa_order: str):
    """Return (rt_map, at_map, at_is_real) mapping backbone atoms to tmol indices."""
    co = canonical_ordering_for_backbone_coords()
    name3s = [one2three(aa) for aa in aa_order]
    bb_atoms_for_name3 = {name3: list(_BACKBONE_ATOM_NAMES) for name3 in name3s}
    return co.create_src_2_tmol_mappings(name3s, bb_atoms_for_name3, device)
