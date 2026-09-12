"""Build a PoseStack from backbone-only N/CA/C/O coordinates.

tmol builds missing leaf atoms but rejects blocks missing non-leaf atoms, so
the other adapters cannot take a bare backbone. This one completes the absent
side chains with the packer. Chemistry is shared with the atomworks adapter.
"""

from typing import Literal

import toolz
import torch

from tmol.chemical import one2three
from tmol.io import CanonicalForm
from tmol.io._build_context import PoseBuildContext
from tmol.io._pose_stack_from_atomworks import (
    ATOMWORKS_NAME3S,
    _get_aw_2_tmol_mappings,
    _paramdb_for_atomworks,
    _restype_set_for_atomworks,
    canonical_ordering_for_atomworks,
    packed_block_types_for_atomworks,
)
from tmol.pose import PoseStack

# AlphaFold2/mosaic ordering; alphabetical by three-letter code and so equal to
# ATOMWORKS_NAME3S[1:21].
_DEFAULT_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# N, CA, C, O in the atom37 layout. The same four slots for all 20 amino acids,
# which is what lets a bare (L, 4, 3) tensor reuse the atomworks tables.
_BACKBONE_ATOM37_SLOTS = (0, 1, 2, 4)


def pose_stack_from_backbone_coords(
    coords: torch.Tensor,
    res_types: torch.Tensor,
    chain_id: torch.Tensor,
    device: torch.device,
    *,
    aa_order: str = _DEFAULT_AA_ORDER,
    sidechain_completion: Literal["pack", "none"] = "pack",
    no_optH: bool = False,
    **kwargs,
) -> PoseStack:
    """Build a PoseStack from backbone N/CA/C/O coordinates, with no file I/O.

    The input has no side-chain heavy atoms, so ``sidechain_completion`` says
    how to supply them. ``"pack"`` runs build_missing_sidechains -- a full
    score-function-driven Dunbrack/OptH job, far more expensive than the
    conversion, whose output is not a differentiable function of the input.
    ``"none"`` converts only, leaving absent side chains NaN and the pose
    unscorable. Either way the supplied backbone is returned unchanged and
    stays on the autograd tape.

    Args:
        coords: ``(max_n_res, 4, 3)`` or ``(n_poses, max_n_res, 4, 3)`` in
            N, CA, C, O order. Non-finite entries mark absent atoms. Cast to
            float32, as CanonicalForm requires.
        res_types: indices into ``aa_order``; ``-1`` marks padding.
        chain_id: chain identifiers, shaped like ``res_types``. Residues in a
            chain must be consecutive.
        device: device for the returned PoseStack.
        aa_order: one-letter codes defining the ``res_types`` mapping.
        no_optH: leave rebuilt hydrogens at ideal positions when packing.
        kwargs: passed through to pose_stack_from_canonical_form.

    Example::

        ps = tmol.pose_stack_from_backbone_coords(
            coords, res_types, chain_id, torch.device("cuda")
        )
        sfxn = tmol.beta2016_score_function(ps.packed_block_types.device)
        energy = sfxn.render_whole_pose_scoring_module(ps)(ps.coords).sum()
    """
    from tmol.io import (
        canonical_form_from_pose_stack,
        pose_stack_from_canonical_form,
    )

    if sidechain_completion not in ("pack", "none"):
        raise ValueError(
            f"sidechain_completion must be 'pack' or 'none'; "
            f"got {sidechain_completion!r}"
        )

    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
        res_types = res_types.unsqueeze(0)
        chain_id = chain_id.unsqueeze(0)

    cf = canonical_form_from_backbone_coords(
        coords.to(device), res_types.to(device), chain_id.to(device), aa_order
    )
    context = _build_context_for_backbone_coords(device)
    co = context.canonical_ordering
    pbt = context.packed_block_types

    packing = sidechain_completion == "pack"
    wants_missing_mask = bool(kwargs.pop("return_block_has_missing_atoms", False))
    wants_atom_mapping = bool(kwargs.pop("return_atom_mapping", False))

    pose_stack, opt_return_vals = pose_stack_from_canonical_form(
        co,
        pbt,
        *cf,
        return_block_has_missing_atoms=True,
        return_atom_mapping=wants_atom_mapping or packing,
        **kwargs,
    )
    block_has_missing_atoms = opt_return_vals.pop("block_has_missing_atoms")
    has_missing_atoms = block_has_missing_atoms is not None and bool(
        torch.any(block_has_missing_atoms)
    )

    if packing and block_has_missing_atoms is not None:
        from tmol.io._pose_stack_from_biotite import _restore_canonical_input_coords
        from tmol.pack import build_missing_sidechains

        if has_missing_atoms or not no_optH:
            pose_stack = build_missing_sidechains(
                pose_stack,
                (
                    context._packing_score_function
                    if has_missing_atoms
                    else context._opth_score_function
                ),
                context._dunbrack_sampler,
                block_has_missing_atoms,
                no_optH=no_optH,
                has_missing_atoms=has_missing_atoms,
            )

            if has_missing_atoms:
                # HA is built from CA/N/CB, and leaf building runs before
                # packing, so a bare backbone leaves it unplaced. Rebuild now
                # that the heavy atoms exist. No missing-atom flag here: a gap
                # that survives packing should raise, not come back NaN.
                pose_stack, opt_return_vals = pose_stack_from_canonical_form(
                    co,
                    pbt,
                    *canonical_form_from_pose_stack(co, pose_stack),
                    return_atom_mapping=True,
                    **kwargs,
                )

            # Packing works on values, not the autograd graph; route the input
            # atoms back to the canonical tensor.
            pose_stack = _restore_canonical_input_coords(
                pose_stack,
                cf.coords,
                opt_return_vals["can_atom_mapping"],
                opt_return_vals["ps_atom_mapping"],
            )

    if wants_missing_mask:
        opt_return_vals["block_has_missing_atoms"] = block_has_missing_atoms
    if not wants_atom_mapping:
        opt_return_vals.pop("can_atom_mapping", None)
        opt_return_vals.pop("ps_atom_mapping", None)

    if opt_return_vals:
        return pose_stack, opt_return_vals
    return pose_stack


def canonical_form_from_backbone_coords(
    coords: torch.Tensor,
    res_types: torch.Tensor,
    chain_id: torch.Tensor,
    aa_order: str = _DEFAULT_AA_ORDER,
) -> CanonicalForm:
    """Build a CanonicalForm from backbone N/CA/C/O coordinates.

    Every non-backbone atom is NaN, so the result needs
    ``return_block_has_missing_atoms=True`` to reach a PoseStack. Its residue
    type indices refer to canonical_ordering_for_atomworks, which is what to
    rebuild with after a round-trip through disk.
    """
    assert coords.dim() == 4, "coords must be 4D (n_poses, max_n_res, 4, 3)"
    assert coords.shape[2:] == (4, 3), "atom dimension must be 4: N, CA, C, O"
    assert res_types.shape == coords.shape[:2], "res_types must be (n_poses, max_n_res)"
    assert chain_id.shape == coords.shape[:2], "chain_id must be (n_poses, max_n_res)"
    assert coords.device == res_types.device
    assert coords.device == chain_id.device

    device = coords.device
    n_poses, max_n_res = coords.shape[:2]

    co = canonical_ordering_for_atomworks()
    aw2t_rtmap, aw2t_atmap, aw_at_is_real = _get_aw_2_tmol_mappings(device)

    # Slice the atom tables down to the backbone up front, so nothing of size
    # (n_poses, max_n_res, 37) is materialized.
    slots = torch.tensor(_BACKBONE_ATOM37_SLOTS, dtype=torch.int64, device=device)
    bb_atmap = aw2t_atmap[:, slots]
    bb_is_real = aw_at_is_real[:, slots]

    res_types = res_types.to(torch.int64)
    padding = res_types < 0
    tokens = _atomworks_tokens_for_aa_order(aa_order, device)
    if bool(torch.any(res_types[~padding] >= len(tokens))):
        bad = res_types[~padding & (res_types >= len(tokens))].unique()
        raise ValueError(
            f"res_types must be in range [0, {len(tokens) - 1}] or -1 for "
            f"padding; got out-of-range values: {bad.tolist()}"
        )
    # Padding becomes the atomworks "<M>" token, which the mappings already
    # send to restype -1 with no real atoms.
    aw_tokens = torch.where(padding, 0, tokens[res_types.clamp(min=0)])

    tmol_restypes = aw2t_rtmap[aw_tokens]
    atom_mapping = bb_atmap[aw_tokens]
    at_is_real = bb_is_real[aw_tokens]

    n_bb_ats = len(_BACKBONE_ATOM37_SLOTS)
    pose_ind = (
        torch.arange(n_poses, dtype=torch.int64, device=device)
        .reshape(-1, 1, 1)
        .expand(n_poses, max_n_res, n_bb_ats)
    )
    res_ind = (
        torch.arange(max_n_res, dtype=torch.int64, device=device)
        .reshape(1, -1, 1)
        .expand(n_poses, max_n_res, n_bb_ats)
    )

    tmol_coords = torch.full(
        (n_poses, max_n_res, co.max_n_canonical_atoms, 3),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    tmol_coords[
        pose_ind[at_is_real],
        res_ind[at_is_real],
        atom_mapping[at_is_real],
    ] = coords.to(torch.float32)[at_is_real]

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
def _atomworks_tokens_for_aa_order(aa_order: str, device: torch.device):
    """Map positions in aa_order onto atomworks protein token indices."""
    if len(set(aa_order)) != len(aa_order):
        raise ValueError(f"aa_order must not repeat a one-letter code: {aa_order!r}")
    tokens = []
    for aa in aa_order:
        name3 = one2three(aa)
        if name3 not in ATOMWORKS_NAME3S:
            raise ValueError(f"aa_order entry {aa!r} ({name3}) is not a canonical AA")
        tokens.append(ATOMWORKS_NAME3S.index(name3))
    return torch.tensor(tokens, dtype=torch.int64, device=device)


@toolz.functoolz.memoize
def _build_context_for_backbone_coords(device: torch.device) -> PoseBuildContext:
    """Build context for the canonical amino acids, shared across calls."""
    return PoseBuildContext(
        canonical_ordering=canonical_ordering_for_atomworks(),
        packed_block_types=packed_block_types_for_atomworks(device),
        parameter_database=_paramdb_for_atomworks(),
        restype_set=_restype_set_for_atomworks(),
    )
