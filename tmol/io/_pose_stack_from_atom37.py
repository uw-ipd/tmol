"""One tensor-only path from Atom37 coordinates to a PoseStack.

Canonical and noncanonical poses differ only in how residue identity and the
Atom37 slot layout are *resolved*, not in how a pose is built. Once a
:py:class:`~tmol.io.PoseBuildContext` has resolved the residue types, this
module takes tensors the whole way: it scatters Atom37 coordinates into
canonical atom slots and hands the result to the shared constructor.

Nothing here inspects an ``AtomArray``. Callers that have one should use it to
*derive* these tensors once (see
:py:func:`~tmol.io.prepare_atom37_pose_builder`) and then stay on this path.
"""

from typing import Mapping, Optional, Sequence

import torch

from tmol.io._build_context import PoseBuildContext
from tmol.io._canonical_form import CanonicalForm
from tmol.io._canonical_ordering import CanonicalOrdering
from tmol.pose import PoseStack


def atom37_slot_map_for_ordering(
    canonical_ordering: CanonicalOrdering,
    atom_names_by_slot: Mapping[str, Sequence[str]],
    device: torch.device,
) -> torch.Tensor:
    """Map each residue type's Atom37 slots to canonical atom indices.

    The mapping is a property of the residue-type table and the slot layout,
    not of any structure, so it is built once per (ordering, layout) pair.

    Atoms that a terminus patch adds are deliberately left unmapped. They exist
    only on terminal residue variants, and offering one on an interior residue
    would disqualify that residue's block types; tmol applies termini itself
    from the chain boundaries. Which atoms those are is read from the chemical
    database rather than named here.

    Args:
      canonical_ordering: Residue and atom ordering the ``res_types`` index
        into.
      atom_names_by_slot: Atom name occupying each slot, per residue name3. An
        empty name marks an unused slot.
      device: Device for the returned tensor.

    Returns:
      ``[n_residue_types, n_slots]`` canonical atom index per slot, ``-1``
      where the slot is unused or the atom is not part of that residue type.

    Examples:
      >>> slot_map = atom37_slot_map_for_ordering(co, layout, torch.device("cpu"))
      >>> slot_map.shape[1]
      37
    """
    n_slots = max((len(names) for names in atom_names_by_slot.values()), default=0)
    slot_map = torch.full(
        (canonical_ordering.n_restype_io_equiv_classes, n_slots),
        -1,
        dtype=torch.int64,
    )
    terminus_atoms = {
        atom
        for atoms in canonical_ordering.termini_patch_added_atoms.values()
        for atom in atoms
    }
    for restype_index, name3 in enumerate(canonical_ordering.restype_io_equiv_classes):
        names = atom_names_by_slot.get(name3)
        if names is None:
            continue
        atom_index = canonical_ordering.restypes_atom_index_mapping[name3]
        for slot, atom_name in enumerate(names):
            if not atom_name or atom_name in terminus_atoms:
                continue
            canonical_atom = atom_index.get(atom_name)
            if canonical_atom is not None:
                slot_map[restype_index, slot] = canonical_atom
    return slot_map.to(device=device)


def _default_slot_map(
    canonical_ordering: CanonicalOrdering, device: torch.device
) -> torch.Tensor:
    """Build the slot map for the standard Atom37 layout."""
    # Imported here because the module holding the layout table imports this
    # one for its own scatter.
    from tmol.io._pose_stack_from_atomworks import ATOMWORKS_ATOM37_NAMES

    return atom37_slot_map_for_ordering(
        canonical_ordering, ATOMWORKS_ATOM37_NAMES, device
    )


def canonical_form_from_atom37(
    atom37_coords: torch.Tensor,
    res_types: torch.Tensor,
    chain_id: torch.Tensor,
    canonical_ordering: CanonicalOrdering,
    *,
    slot_map: Optional[torch.Tensor] = None,
    disulfides: Optional[torch.Tensor] = None,
    cyclic_bonds: Optional[torch.Tensor] = None,
    covalent_bonds: Optional[torch.Tensor] = None,
    res_not_connected: Optional[torch.Tensor] = None,
) -> CanonicalForm:
    """Scatter Atom37 coordinates into a canonical form.

    Args:
      atom37_coords: ``[n_poses, n_tokens, n_slots, 3]``, autograd-tracked.
      res_types: ``[n_poses, n_tokens]`` index into ``canonical_ordering``;
        ``-1`` marks padding.
      chain_id: ``[n_poses, n_tokens]`` chain index per token.
      canonical_ordering: Ordering the ``res_types`` index into.
      slot_map: ``[n_residue_types, n_slots]`` from
        :py:func:`atom37_slot_map_for_ordering`. Defaults to the standard
        Atom37 layout, which covers the canonical amino acids; supply one built
        from an extended layout for other chemistry.
      disulfides: ``[n, 3]`` explicit ``(pose, res1, res2)`` rows.
      cyclic_bonds: ``[n, 3]`` explicit head-to-tail closures.
      covalent_bonds: ``[n, 5]`` explicit
        ``(pose, res1, atom1, res2, atom2)`` links in canonical ordering.
      res_not_connected: ``[n_poses, n_tokens, 2]`` suppressed polymer
        connections.

    Returns:
      A canonical form whose coordinates carry gradients back to
      ``atom37_coords``.
    """
    device = atom37_coords.device
    if slot_map is None:
        slot_map = _default_slot_map(canonical_ordering, device)
    slot_map = slot_map.to(device=device)

    n_poses, n_tokens, n_slots = atom37_coords.shape[:3]
    if slot_map.shape[1] < n_slots:
        raise ValueError(
            f"slot map covers {slot_map.shape[1]} slots but coordinates supply "
            f"{n_slots}"
        )

    # Padding carries no residue type, so give it a valid row to gather and
    # mask it out afterwards rather than indexing with -1.
    is_real_residue = res_types >= 0
    lookup = torch.where(is_real_residue, res_types, torch.zeros_like(res_types))
    canonical_atom = slot_map[lookup.to(torch.int64)][:, :, :n_slots]
    routed = (canonical_atom >= 0) & is_real_residue.unsqueeze(-1)

    pose_index = (
        torch.arange(n_poses, dtype=torch.int64, device=device)
        .reshape(-1, 1, 1)
        .expand(n_poses, n_tokens, n_slots)
    )
    token_index = (
        torch.arange(n_tokens, dtype=torch.int64, device=device)
        .reshape(1, -1, 1)
        .expand(n_poses, n_tokens, n_slots)
    )

    coords = torch.full(
        (n_poses, n_tokens, canonical_ordering.max_n_canonical_atoms, 3),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    coords[pose_index[routed], token_index[routed], canonical_atom[routed]] = (
        atom37_coords[routed]
    )

    return CanonicalForm(
        chain_id=chain_id.to(torch.int32),
        res_types=res_types.to(torch.int32),
        coords=coords,
        res_labels=None,
        residue_insertion_codes=None,
        chain_labels=None,
        atom_occupancy=None,
        atom_b_factor=None,
        disulfides=disulfides,
        res_not_connected=res_not_connected,
        cyclic_bonds=cyclic_bonds,
        covalent_bonds=covalent_bonds,
    )


def pose_stack_from_atom37(
    atom37_coords: torch.Tensor,
    res_types: torch.Tensor,
    chain_id: torch.Tensor,
    context: PoseBuildContext,
    *,
    slot_map: Optional[torch.Tensor] = None,
    disulfides: Optional[torch.Tensor] = None,
    cyclic_bonds: Optional[torch.Tensor] = None,
    covalent_bonds: Optional[torch.Tensor] = None,
    res_not_connected: Optional[torch.Tensor] = None,
    no_optH: bool = True,
    **kwargs: object,
) -> PoseStack:
    """Build a differentiable PoseStack from Atom37 tensors alone.

    One path for canonical and noncanonical poses. Residue identity comes from
    ``res_types`` indexed into the context's canonical ordering, and any
    covalent chemistry the polymer connections do not already describe comes
    from ``covalent_bonds``; no ``AtomArray`` is consulted.

    Args:
      atom37_coords: ``[n_poses, n_tokens, n_slots, 3]``, autograd-tracked.
      res_types: ``[n_poses, n_tokens]`` index into
        ``context.canonical_ordering``; ``-1`` marks padding.
      chain_id: ``[n_poses, n_tokens]`` chain index per token.
      context: Chemistry resolved once, from any of the
        ``build_context_from_*`` helpers.
      slot_map: Atom37 slot layout; see
        :py:func:`atom37_slot_map_for_ordering`.
      disulfides: Explicit ``(pose, res1, res2)`` rows.
      cyclic_bonds: Explicit head-to-tail closures.
      covalent_bonds: Explicit ``(pose, res1, atom1, res2, atom2)`` links.
      res_not_connected: Suppressed polymer connections.
      no_optH: Preserve finite input hydrogens rather than optimizing them.

    Returns:
      A pose whose coordinates carry gradients back to ``atom37_coords``.

    Examples:
      >>> context = build_context_from_biotite(array, device)
      >>> pose = pose_stack_from_atom37(coords, res_types, chain_id, context)
      >>> pose.coords.requires_grad
      True
    """
    # Imported here because the module defining the shared constructor is
    # loaded after this one.
    from tmol.io._pose_stack_from_biotite import (
        pose_stack_from_canonical_form_and_context,
    )

    canonical_form = canonical_form_from_atom37(
        atom37_coords,
        res_types,
        chain_id,
        context.canonical_ordering,
        slot_map=slot_map,
        disulfides=disulfides,
        cyclic_bonds=cyclic_bonds,
        covalent_bonds=covalent_bonds,
        res_not_connected=res_not_connected,
    )
    return pose_stack_from_canonical_form_and_context(
        canonical_form,
        context,
        no_optH=no_optH,
        atom37_coords=atom37_coords,
        **kwargs,
    )
