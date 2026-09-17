"""One tensor-only path builds canonical and noncanonical poses alike."""

import biotite.structure as struc
import biotite.structure.io
import pytest
import torch

from tmol.io import (
    atom37_slot_map_for_ordering,
    build_context_from_biotite,
    canonical_form_from_biotite,
    pose_stack_from_atom37,
    pose_stack_from_canonical_form_and_context,
)
from tmol.tests.data import data_path


def _load(path) -> struc.AtomArray:
    structure = biotite.structure.io.load_structure(str(path), model=1)
    assert isinstance(structure, struc.AtomArray)
    return structure


def _full_atom_layout(canonical_ordering):
    """Give every residue type a slot per canonical atom, in order.

    A caller with noncanonical chemistry supplies the layout their coordinates
    use; this is the simplest such layout and needs no per-residue knowledge.
    """
    return {
        name3: list(canonical_ordering.restypes_ordered_atom_names[name3])
        for name3 in canonical_ordering.restype_io_equiv_classes
    }


def _scatter_to_slots(coords, res_types, slot_map):
    """Spread canonical-form coordinates back out into slot-major order."""
    n_poses, n_res = res_types.shape
    n_slots = slot_map.shape[1]
    is_real = res_types >= 0
    lookup = torch.where(is_real, res_types, torch.zeros_like(res_types))
    canonical_atom = slot_map[lookup.to(torch.int64)]
    routed = (canonical_atom >= 0) & is_real.unsqueeze(-1)

    slots = torch.full(
        (n_poses, n_res, n_slots, 3),
        float("nan"),
        dtype=torch.float32,
        device=coords.device,
    )
    pose_index = (
        torch.arange(n_poses, device=coords.device)
        .reshape(-1, 1, 1)
        .expand(n_poses, n_res, n_slots)
    )
    res_index = (
        torch.arange(n_res, device=coords.device)
        .reshape(1, -1, 1)
        .expand(n_poses, n_res, n_slots)
    )
    slots[routed] = coords[
        pose_index[routed], res_index[routed], canonical_atom[routed]
    ]
    return slots


@pytest.mark.parametrize(
    "source",
    [
        ("pdb", "1ubq.pdb"),
        ("covalent_fixtures", "lys_biotin_1bdo.cif"),
        ("covalent_fixtures", "nglycan_tree_1ax2.cif"),
    ],
    ids=["canonical_protein", "conjugated_lysine", "glycan_tree"],
)
def test_tensor_path_matches_the_atom_array_path(source, torch_device):
    """Tensors alone reproduce the pose the AtomArray route builds.

    The AtomArray is used only to resolve chemistry and to produce the input
    tensors; pose construction itself sees no structure object.
    """
    structure = _load(data_path(*source))
    context = build_context_from_biotite(structure, torch_device)
    canonical_ordering = context.canonical_ordering

    canonical_form = canonical_form_from_biotite(
        structure,
        torch_device,
        co=canonical_ordering,
        missing_density_distance_threshold=0.0,
    )
    reference = pose_stack_from_canonical_form_and_context(
        canonical_form, context, no_optH=True, atom37_coords=None
    )

    slot_map = atom37_slot_map_for_ordering(
        canonical_ordering, _full_atom_layout(canonical_ordering), torch_device
    )
    slots = _scatter_to_slots(canonical_form.coords, canonical_form.res_types, slot_map)
    slots.requires_grad_(True)

    pose = pose_stack_from_atom37(
        slots,
        canonical_form.res_types,
        canonical_form.chain_id,
        context,
        slot_map=slot_map,
        disulfides=canonical_form.disulfides,
        cyclic_bonds=canonical_form.cyclic_bonds,
        covalent_bonds=canonical_form.covalent_bonds,
        res_not_connected=canonical_form.res_not_connected,
    )

    assert pose.n_poses == reference.n_poses
    assert torch.equal(pose.block_type_ind, reference.block_type_ind)
    assert torch.equal(
        pose.inter_residue_connections, reference.inter_residue_connections
    )
    assert [bt.name for bt in pose.packed_block_types.active_block_types] == [
        bt.name for bt in reference.packed_block_types.active_block_types
    ]
    # Atoms routed from the input slots are copied, and match to within the
    # default tolerance. The exception is an atom neither path routes: a
    # terminus patch's own atoms are rebuilt from internal coordinates on both
    # sides, so they agree only to the arithmetic's last bits -- measured worst
    # case 2.3e-4 A on this structure's MET:nterm H1, inside the 1e-3 A the
    # input coordinates themselves carry.
    torch.testing.assert_close(
        pose.coords[pose.real_atoms],
        reference.coords[reference.real_atoms],
        rtol=0,
        atol=1e-3,
    )

    # The whole point of staying on tensors is that gradients survive.
    pose.coords[pose.real_atoms].sum().backward()
    assert torch.count_nonzero(slots.grad) > 0


def test_slot_map_leaves_terminus_atoms_unmapped(torch_device):
    """Atoms a terminus patch adds must not be routed onto interior residues.

    Which atoms those are comes from the chemical database, so this holds for
    any chemistry the ordering covers rather than for a named list.
    """
    structure = _load(data_path("pdb", "1ubq.pdb"))
    context = build_context_from_biotite(structure, torch_device)
    canonical_ordering = context.canonical_ordering

    layout = _full_atom_layout(canonical_ordering)
    slot_map = atom37_slot_map_for_ordering(canonical_ordering, layout, torch_device)

    by_class = canonical_ordering.termini_only_atoms_by_class
    assert by_class, "expected the database to define terminus-added atoms"

    checked = 0
    for restype_index, name3 in enumerate(canonical_ordering.restype_io_equiv_classes):
        # Per residue type, not pooled: H1, H2 and H3 are terminus-only on an
        # amino acid but ordinary base atoms of DG, DA, DT and water, and a
        # pooled set would demand they go unrouted there too.
        terminus_atoms = by_class.get(name3, ())
        for slot, atom_name in enumerate(layout[name3]):
            if atom_name in terminus_atoms:
                assert int(slot_map[restype_index, slot]) == -1
                checked += 1
    assert checked > 0, "expected at least one terminus atom in the layout"


def test_tensor_path_rejects_more_slots_than_the_map_covers(torch_device):
    """A layout narrower than the coordinates is a caller error, not a crop."""
    structure = _load(data_path("pdb", "1ubq.pdb"))
    context = build_context_from_biotite(structure, torch_device)
    canonical_ordering = context.canonical_ordering
    canonical_form = canonical_form_from_biotite(
        structure, torch_device, co=canonical_ordering
    )

    slot_map = atom37_slot_map_for_ordering(
        canonical_ordering, _full_atom_layout(canonical_ordering), torch_device
    )
    n_res = canonical_form.res_types.shape[1]
    too_many = torch.zeros(
        (1, n_res, slot_map.shape[1] + 1, 3), dtype=torch.float32, device=torch_device
    )

    with pytest.raises(ValueError, match="slot map covers"):
        pose_stack_from_atom37(
            too_many,
            canonical_form.res_types,
            canonical_form.chain_id,
            context,
            slot_map=slot_map,
        )
