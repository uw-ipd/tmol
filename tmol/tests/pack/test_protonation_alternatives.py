"""Opt-in protonation alternatives: the palette, the offset energy and packing."""

import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.io._protonation import PROTONATION_ALTERNATIVES
from tmol.pack import PackerPalette, PackerTask, SetPackerTask, pack_rotamers
from tmol.pack._pack_rotamers import _calculate_packer_energies
from tmol.pack.protonation_alternatives import (
    block_alternatives,
    chosen_protonation_variants,
)
from tmol.pack.rotamer import (
    FixedAAChiSampler,
    IncludeCurrentSampler,
    build_rotamers,
)
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path

HIS_POS_OFFSET = 1.364 * (7.4 - 6.5)
CYS_DEP_OFFSET = 1.364 * (8.3 - 7.4)
DMZ = ("pdb", "6DMZ_A.pdb")
F8B = ("pdb", "bysize_300_res_6f8b.pdb")


def _pose(path, device, alternatives):
    """The structure without its hydrogens, which would state every residue's state."""
    structure = atom_array_from_cif(data_path(*path))
    structure = structure[structure.element != "H"]
    return pose_stack_from_biotite(
        structure, device, protonation_alternatives=alternatives
    )


def _names(task, pose, block):
    types = task.pbt.active_block_types
    n = int(task.per_block_n_considered_block_types[pose, block])
    considered = task.per_block_considered_block_types[pose, block, :n].tolist()
    return [types[j].base_name for j in considered]


def _repacking_task(pose_stack, palette, res_labels):
    """Repack the blocks within 10 A (CA) of the blocks numbered ``res_labels``."""
    db = ParameterDatabase.get_default()
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.add_conformer_sampler(
        create_dunbrack_sampler_from_database(db, pose_stack.device)
    )
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(IncludeCurrentSampler())
    types = pose_stack.packed_block_types.active_block_types
    ca = torch.stack(
        [
            pose_stack.coords[
                0,
                int(pose_stack.block_coord_offset[0, b])
                + types[int(pose_stack.block_type_ind64[0, b])].atom_to_idx["CA"],
            ]
            for b in range(pose_stack.max_n_blocks)
        ]
    )
    labels = torch.as_tensor(pose_stack.pdb_info.residue_labels[0])
    centres = ca[torch.isin(labels, torch.as_tensor(res_labels)).to(ca.device)]
    near = (torch.cdist(ca, centres) < 10.0).any(dim=1)
    task.disable_packing_by_block_mask(~near[None])
    return task


@pytest.mark.parametrize(
    ("path", "flagged", "added", "offset"),
    [
        (DMZ, {("A", 8), ("A", 46)}, "HIS_POS", HIS_POS_OFFSET),
        (
            ("cif", "3N0I.cif"),
            {(c, r) for c in "ABC" for r in (196, 213, 333)},
            "CYS_DEP",
            CYS_DEP_OFFSET,
        ),
    ],
    ids=["6dmz_free_his_disulfide_cys", "3n0i_free_cys_zinc_his"],
)
def test_the_palette_offers_alternatives_only_when_asked_and_where_recorded(
    path, flagged, added, offset, torch_device
):
    """Disulfide cysteines and zinc-bound histidines record none."""
    pose_stack = _pose(path, torch_device, alternatives=True)
    alternatives = block_alternatives(pose_stack)
    info = pose_stack.pdb_info
    assert {
        (info.chain_labels[p, b], int(info.residue_labels[p, b]))
        for p, b in alternatives
    } == flagged

    default = PackerTask(pose_stack, PackerPalette())
    widened = PackerTask(pose_stack, PackerPalette(protonation_alternatives=True))
    assert default.per_block_considered_block_type_offset is None
    offsets = widened.per_block_considered_block_type_offset
    for pose, block in torch.nonzero(pose_stack.block_type_ind64 >= 0).tolist():
        recorded = alternatives.get((pose, block), {})
        names = _names(widened, pose, block)
        assert names == _names(default, pose, block) + [added] * bool(recorded)
        assert recorded.get(added, offset) == pytest.approx(offset)
        torch.testing.assert_close(
            offsets[pose, block, : len(names)].cpu(),
            torch.tensor([recorded.get(name, 0.0) for name in names]),
        )


def _packer_tables(pose_stack, palette, offsets=True):
    """The rotamers and packer energy tables of repacking around HIS 8 and 46."""
    db = ParameterDatabase.get_default()
    set_task = SetPackerTask.from_packer_task(
        _repacking_task(pose_stack, palette, [8, 46])
    )
    if not offsets:
        set_task.per_block_considered_block_type_offset = None
    pose_stack, rotamer_set = build_rotamers(pose_stack, set_task, db.chemical)
    sfxn = beta2016_score_function(pose_stack.device, param_db=db)
    energies = _calculate_packer_energies(pose_stack, sfxn, rotamer_set, set_task)
    return rotamer_set, energies[0], energies[4].to(torch.int64)


def test_the_packer_adds_each_block_types_offset_to_its_rotamers(torch_device):
    pose_stack = _pose(DMZ, torch_device, alternatives=True)
    palette = PackerPalette(protonation_alternatives=True)
    rotamer_set, offset, bc_rot_to_orig_rot = _packer_tables(pose_stack, palette)
    _, plain, _ = _packer_tables(pose_stack, palette, offsets=False)

    types = pose_stack.packed_block_types.active_block_types
    block_type = rotamer_set.block_type_ind_for_rot[bc_rot_to_orig_rot].tolist()
    is_his_pos = torch.tensor([types[j].base_name == "HIS_POS" for j in block_type])
    assert is_his_pos.any()
    torch.testing.assert_close(
        (offset.energy1b - plain.energy1b).cpu(),
        is_his_pos.to(torch.float32) * HIS_POS_OFFSET,
        atol=1e-4,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("path", "res_label", "label", "charge", "hydrogens"),
    [
        (F8B, 140, "HIS_POS", 1, {"ND1": 1, "NE2": 1}),
        (DMZ, 46, "HIS_D", 0, {"ND1": 1, "NE2": 0}),
    ],
    ids=["6f8b_buried_his_by_a_carboxylate", "6dmz_isolated_exposed_his"],
)
def test_packing_reports_the_protonation_it_chose(
    path, res_label, label, charge, hydrogens, torch_device
):
    """What beta2016 and the offsets give, not a calibrated pKa.

    6F8B's buried HIS 140, its ring 2.9 A from a carboxylate, takes the
    cation; 6DMZ's exposed HIS 46, 19 A from any, stays neutral.
    """
    pose_stack = _pose(path, torch_device, alternatives=True)
    palette = PackerPalette(protonation_alternatives=True)
    task = _repacking_task(pose_stack, palette, [res_label])
    sfxn = beta2016_score_function(torch_device)

    packed = pack_rotamers(pose_stack, sfxn, task)

    (choice,) = [
        c for c in chosen_protonation_variants(packed) if c.res_label == res_label
    ]
    assert (choice.label, choice.charge, choice.hydrogens) == (label, charge, hydrogens)
    offsets = block_alternatives(pose_stack)[choice.pose, choice.block]
    assert choice.offset == offsets[label]


def test_recording_alternatives_leaves_the_default_path_unchanged(torch_device):
    """The default palette packs a pose recording alternatives from identical tables."""
    plain, flagged = (_pose(DMZ, torch_device, alternatives=a) for a in (False, True))
    names = plain.pdb_info.residue_annotations.dtype.names
    assert flagged.pdb_info.residue_annotations.dtype.names == (
        *names,
        PROTONATION_ALTERNATIVES,
    )
    assert torch.equal(plain.block_type_ind64, flagged.block_type_ind64)
    assert torch.equal(plain.coords, flagged.coords)
    assert chosen_protonation_variants(plain) == []

    (rotamers, tables, _), (flagged_rotamers, flagged_tables, _) = (
        _packer_tables(pose_stack, PackerPalette()) for pose_stack in (plain, flagged)
    )
    assert torch.equal(rotamers.coords, flagged_rotamers.coords)
    assert torch.equal(tables.energy1b, flagged_tables.energy1b)
    assert torch.equal(tables.energy2b, flagged_tables.energy2b)
