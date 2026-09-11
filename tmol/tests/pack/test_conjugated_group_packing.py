"""Packing a residue together with whatever is bonded to its sidechain.

A glycan or a ligand joined to a sidechain has to be sampled with it: the torsion
about a linkage bond has its fourth atom in the neighbouring residue, and moving
one without the other pulls the bond apart. These tests pin the three things that
can go wrong -- the members falling out of step, the energies the packer works
from disagreeing with what the pose actually scores, and the bond not surviving a
pack at all.
"""

import pytest
import torch

from tmol.io import pose_stack_from_biotite
from tmol.io._cif import atom_array_from_cif
from tmol.pack import PackerTask, PackerPalette, pack_rotamers
from tmol.pack._impose_rotamers import (
    chosen_rotamer_for_block,
    impose_top_rotamer_assignments,
)
from tmol.pack._pack_rotamers import _calculate_packer_energies
from tmol.pack._packer_task import SetPackerTask
from tmol.pack import run_simulated_annealing
from tmol.pack.rotamer import FixedAAChiSampler, build_rotamers
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.pack.rotamer._conjugated_groups import (
    add_conjugated_group_sampler,
    find_conjugated_groups,
    write_group_members,
)
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path

FIXTURES = {
    "biotin": "lys_biotin_1bdo",
    "oglycan": "oglycan_sia_1g1s",
    "nglycan": "nglycan_tree_1ax2",
}

# groups big enough that a full pack is only practical on the gpu
BIG = {"oglycan", "nglycan"}


def _pose(stem, device):
    aa = atom_array_from_cif(data_path("covalent_fixtures", stem + ".cif"))
    return pose_stack_from_biotite(
        aa, device, prepare_ligands=True, no_optH=True, return_context=True
    )


def _task(pose_stack, param_db, device):
    task = PackerTask(pose_stack, PackerPalette())
    task.add_conformer_sampler(create_dunbrack_sampler_from_database(param_db, device))
    task.add_conformer_sampler(FixedAAChiSampler())
    sampler = add_conjugated_group_sampler(task, pose_stack)
    task.restrict_to_repacking()
    return task, sampler


def _linkage_lengths(pose_stack, groups):
    """Length of every bond joining two blocks of a group."""
    pbt = pose_stack.packed_block_types
    out = {}
    for group in groups:
        for parent_i, pconn, child_i, cconn in group.links:
            pb, cb = group.blocks[parent_i], group.blocks[child_i]
            tp = pbt.active_block_types[int(pose_stack.block_type_ind[0, pb])]
            tc = pbt.active_block_types[int(pose_stack.block_type_ind[0, cb])]
            p = pose_stack.coords[
                0,
                int(pose_stack.block_coord_offset[0, pb])
                + int(tp.ordered_connection_atoms[pconn]),
            ]
            c = pose_stack.coords[
                0,
                int(pose_stack.block_coord_offset[0, cb])
                + int(tc.ordered_connection_atoms[cconn]),
            ]
            out[(pb, cb)] = float((p - c).norm())
    return out


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_every_member_of_a_group_gets_the_same_rotamers(fixture, torch_device):
    """Rotamer k of each member is group conformer k, so the counts must agree.

    If they diverge the correspondence is silently broken and the packer can
    pair one member's conformer with another's.
    """
    pose_stack, ctx = _pose(FIXTURES[fixture], torch_device)
    param_db = ctx.parameter_database
    task, sampler = _task(pose_stack, param_db, torch_device)

    pose_stack, rotamer_set = build_rotamers(
        pose_stack, SetPackerTask.from_packer_task(task), param_db.chemical
    )
    groups = find_conjugated_groups(pose_stack)
    assert groups, "the fixture is supposed to have a conjugated group"

    for group in groups:
        counts = {
            int(rotamer_set.n_rots_for_block[group.pose, b]) for b in group.blocks
        }
        assert len(counts) == 1, (
            f"members of the group at anchor {group.anchor} disagree on how "
            f"many rotamers they have: {counts}"
        )
        assert counts.pop() > 1, "the group should be sampled, not frozen"


def test_a_group_is_sampled_as_the_product_of_its_parts(torch_device):
    """The count is the anchor's library rotamers times the tree's conformers."""
    pose_stack, ctx = _pose(FIXTURES["biotin"], torch_device)
    param_db = ctx.parameter_database
    task, sampler = _task(pose_stack, param_db, torch_device)
    set_task = SetPackerTask.from_packer_task(task)

    # the library sampler annotates the block types when the packer runs it;
    #    asking it for chi directly means doing that first, and the per-type
    #    annotation has to precede the packed one
    pbt = pose_stack.packed_block_types
    for rt in pbt.active_block_types:
        sampler.library_sampler.annotate_residue_type(rt)
    sampler.library_sampler.annotate_packed_block_types(pbt)
    groups = find_conjugated_groups(pose_stack)
    anchor_chi = sampler.anchor_library_chi(pose_stack, set_task, groups)
    enumerated = sampler.group_conformers(pose_stack, anchor_chi)
    assert enumerated, "the fixture is supposed to have a sampled group"

    for group, columns, conformers in enumerated:
        tree_cols = [c for c in columns if c[0] != 0]
        anchor_cols = [c for c in columns if c[0] == 0]
        assert anchor_cols, "the anchor's own chi belong in the product"
        n_tree = 1
        for owner, name, _b, _c in tree_cols:
            bt = pose_stack.packed_block_types.active_block_types[
                int(pose_stack.block_type_ind[group.pose, group.blocks[owner]])
            ]
            cs = next(x for x in bt.chi_samples if x.chi_dihedral == name)
            n_tree *= len(cs.samples) * (1 + 2 * len(cs.expansions))
        # the grid is a product of the per-chi sample counts, plus the one
        #    conformer the group came in with
        assert (conformers.shape[0] - 1) % n_tree == 0
        assert conformers.shape[1] == len(columns)


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_the_bond_survives_packing(fixture, torch_device):
    """Every bond joining a group's blocks keeps its length through a pack."""
    if fixture in BIG and torch_device.type == "cpu":
        pytest.skip(f"{fixture}'s group is too big to pack on cpu; cuda covers it")
    pose_stack, ctx = _pose(FIXTURES[fixture], torch_device)
    param_db = ctx.parameter_database
    sfxn = beta2016_score_function(torch_device, param_db=param_db)

    groups = find_conjugated_groups(pose_stack)
    before = _linkage_lengths(pose_stack, groups)

    task, _ = _task(pose_stack, param_db, torch_device)
    packed = pack_rotamers(pose_stack, sfxn, task, verbose=False)
    after = _linkage_lengths(packed, groups)

    for key, length in before.items():
        assert after[key] == pytest.approx(length, abs=1e-2), (
            f"the bond between blocks {key} went from {length:.3f} to "
            f"{after[key]:.3f} A during packing"
        )


@pytest.mark.parametrize("fixture", sorted(FIXTURES))
def test_the_packers_energy_matches_what_the_pose_scores(fixture, torch_device):
    """What annealing reports must be what the imposed pose actually scores.

    A group's rotamers are folded onto one representative before the energies
    are assembled; if any pair energy is dropped or double counted there, the
    number the packer works from stops describing the structure it picks.
    """
    if fixture in BIG and torch_device.type == "cpu":
        pytest.skip(f"{fixture}'s group is too big to pack on cpu; cuda covers it")
    pose_stack, ctx = _pose(FIXTURES[fixture], torch_device)
    param_db = ctx.parameter_database
    sfxn = beta2016_score_function(torch_device, param_db=param_db)
    task, _ = _task(pose_stack, param_db, torch_device)
    set_task = SetPackerTask.from_packer_task(task)

    pose_stack, rotamer_set = build_rotamers(pose_stack, set_task, param_db.chemical)
    (
        tables,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        collapse,
        bg_bg_energies,
        _t2,
        _t3,
    ) = _calculate_packer_energies(
        pose_stack, sfxn, rotamer_set, set_task, verbose=False
    )
    assert collapse is not None, "the group should have been folded together"

    scores, assignments = run_simulated_annealing(tables)
    # blocks treated as background are left out of the packer's tables
    scores = scores + bg_bg_energies.unsqueeze(1)
    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack,
        rotamer_set,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        assignments,
    )
    new_pose_stack = write_group_members(
        new_pose_stack,
        rotamer_set,
        collapse,
        chosen_rotamer_for_block(
            pose_stack,
            rotamer_for_nonmolten_block,
            n_molten_blocks_per_pose,
            bc_rot_offset_for_molten_block,
            bc_rot_to_orig_rot,
            assignments[:, 0, :],
        ),
    )

    wpsm = sfxn.render_whole_pose_scoring_module(new_pose_stack)
    torch.testing.assert_close(
        scores[:, 0], wpsm(new_pose_stack.coords), atol=1e-3, rtol=1e-5
    )
