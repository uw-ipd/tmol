"""A head-to-tail cyclic peptide has to close through both readers and survive relax.

SFTI-1 (PDB 1JBL) is 14 canonical residues joined C(14)-N(1), plus one
disulfide. The closing bond is the point: nothing distinguishes it from the
other thirteen peptide bonds chemically or geometrically, so a builder that
bonds residue i to i+1 by index leaves it out and caps both ends instead.

The CIF declares the bond in struct_conn; the PDB does not carry it into
tmol's parser at all, so the two readers have to arrive at the same answer by
different routes.
"""

import numpy
import torch
import pytest

from tmol.io import pose_stack_from_cif, pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import (
    build_rotamers,
    FixedAAChiSampler,
    IncludeCurrentSampler,
)
from tmol.score.backbone_torsion import BackboneTorsionEnergyTerm
from tmol.kinematics import FoldForest, MoveMap
from tmol.relax import fast_relax
from tmol.tests.data import data_path
from tmol.tests.relax.test_fast_relax import get_relax_sfxn

SEQUENCE = "GLY ARG CYS THR LYS SER ILE PRO PRO ILE CYS PHE PRO ASP".split()
DISULFIDE = (2, 10)

# a peptide bond, generously bounded; the deposited ones span 1.328-1.351
BOND_RANGE = (1.2, 1.5)

# enough to tell relax apart from a no-op, far below the drop it actually gets
MIN_RELAX_ENERGY_DROP = 1.0


def cyclic_pose_stack(reader, torch_device):
    if reader == "cif":
        return pose_stack_from_cif(
            data_path("cif", "cyclic_peptide_1jbl.cif"), torch_device
        )
    return pose_stack_from_pdb(
        str(data_path("pdb", "cyclic_peptide_1jbl.pdb")), torch_device
    )


def block_types(pose_stack):
    pbt = pose_stack.packed_block_types
    return [pbt.active_block_types[i] for i in pose_stack.block_type_ind64[0].tolist()]


def atom_coord(pose_stack, block, name):
    bt = block_types(pose_stack)[block]
    offset = int(pose_stack.block_coord_offset[0, block])
    coord = pose_stack.coords[0, offset + bt.atom_to_idx[name]]
    return coord.detach().cpu().numpy()


def peptide_bond_lengths(pose_stack):
    """C(i)-N(i+1) for every i, wrapping the last residue onto the first."""
    n_res = len(SEQUENCE)
    return numpy.array(
        [
            numpy.linalg.norm(
                atom_coord(pose_stack, i, "C")
                - atom_coord(pose_stack, (i + 1) % n_res, "N")
            )
            for i in range(n_res)
        ]
    )


def connection_partner(pose_stack, block, conn_name):
    bt = block_types(pose_stack)[block]
    names = [c.name for c in bt.connections]
    if conn_name not in names:
        return None
    return int(
        pose_stack.inter_residue_connections[0, block, names.index(conn_name), 0]
    )


@pytest.mark.parametrize("reader", ["cif", "pdb"])
def test_cyclic_fixture_reads_intact(reader, torch_device):
    """Guards the fixture itself, not the cyclization."""
    pose_stack = cyclic_pose_stack(reader, torch_device)
    types = block_types(pose_stack)
    assert len(types) == len(SEQUENCE)
    # the two cysteines pair, so they arrive as the disulfide-bonded form
    assert [t.base_name for t in types] == [
        "CYD" if i in DISULFIDE else name for i, name in enumerate(SEQUENCE)
    ]
    # the disulfide is the connection the builder already knows how to make
    assert connection_partner(pose_stack, DISULFIDE[0], "dslf") == DISULFIDE[1]
    assert connection_partner(pose_stack, DISULFIDE[1], "dslf") == DISULFIDE[0]


def test_cif_and_pdb_paths_agree(torch_device):
    """Both readers must reach the same chemistry from the same coordinates."""
    from_cif = cyclic_pose_stack("cif", torch_device)
    from_pdb = cyclic_pose_stack("pdb", torch_device)
    assert [t.name for t in block_types(from_cif)] == [
        t.name for t in block_types(from_pdb)
    ]
    for i in range(len(SEQUENCE)):
        for conn in ("down", "up", "dslf"):
            assert connection_partner(from_cif, i, conn) == connection_partner(
                from_pdb, i, conn
            ), (i, conn)


@pytest.mark.parametrize("reader", ["cif", "pdb"])
def test_cyclic_backbone_closes(reader, torch_device):
    pose_stack = cyclic_pose_stack(reader, torch_device)
    last = len(SEQUENCE) - 1
    assert connection_partner(pose_stack, 0, "down") == last
    assert connection_partner(pose_stack, last, "up") == 0


@pytest.mark.parametrize("reader", ["cif", "pdb"])
def test_cyclized_residues_take_no_terminus_variant(reader, torch_device):
    """A residue whose backbone continues is not a terminus, whatever its index."""
    pose_stack = cyclic_pose_stack(reader, torch_device)
    variants = [name for t in block_types(pose_stack) for name in t.name.split(":")[1:]]
    assert not [v for v in variants if v in ("nterm", "cterm")]


def _total_energy(sfxn, pose_stack):
    module = sfxn.render_whole_pose_scoring_module(pose_stack)
    return float(module(pose_stack.coords).sum().detach())


def test_relax_maintains_backbone_bonds(default_database, dun_sampler, torch_device):
    """Every peptide bond, the closing one included, stays a peptide bond."""
    pose_stack = cyclic_pose_stack("cif", torch_device)
    before = peptide_bond_lengths(pose_stack)
    assert numpy.all((before > BOND_RANGE[0]) & (before < BOND_RANGE[1]))

    sfxn = get_relax_sfxn(default_database, torch_device)
    start_energy = _total_energy(sfxn, pose_stack)
    move_map = MoveMap.from_pose_stack(pose_stack)
    move_map.move_all_jumps = True
    move_map.move_all_named_torsions = True

    def task_op(task):
        task.restrict_to_repacking()
        task.or_bump_check(True)
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(FixedAAChiSampler())
        task.add_conformer_sampler(IncludeCurrentSampler())

    relaxed = fast_relax(
        pose_stack,
        sfxn,
        PackerPalette(),
        move_map,
        FoldForest.reasonable_fold_forest(pose_stack),
        task_operations=[task_op],
        num_repeats=1,
    )

    after = peptide_bond_lengths(relaxed)
    assert numpy.all((after > BOND_RANGE[0]) & (after < BOND_RANGE[1])), after

    # relax accepts to best, so it cannot end worse than it started
    assert _total_energy(sfxn, relaxed) < start_energy - MIN_RELAX_ENERGY_DROP


def _backbone_torsion_term(default_database, pose_stack):
    term = BackboneTorsionEnergyTerm(default_database, pose_stack.device)
    for bt in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    return term


def test_closure_block_pair_energy_is_upper_triangular(default_database, torch_device):
    """The closing bond's omega/rama land at [0, 13], not the lower triangle."""
    pose_stack = cyclic_pose_stack("cif", torch_device)
    term = _backbone_torsion_term(default_database, pose_stack)

    scores = term.render_block_pair_scoring_module(pose_stack)(pose_stack.coords)
    scores = scores.sum(dim=0)[0]
    last = len(SEQUENCE) - 1

    assert scores[0, last] != 0.0
    assert scores[last, 0] == 0.0


def test_closure_rotamer_pairs_reach_the_packer(
    default_database, dun_sampler, torch_device
):
    """The packer drops any rotamer pair given outside the upper triangle."""
    pose_stack = cyclic_pose_stack("cif", torch_device)

    task = PackerTask(pose_stack, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(FixedAAChiSampler())
    task = SetPackerTask.from_packer_task(task)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, default_database.chemical
    )

    term = _backbone_torsion_term(default_database, pose_stack)
    scorer = term.render_rotamer_scoring_module(pose_stack, rotamer_set)
    indices = scorer.forward_split(rotamer_set.coords).coalesce().indices()

    # (subterm, pose, rot1, rot2)
    rot1, rot2 = indices[-2], indices[-1]
    assert torch.all(rot1 <= rot2)

    block_for_rot = rotamer_set.block_ind_for_rot
    pairs = {(int(a), int(b)) for a, b in zip(block_for_rot[rot1], block_for_rot[rot2])}
    assert (0, len(SEQUENCE) - 1) in pairs
