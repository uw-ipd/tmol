import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.kinematics import (
    validate_fold_forest,
    FoldForest,
    EdgeType,
    _build_pose_fold_forest,
)


def _real_edges(fold_forest, pose_idx):
    """Return the set of (type, start, end) tuples for all real edges in a pose."""
    return {
        (EdgeType(int(e[0])), int(e[1]), int(e[2]))
        for e in fold_forest.edges[pose_idx]
        if e[0] != -1
    }


def _check_jump_indices(fold_forest, pose_idx):
    """Assert that jump indices form a valid 0..n_jumps-1 assignment.

    Only true jumps are numbered; a root jump is identified by its downstream
    block and carries -1, so numbering one would leave a gap in the jump
    indices that validate_fold_forest rejects.
    """
    n_e = fold_forest.n_edges[pose_idx]

    def indices_of(edge_type):
        return [
            int(fold_forest.edges[pose_idx, j, 3])
            for j in range(n_e)
            if fold_forest.edges[pose_idx, j, 0] == edge_type
        ]

    assert sorted(indices_of(EdgeType.jump)) == list(
        range(len(indices_of(EdgeType.jump)))
    )
    assert all(i == -1 for i in indices_of(EdgeType.root_jump))


def test_reasonable_fold_forest_smoke(default_database, erbb2_and_pertuzumab_pdb):
    torch_device = torch.device("cpu")
    p = pose_stack_from_pdb(erbb2_and_pertuzumab_pdb, torch_device)

    pose_stack = PoseStackBuilder.from_poses([p], torch_device)

    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    assert fold_forest.n_edges.shape[0] == pose_stack.n_poses
    assert fold_forest.max_n_edges == 6


def test_jagged_reasonable_fold_forest(
    ubq_pdb, erbb2_and_pertuzumab_pdb, default_database, dun_sampler, torch_device
):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    p2 = pose_stack_from_pdb(erbb2_and_pertuzumab_pdb, torch_device)

    pose_stack = PoseStackBuilder.from_poses([p1, p2], torch_device)
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)

    assert fold_forest.n_edges.shape[0] == pose_stack.n_poses
    assert fold_forest.max_n_edges == 6
    assert fold_forest.n_edges[0] == 2
    assert fold_forest.n_edges[1] == 6

    # Pose 0: ubiquitin — one polymer chain 0..75, one root-jump
    assert _real_edges(fold_forest, 0) == {
        (EdgeType.polymer, 0, 75),
        (EdgeType.root_jump, -1, 0),
    }
    _check_jump_indices(fold_forest, 0)

    # Pose 1: erbb2 + pertuzumab — three disconnected polymer chains
    assert _real_edges(fold_forest, 1) == {
        (EdgeType.polymer, 0, 554),
        (EdgeType.root_jump, -1, 0),
        (EdgeType.polymer, 555, 768),
        (EdgeType.root_jump, -1, 555),
        (EdgeType.polymer, 769, 990),
        (EdgeType.root_jump, -1, 769),
    }
    _check_jump_indices(fold_forest, 1)


# Two synthetic block types for the connectivity-only tests below.
# Type 0 is a polymer residue: down in slot 0, up in slot 1, the disulfide in
# slot 2, and slot 3 free for a conjugation. Type 1 is a non-polymer residue
# (a sugar, a ligand) whose four connections are all conjugations.
_UP_C = numpy.array([1, -1], dtype=numpy.int64)
_DOWN_C = numpy.array([0, -1], dtype=numpy.int64)
_DSLF_C = numpy.array([2, -1], dtype=numpy.int64)
_N_CONN = numpy.array([4, 4], dtype=numpy.int64)


def _linear_polymer_pose(segments, chain_ids, block_types=None, bonds=()):
    """Connectivity for one pose built from disjoint polymer segments.

    segments is a list of (first, last) inclusive residue ranges bonded
    up-to-down along the backbone; chain_ids gives each residue's biological
    chain. block_types names each residue's synthetic type, defaulting to the
    polymer type; bonds adds (res1, conn1, res2, conn2) connections on top.
    """
    n_res = len(chain_ids)
    bti = numpy.zeros(n_res, dtype=numpy.int64)
    if block_types is not None:
        bti = numpy.array(block_types, dtype=numpy.int64)
    irc = numpy.full((n_res, 4, 2), -1, dtype=numpy.int64)
    for first, last in segments:
        for r in range(first, last):
            irc[r, 1] = (r + 1, 0)
            irc[r + 1, 0] = (r, 1)
    for r1, c1, r2, c2 in bonds:
        irc[r1, c1] = (r2, c2)
        irc[r2, c2] = (r1, c1)
    return (
        bti,
        irc,
        _UP_C,
        _DOWN_C,
        _DSLF_C,
        _N_CONN,
        numpy.array(chain_ids, dtype=numpy.int64),
    )


def test_fold_forest_numbers_only_true_jumps():
    """A chain-internal break alongside separate chains must number contiguously.

    Residues 0-5 are one biological chain broken between 2 and 3, so the second
    segment is reached by a true jump; residues 6-8 are a second chain and are
    root-jumped. The true jump is emitted after a root jump, which is what used
    to push its index past the number of jumps.
    """
    edges = _build_pose_fold_forest(
        *_linear_polymer_pose([(0, 2), (3, 5), (6, 8)], [0] * 6 + [1] * 3)
    )
    by_type = {}
    for edge_type, start, end, jump_ind in edges:
        by_type.setdefault(EdgeType(edge_type), []).append((start, end, jump_ind))

    assert by_type[EdgeType.root_jump] == [(-1, 0, -1), (-1, 6, -1)]
    assert by_type[EdgeType.jump] == [(2, 3, 0)]
    assert by_type[EdgeType.polymer] == [(0, 2, -1), (3, 5, -1), (6, 8, -1)]

    validate_fold_forest(
        numpy.array([9], dtype=numpy.int64),
        numpy.array([edges], dtype=numpy.int64),
    )


def _typed_edges(edges):
    by_type = {}
    for edge_type, start, end, extra in edges:
        by_type.setdefault(EdgeType(edge_type), []).append((start, end, extra))
    return by_type


def test_fold_forest_routes_through_a_conjugation():
    """A glycan tree is reached through its bonds, not by jumps.

    Residues 0-3 are a protein chain; residue 2 carries a sugar at 4, which
    branches to 5 and 6. Every sugar must be built by a chemical edge naming
    the connection on its parent, so that a torsion about a glycosidic bond
    moves everything beyond it.
    """
    edges = _build_pose_fold_forest(
        *_linear_polymer_pose(
            [(0, 3)],
            [0] * 7,
            block_types=[0] * 4 + [1] * 3,
            bonds=[(2, 3, 4, 0), (4, 1, 5, 0), (4, 2, 6, 0)],
        )
    )
    by_type = _typed_edges(edges)

    assert by_type[EdgeType.root_jump] == [(-1, 0, -1)]
    # the chain splits at 2: an edge may only start where another one ends
    assert by_type[EdgeType.polymer] == [(0, 2, -1), (2, 3, -1)]
    assert by_type[EdgeType.chemical] == [(2, 4, 3), (4, 5, 1), (4, 6, 2)]
    assert EdgeType.jump not in by_type

    validate_fold_forest(
        numpy.array([7], dtype=numpy.int64),
        numpy.array([edges], dtype=numpy.int64),
    )


def test_fold_forest_leaves_disulfides_alone():
    """A disulfide is not routed through; its two chains stay independent."""
    edges = _build_pose_fold_forest(
        *_linear_polymer_pose([(0, 2), (3, 5)], [0] * 3 + [1] * 3, bonds=[(0, 2, 5, 2)])
    )
    by_type = _typed_edges(edges)

    assert by_type[EdgeType.root_jump] == [(-1, 0, -1), (-1, 3, -1)]
    assert EdgeType.chemical not in by_type


def test_fold_forest_breaks_a_cycle_at_a_chemical_bond():
    """A conjugation that would close a cycle is the bond that gets dropped."""
    edges = _build_pose_fold_forest(
        *_linear_polymer_pose(
            [(0, 3)],
            [0] * 6,
            block_types=[0] * 4 + [1] * 2,
            bonds=[(0, 3, 4, 0), (4, 1, 5, 0), (5, 1, 3, 3)],
        )
    )
    by_type = _typed_edges(edges)

    # three bonds join the two ligands to the chain, but a tree can use only
    # two of them; the chain itself is never broken
    assert by_type[EdgeType.polymer] == [(0, 3, -1)]
    assert len(by_type[EdgeType.chemical]) == 2

    validate_fold_forest(
        numpy.array([6], dtype=numpy.int64),
        numpy.array([edges], dtype=numpy.int64),
    )


def test_fold_forest_builds_outward_from_a_mid_chain_conjugation():
    """A bond landing mid-chain builds that chain in both directions."""
    edges = _build_pose_fold_forest(
        *_linear_polymer_pose([(0, 2), (3, 6)], [0] * 3 + [1] * 4, bonds=[(1, 3, 5, 3)])
    )
    by_type = _typed_edges(edges)

    assert by_type[EdgeType.chemical] == [(1, 5, 3)]
    # the chain splits at 1, where the chemical bond leaves it
    assert by_type[EdgeType.polymer] == [
        (0, 1, -1),
        (1, 2, -1),
        (5, 3, -1),
        (5, 6, -1),
    ]
    assert EdgeType.jump not in by_type

    validate_fold_forest(
        numpy.array([7], dtype=numpy.int64),
        numpy.array([edges], dtype=numpy.int64),
    )
