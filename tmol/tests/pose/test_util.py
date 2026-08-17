import numpy
import pytest
import torch

from tmol.io import create_pose_stack_from_sequences
from tmol.pose import (
    EXTENDED_BACKBONE_TORSIONS,
    extended_pose_stack_from_sequences,
    get_named_torsions,
    get_torsion_names,
    set_named_torsions,
)

LIGAND_SMILES = "CCO"


def block_names(pose_stack, pose):
    pbt = pose_stack.packed_block_types
    return [
        pbt.active_block_types[int(bt_ind)].name
        for bt_ind in pose_stack.block_type_ind64[pose]
        if int(bt_ind) != -1
    ]


def real_coords(pose_stack, pose):
    n_blocks = int((pose_stack.block_type_ind64[pose] != -1).sum())
    last = n_blocks - 1
    bt_ind = int(pose_stack.block_type_ind64[pose, last])
    n_atoms = int(pose_stack.block_coord_offset64[pose, last]) + int(
        pose_stack.packed_block_types.n_atoms[bt_ind]
    )
    return pose_stack.coords[pose, :n_atoms].cpu().numpy()


def min_interatomic_distance(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    dist = numpy.linalg.norm(diff, axis=-1)
    numpy.fill_diagonal(dist, numpy.inf)
    return dist.min()


def assert_backbone_is_ideal(pose_stack, pose):
    """Every polymer backbone torsion sits at its target value."""
    pbt = pose_stack.packed_block_types
    for block in range(pose_stack.max_n_blocks):
        bt_ind = int(pose_stack.block_type_ind64[pose, block])
        if bt_ind == -1:
            continue
        bt = pbt.active_block_types[bt_ind]
        targets = EXTENDED_BACKBONE_TORSIONS.get(bt.properties.polymer.backbone_type)
        if targets is None:
            continue
        measured = get_named_torsions(pose_stack, pose, block)
        for name, target in targets.items():
            if name not in measured or numpy.isnan(measured[name]):
                continue
            delta = (measured[name] - target + 180.0) % 360.0 - 180.0
            assert abs(delta) < 1e-2, (bt.name, name, measured[name], target)


def test_extended_pose_stack_protein_only(torch_device):
    pose_stack = extended_pose_stack_from_sequences("ACDEFG", device=torch_device)

    assert pose_stack.n_poses == 1
    assert block_names(pose_stack, 0) == [
        "ALA:nterm",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY:cterm",
    ]

    coords = real_coords(pose_stack, 0)
    assert numpy.isfinite(coords).all()
    assert min_interatomic_distance(coords) > 0.9
    assert_backbone_is_ideal(pose_stack, 0)


def test_extended_pose_stack_dna_only(torch_device):
    pose_stack = extended_pose_stack_from_sequences("acgt", device=torch_device)

    assert block_names(pose_stack, 0) == ["DA:na5prime", "DC", "DG", "DT:na3prime"]

    coords = real_coords(pose_stack, 0)
    assert numpy.isfinite(coords).all()
    assert min_interatomic_distance(coords) > 0.9
    assert_backbone_is_ideal(pose_stack, 0)

    # the ribose keeps the C2'-endo pucker carried by the icoors
    for block in range(4):
        pucker = get_named_torsions(pose_stack, 0, block)
        assert pucker["nu0"] == pytest.approx(-20.7, abs=2.0)
        assert pucker["nu1"] == pytest.approx(34.0, abs=2.0)
        assert pucker["delta"] == pytest.approx(143.0, abs=2.0)


def test_extended_pose_stack_ligand_only(torch_device):
    pose_stack = extended_pose_stack_from_sequences(
        f"X({LIGAND_SMILES})", device=torch_device
    )

    assert len(block_names(pose_stack, 0)) == 1
    coords = real_coords(pose_stack, 0)
    assert numpy.isfinite(coords).all()
    assert min_interatomic_distance(coords) > 0.9


def test_extended_pose_stack_protein_dna_and_ligand(torch_device):
    pose_stack, context = extended_pose_stack_from_sequences(
        f"ACD:acg:X({LIGAND_SMILES})", device=torch_device, return_context=True
    )

    names = block_names(pose_stack, 0)
    assert names[:3] == ["ALA:nterm", "CYS", "ASP:cterm"]
    assert names[3:6] == ["DA:na5prime", "DC", "DG:na3prime"]
    assert names[6] == context.ligand_names[LIGAND_SMILES]

    # protein, nucleic acid and ligand each land in their own chain
    assert sorted(set(pose_stack.chain_id[0, :7].tolist())) == [0, 1, 2]

    coords = real_coords(pose_stack, 0)
    assert numpy.isfinite(coords).all()
    assert min_interatomic_distance(coords) > 0.9
    assert_backbone_is_ideal(pose_stack, 0)


def test_extended_pose_stack_repeated_protein(torch_device):
    pose_stack = extended_pose_stack_from_sequences(
        ["ACDEFG"] * 10, device=torch_device
    )

    assert pose_stack.n_poses == 10
    for pose in range(1, 10):
        assert block_names(pose_stack, pose) == block_names(pose_stack, 0)
        numpy.testing.assert_allclose(
            real_coords(pose_stack, pose), real_coords(pose_stack, 0), atol=1e-5
        )


def test_set_named_torsions_roundtrip(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)

    before = real_coords(pose_stack, 0)
    moved = set_named_torsions(pose_stack, 0, 1, "chi1", 62.5)

    assert get_named_torsions(moved, 0, 1, "chi1") == pytest.approx(62.5, abs=1e-3)
    assert pose_stack.coords is not moved.coords
    assert get_named_torsions(pose_stack, 0, 1, "chi1") != pytest.approx(62.5, abs=1e-3)

    # rooting at the first residue leaves everything before the bond in place
    n_before = int(moved.block_coord_offset64[0, 1])
    numpy.testing.assert_allclose(
        real_coords(moved, 0)[:n_before], before[:n_before], atol=1e-4
    )


def test_set_named_torsions_batch_roundtrip(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)

    blocks = [1, 2, 3]
    phis = [-57.0, -60.0, -63.0]
    psis = [-47.0, -45.0, -43.0]
    moved = set_named_torsions(pose_stack, [0] * 3, blocks, ["phi"] * 3, phis)
    moved = set_named_torsions(moved, [0] * 3, blocks, ["psi"] * 3, psis)

    for block, phi, psi in zip(blocks, phis, psis):
        measured = get_named_torsions(moved, 0, block)
        assert measured["phi"] == pytest.approx(phi, abs=1e-3)
        assert measured["psi"] == pytest.approx(psi, abs=1e-3)


def test_set_named_torsions_radians(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)

    target = numpy.radians(-71.0)
    moved = set_named_torsions(pose_stack, 0, 2, "chi1", target, degrees=False)

    assert get_named_torsions(moved, 0, 2, "chi1", degrees=False) == pytest.approx(
        target, abs=1e-5
    )


def test_set_named_torsions_absent_torsion_raises(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)

    # the nterm patch removes the down connection, and phi along with it
    assert "phi" not in get_torsion_names(pose_stack, 0, 0)
    with pytest.raises(ValueError, match="no torsion"):
        set_named_torsions(pose_stack, 0, 0, "phi", -60.0)


def test_set_named_torsions_undefined_torsion_raises(torch_device):
    pose_stack = extended_pose_stack_from_sequences(
        "AKLFG", device=torch_device, termini=False
    )

    # unpatched, residue 0 keeps phi, but it reaches a residue that is not there
    assert "phi" in get_torsion_names(pose_stack, 0, 0)
    assert numpy.isnan(get_named_torsions(pose_stack, 0, 0, "phi"))
    with pytest.raises(ValueError, match="undefined"):
        set_named_torsions(pose_stack, 0, 0, "phi", -60.0)


def c_to_n_fold_forest(pose_stack):
    """Fold forest rooting each single-chain pose at its last residue."""
    from tmol.kinematics import EdgeType, FoldForest

    n_poses = pose_stack.n_poses
    edges = numpy.full((n_poses, 2, 4), -1, dtype=int)
    for pose in range(n_poses):
        last = int((pose_stack.block_type_ind64[pose] != -1).sum()) - 1
        edges[pose, 0] = [EdgeType.root_jump, -1, last, -1]
        edges[pose, 1] = [EdgeType.polymer, last, 0, -1]
    return FoldForest.from_edges(edges)


def test_named_torsions_agree_across_fold_forests(torch_device):
    pose_stack = extended_pose_stack_from_sequences("AKLFG", device=torch_device)
    reversed_ff = c_to_n_fold_forest(pose_stack)

    targets = {"phi": -61.0, "psi": -43.0, "chi1": 58.0}
    names, values = list(targets), list(targets.values())
    default_moved = set_named_torsions(pose_stack, 0, 2, names, values)
    reversed_moved = set_named_torsions(
        pose_stack, 0, 2, names, values, fold_forest=reversed_ff
    )

    # both trees drive the torsion to the requested value
    for name, target in targets.items():
        assert get_named_torsions(default_moved, 0, 2, name) == pytest.approx(
            target, abs=1e-3
        )
        assert get_named_torsions(reversed_moved, 0, 2, name) == pytest.approx(
            target, abs=1e-3
        )

    # ... but they move opposite ends of the chain
    start = real_coords(pose_stack, 0)
    n_first = int(pose_stack.block_coord_offset64[0, 1])
    last_offset = int(pose_stack.block_coord_offset64[0, 4])
    numpy.testing.assert_allclose(
        real_coords(default_moved, 0)[:n_first], start[:n_first], atol=1e-4
    )
    numpy.testing.assert_allclose(
        real_coords(reversed_moved, 0)[last_offset:], start[last_offset:], atol=1e-4
    )
    assert not numpy.allclose(
        real_coords(default_moved, 0), real_coords(reversed_moved, 0), atol=1e-3
    )


PROTEIN_GOLD_TORSION_NAMES = {
    "ALA:nterm": ["psi", "omega"],
    "LYS": ["phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4"],
    "PRO": ["phi", "psi", "omega", "chi1", "chi2", "chi3"],
    "VAL": ["phi", "psi", "omega", "chi1"],
    "TRP:cterm": ["phi", "chi1", "chi2"],
}

DNA_GOLD_TORSION_NAMES = {
    "DA:na5prime": [
        "gamma", "delta", "epsilon", "zeta", "chi1", "nu0", "nu1", "nu4", "chi3",
    ],
    "DC": [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
        "chi1", "nu0", "nu1", "nu4",
    ],
    "DG": [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
        "chi1", "nu0", "nu1", "nu4",
    ],
    "DT:na3prime": [
        "alpha", "beta", "gamma", "delta", "chi1", "nu0", "nu1", "nu4", "chi4",
    ],
}  # fmt: skip

RNA_GOLD_TORSION_NAMES = {
    "RA:na5prime": [
        "gamma", "delta", "epsilon", "zeta", "chi1", "chi2",
        "nu0", "nu1", "nu4", "chi3",
    ],
    "RC": [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi1", "chi2",
        "nu0", "nu1", "nu4",
    ],
    "RG": [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi1", "chi2",
        "nu0", "nu1", "nu4",
    ],
    "RU:na3prime": [
        "alpha", "beta", "gamma", "delta", "chi1", "chi2",
        "nu0", "nu1", "nu4", "chi4",
    ],
}  # fmt: skip


@pytest.mark.parametrize(
    "seq,gold",
    [
        ("AKPVW", PROTEIN_GOLD_TORSION_NAMES),
        ("acgt", DNA_GOLD_TORSION_NAMES),
        ("a[RA]c[RC]g[RG]u[RU]", RNA_GOLD_TORSION_NAMES),
    ],
    ids=["protein", "dna", "rna"],
)
def test_get_torsion_names(seq, gold, torch_device):
    # names come from the block type, so the zero-coordinate builder suffices
    pose_stack = create_pose_stack_from_sequences(seq, device=torch_device)

    got = {
        name: get_torsion_names(pose_stack, 0, block)
        for block, name in enumerate(block_names(pose_stack, 0))
    }
    assert got == gold


def test_get_torsion_names_non_polymer(torch_device):
    pose_stack = create_pose_stack_from_sequences("A[HOH]", device=torch_device)
    assert get_torsion_names(pose_stack, 0, 0) == []


def test_get_torsion_names_rejects_absent_block(torch_device):
    pose_stack = create_pose_stack_from_sequences(["AAA", "AA"], device=torch_device)
    with pytest.raises(ValueError, match="not a real block"):
        get_torsion_names(pose_stack, 1, 2)


def test_extended_pose_stack_device(torch_device):
    pose_stack = extended_pose_stack_from_sequences("ACD", device=torch_device)
    assert pose_stack.coords.device.type == torch_device.type
    assert pose_stack.coords.dtype == torch.float32
