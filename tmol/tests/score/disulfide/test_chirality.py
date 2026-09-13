"""Independent differentiable reference, including the mixed D/L distribution.

Rosetta reference: FullatomDisulfideParams13 and score_this_disulfide at
RosettaCommons/rosetta commit de92a3c0dea8a010d372a22025e3e50bd4e2f33f.
The reference below differentiates the energy itself with PyTorch; it does not
reuse Rosetta's hand-written derivative or tmol's native geometry helpers.
"""

import math

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite
from tmol.score import ScoreFunction
from tmol.score._score_types import ScoreType
from tmol.tests.data import data_path


def pose_for(pair, device, reverse=False, mirror=False):
    array = pdbx.get_structure(
        pdbx.CIFFile.read(data_path("ncaa_fixtures/6dmz_mod_l.cif")),
        model=1,
        include_bonds=True,
    )
    residues = [array[array.res_id == r].copy() for r in (3, 47)]
    for i, residue in enumerate(residues):
        residue.res_name[:] = "DCY" if pair[i] == "d" else "CYS"
        residue.chain_id[:] = chr(ord("A") + i)
        if mirror:
            residue.coord *= -1
    if reverse:
        residues.reverse()
    pose = pose_stack_from_biotite(struc.concatenate(residues), device, no_optH=True)
    atoms = []
    for i, names in enumerate((("CA", "CB", "SG"), ("SG", "CB", "CA"))):
        bt = pose.packed_block_types.active_block_types[int(pose.block_type_ind[0, i])]
        start = int(pose.block_coord_offset[0, i])
        atoms.extend(start + bt.atom_to_idx[n] for n in names)
    return pose, atoms


def dihedral(a, b, c, d):
    axis = c - b
    axis = axis / axis.norm()
    left = a - b
    right = d - c
    left = left - left.dot(axis) * axis
    right = right - right.dot(axis) * axis
    return torch.atan2(torch.cross(axis, left, dim=0).dot(right), left.dot(right))


def angle(a, b, c):
    x, y = a - b, c - b
    return torch.atan2(torch.cross(x, y, dim=0).norm(), x.dot(y))


def reference(coords, pair, params):
    ca1, cb1, s1, s2, cb2, ca2 = coords

    # Parameter transport uses float32 before the scoring module is cast.
    def p(n):
        return float(torch.tensor(getattr(params, n), dtype=torch.float32))

    floor = math.exp(-20)
    z = ((s1 - s2).norm() - p("d_location")) / p("d_scale")
    energy = -p("shift") + p("wt_len") * (
        z.square() / 2 - torch.log(torch.erfc(-p("d_shape") * z / math.sqrt(2)) + floor)
    )
    for value in (angle(cb1, s1, s2), angle(cb2, s2, s1)):
        energy = energy + p("wt_ang") * (
            -p("a_logA") - p("a_kappa") * torch.cos(value - p("a_mu"))
        )

    def mixture(value, prefix, count, sign=1):
        terms = [
            torch.exp(
                p(f"{prefix}_logA{i}")
                + p(f"{prefix}_kappa{i}")
                * torch.cos(value - sign * p(f"{prefix}_mu{i}"))
            )
            for i in range(1, count + 1)
        ]
        return -torch.log(sum(terms) + floor)

    mixed = pair[0] != pair[1]
    energy = energy + p("wt_dih_ss") * mixture(
        dihedral(cb1, s1, s2, cb2),
        "dss_mixed" if mixed else "dss",
        2,
        -1 if not mixed and pair[0] == "d" else 1,
    )
    for i, value in enumerate((dihedral(ca1, cb1, s1, s2), dihedral(ca2, cb2, s2, s1))):
        energy = energy + p("wt_dih_cs") * mixture(
            value, "dcs", 3, -1 if pair[i] == "d" else 1
        )
    return energy


@pytest.mark.parametrize("pair", ["ll", "ld", "dl", "dd"])
def test_energy_and_both_native_gradient_paths_match_reference(pair, torch_device):
    pose, atoms = pose_for(pair, torch_device)
    db = ParameterDatabase.get_default()
    score = ScoreFunction(db, torch_device)
    score.set_weight(ScoreType.disulfide, 1.0)
    whole = score.render_whole_pose_scoring_module(pose)
    blocks = score.render_block_pair_scoring_module(pose)
    for distance in (1.90, 2.05, 2.35):
        coords = pose.coords.double().detach().clone()
        direction = coords[0, atoms[3]] - coords[0, atoms[2]]
        coords[0, atoms[3]] = (
            coords[0, atoms[2]] + distance * direction / direction.norm()
        )
        coords.requires_grad_(True)
        expected = reference(
            coords[0, atoms], pair, db.scoring.disulfide.global_parameters
        )
        expected_grad = torch.autograd.grad(expected, coords)[0]
        actual = whole(coords).sum()
        actual_grad = torch.autograd.grad(actual, coords)[0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-5, rtol=3e-5)
        # Block-pair scores use the upper triangle. Asymmetric weights
        # exercise the separate backward kernel and this storage contract.
        weights = torch.tensor([[0.0, 0.7], [1.9, 0.0]], device=torch_device)
        weighted = (blocks(coords) * weights).sum()
        weighted_grad = torch.autograd.grad(weighted, coords)[0]
        torch.testing.assert_close(weighted, expected * 0.7, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(
            weighted_grad, expected_grad * 0.7, atol=4e-5, rtol=4e-5
        )


@pytest.mark.parametrize("pair", ["ll", "ld", "dl", "dd"])
def test_energy_and_gradient_are_permutation_and_mirror_invariant(pair, torch_device):
    db = ParameterDatabase.get_default()
    score = ScoreFunction(db, torch_device)
    score.set_weight(ScoreType.disulfide, 1.0)
    results = []
    mirror_pair = "".join("d" if c == "l" else "l" for c in pair)
    for current, reverse, mirror in (
        (pair, False, False),
        (pair, True, False),
        (mirror_pair, False, True),
    ):
        pose, atoms = pose_for(current, torch_device, reverse, mirror)
        coords = pose.coords.double().requires_grad_(True)
        energy = score.render_whole_pose_scoring_module(pose)(coords).sum()
        gradient = torch.autograd.grad(energy, coords)[0][0, atoms]
        if reverse:
            gradient = gradient.flip(0)
        if mirror:
            gradient = -gradient
        results.append((energy, gradient))
    for energy, gradient in results[1:]:
        torch.testing.assert_close(energy, results[0][0], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(gradient, results[0][1], atol=5e-5, rtol=5e-5)
