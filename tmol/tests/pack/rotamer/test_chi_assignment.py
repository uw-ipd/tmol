"""Chi writes preserve ring offsets, sparse columns and other conformer DOFs."""

import numpy
import pytest
import torch

from tmol.pack.rotamer import (
    assign_chi_dofs_from_samples,
    coalesce_single_residue_kinforests,
    construct_single_residue_kinforest,
)
from tmol.pack.rotamer._build_rotamers import _build_ring_chi_phi_c_corrections
from tmol.pose import PackedBlockTypes


def chi_assignment_inputs(database, restypes, device):
    types = [restypes.restype_map[name][0] for name in ("ILE", "ALA", "PRO")]
    pbt = PackedBlockTypes.from_restype_list(
        database.chemical, restypes, types, device=device
    )
    for rt in types:
        construct_single_residue_kinforest(rt)
    coalesce_single_residue_kinforests(pbt)
    block_types = torch.tensor([0, 1, 2, 0, 2], device=device)
    selected = torch.tensor([4, 0, 2], device=device)
    counts = pbt.n_atoms[block_types].long()
    offsets = counts.cumsum(0) - counts
    chi_atoms = torch.full((3, 7), -1, dtype=torch.int32, device=device)
    for row, ti in enumerate((2, 0, 2)):
        chis = [
            uaids
            for name, uaids in types[ti].torsion_to_uaids.items()
            if name.startswith("chi") and all(u[0] >= 0 for u in uaids)
        ]
        for col, uaids in enumerate(chis):
            chi_atoms[row, 2 * col] = int(uaids[2][0])
    chi = torch.arange(21, dtype=torch.float32, device=device).reshape(3, 7) / 10
    dofs = (
        torch.arange(
            (int(counts.sum()) + 1) * 9, device=device, dtype=torch.float32
        ).reshape(-1, 9)
        / 100
    )
    return [
        pbt,
        block_types,
        selected,
        torch.tensor([1, 0, 2], dtype=torch.int32, device=device),
        torch.tensor([2, 0, 2], dtype=torch.int32, device=device),
        offsets,
        chi_atoms,
        chi,
        dofs,
    ]


def _signed_dihedral(xyz):
    # Independent projection formula, without the production chi helper.
    a, b, c, d = xyz
    axis = c - b
    axis /= numpy.linalg.norm(axis)
    first, last = a - b, d - c
    first -= numpy.dot(first, axis) * axis
    last -= numpy.dot(last, axis) * axis
    return numpy.arctan2(
        numpy.dot(numpy.cross(axis, first), last), numpy.dot(first, last)
    )


@pytest.mark.parametrize(
    "layout", ["gapped", "strided", "missing", "zero_columns", "zero_rows"]
)
def test_chi_assignment_preserves_independent_dof_oracle(
    layout, default_database, fresh_default_restype_set, torch_device, monkeypatch
):
    args = chi_assignment_inputs(
        default_database, fresh_default_restype_set, torch_device
    )
    pbt, bt, selected, _, _, offsets, atoms, chi, dofs = args
    if layout == "missing":
        atoms.fill_(-1)
    elif layout == "strided":
        atoms, chi = atoms[:, ::2], chi[:, ::2]
    elif layout == "zero_columns":
        atoms, chi = atoms[:, :0], chi[:, :0]
    elif layout == "zero_rows":
        args[2] = selected = selected[:0]
        args[4] = args[4][:0]
        atoms, chi = atoms[:0], chi[:0]
    args[6], args[7] = atoms, chi
    original_chi = chi.clone()
    expected = dofs.clone()
    ring_offsets = []
    for row in range(atoms.shape[0]):
        global_rot = int(selected[row])
        rt = pbt.active_block_types[int(bt[global_rot])]
        for col in range(atoms.shape[1]):
            atom = int(atoms[row, col])
            if atom < 0:
                continue
            four = next(
                [int(u[0]) for u in uaids]
                for name, uaids in rt.torsion_to_uaids.items()
                if name.startswith("chi") and uaids[2][0] == atom
            )
            kfo = int(rt.rotamer_kinforest.kinforest_idx[atom])
            # PRO chi3 closes its ring onto N; its fourth atom cannot move.
            correction = 0.0
            if rt.name == "PRO" and rt.atoms[four[3]].name == "N":
                correction = _signed_dihedral(
                    rt.ideal_coords[rt.at_to_icoor_ind][four].astype(numpy.float64)
                ) - float(rt.rotamer_kinforest.dofs_ideal[kfo, 3])
                ring_offsets.append(correction)
            expected[int(offsets[global_rot]) + kfo + 1, 3] = chi[
                row, col
            ] - numpy.float32(correction)
    if layout in ("gapped", "strided"):
        assert ring_offsets and all(abs(value) > 0.01 for value in ring_offsets)

    def host_copy(*args, **kwargs):
        raise AssertionError(
            "Chi assignment must not copy conformer indices to the host"
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", host_copy)
        assign_chi_dofs_from_samples(*args)
    torch.testing.assert_close(dofs, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(chi, original_chi, rtol=0, atol=0)
    table = _build_ring_chi_phi_c_corrections(pbt)
    assert table.device == torch_device
    assert _build_ring_chi_phi_c_corrections(pbt) is table
