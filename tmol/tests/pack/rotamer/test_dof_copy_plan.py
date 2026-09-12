"""Copy plans preserve named-atom oracles and selected conformer order."""

from types import SimpleNamespace

import pytest
import torch

from tmol.pack.rotamer import create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler
from tmol.pack.rotamer._build_rotamers import (
    _chi4_and_kfo_device_tables,
    _kinforest_device_indices,
)
from tmol.tests.pack.rotamer import test_build_rotamers as existing_tests


@pytest.mark.parametrize("selected", [[9, 0, 5, 2, 7], list(range(9, -1, -1))])
def test_selected_order_matches_named_backbone_atom_oracle(
    selected, default_database, ubq_pdb, torch_device, dun_sampler, monkeypatch
):
    captured = []

    def capture(*args):
        result = create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(*args)
        captured.append((args, result))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(
            existing_tests,
            "create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler",
            capture,
        )
        # This existing jagged-pose test checks both source and destination
        # against six explicitly named backbone atoms, including source offsets.
        existing_tests.test_create_dof_inds_to_copy_from_orig_to_rotamers(
            default_database, ubq_pdb, torch_device, dun_sampler
        )
    ((original, full),) = captured
    args = list(original)
    selection = torch.tensor(selected, dtype=torch.int64, device=torch_device)
    args[5] = selection
    args[7] = args[3][selection].to(torch.int32)
    args[6] = torch.bincount(args[7].long(), minlength=len(args[6])).to(torch.int32)

    def host_copy(*args, **kwargs):
        raise AssertionError("DOF-copy indexing must stay on the tensor device")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", host_copy)
        result = create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(*args)
    for actual, all_rows in zip(result, full):
        expected = all_rows.reshape(10, 6)[selection].reshape(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    pbt = args[0].packed_block_types
    assert _chi4_and_kfo_device_tables(pbt, torch_device)[
        1
    ] is _kinforest_device_indices(pbt, torch_device)


def test_empty_selection_needs_no_pose_annotations(torch_device):
    empty = torch.empty(0, dtype=torch.int64, device=torch_device)
    dst, src = create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(
        None, None, "unused", empty, empty, empty, empty.int(), empty.int(), empty
    )
    assert dst is empty and src is empty


def test_no_retained_regions_produce_an_empty_plan(torch_device):
    pbt = SimpleNamespace(
        mc_fingerprints=SimpleNamespace(atom_mapping=torch.empty(0, 0, 1, 0))
    )
    pose = SimpleNamespace(packed_block_types=pbt)
    row = torch.zeros(1, dtype=torch.int64, device=torch_device)
    dst, src = create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(
        pose, None, "unused", row, row, row, row.int(), row.int(), row
    )
    assert dst.numel() == src.numel() == 0
    assert dst.device == src.device == torch_device
