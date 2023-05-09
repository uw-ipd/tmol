import pytest
import torch
from tmol.io.canonical_ordering import canonical_form_from_pdb_lines
from tmol.io.basic_resolution import pose_stack_from_canonical_form


@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form")
def test_build_pose_stack_from_canonical_form_ubq_benchmark(
    benchmark, torch_device, ubq_pdb
):
    ch_beg, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(ubq_pdb)

    ch_beg = torch.tensor(ch_beg, device=torch_device)
    can_rts = torch.tensor(can_rts, device=torch_device)
    coords = torch.tensor(coords, device=torch_device)
    at_is_pres = torch.tensor(at_is_pres, device=torch_device)

    # warmup
    pose_stack_from_canonical_form(ch_beg, can_rts, coords, at_is_pres)

    @benchmark
    def create_pose_stack():
        pose_stack = pose_stack_from_canonical_form(ch_beg, can_rts, coords, at_is_pres)
        return pose_stack


@pytest.mark.benchmark(group="setup_pose_stack_from_canonical_form")
def test_build_pose_stack_from_canonical_form_pertuzumab_benchmark(
    benchmark, torch_device, pertuzumab_lines
):
    ch_beg, can_rts, coords, at_is_pres = canonical_form_from_pdb_lines(
        pertuzumab_lines
    )

    ch_beg = torch.tensor(ch_beg, device=torch_device)
    can_rts = torch.tensor(can_rts, device=torch_device)
    coords = torch.tensor(coords, device=torch_device)
    at_is_pres = torch.tensor(at_is_pres, device=torch_device)

    # warmup
    pose_stack_from_canonical_form(ch_beg, can_rts, coords, at_is_pres)

    @benchmark
    def create_pose_stack():
        pose_stack = pose_stack_from_canonical_form(ch_beg, can_rts, coords, at_is_pres)
        return pose_stack
