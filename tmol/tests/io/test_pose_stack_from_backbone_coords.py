"""Tests for backbone-only (N/CA/C/O) side-chain completion."""

import pytest
import torch

from tmol.chemical import three2one
from tmol.io import (
    canonical_form_from_backbone_coords,
    canonical_form_from_pdb,
    canonical_form_from_pose_stack,
    canonical_ordering_for_atomworks,
    packed_block_types_for_atomworks,
    pose_stack_from_backbone_coords,
)
from tmol.io._pose_stack_from_atomworks import _paramdb_for_atomworks
from tmol.tests._torch import requires_cuda
from tmol.io._pose_stack_from_backbone_coords import (
    _BACKBONE_ATOM37_SLOTS,
    _DEFAULT_AA_ORDER,
    _atomworks_tokens_for_aa_order,
    _build_context_for_backbone_coords,
)

_BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
_CANONICALIZE_NAME3 = {"HIS_D": "HIS", "HIS_POS": "HIS", "CYD": "CYS"}


def _backbone_from_pdb(pdb, device):
    """Extract (coords, res_types, chain_id) for a PDB via the canonical form."""
    co = canonical_ordering_for_atomworks()
    cf = canonical_form_from_pdb(co, pdb, device)

    n_poses, max_n_res = cf.res_types.shape[:2]
    coords = torch.full(
        (n_poses, max_n_res, 4, 3), float("nan"), dtype=torch.float32, device=device
    )
    res_types = torch.full((n_poses, max_n_res), -1, dtype=torch.int64, device=device)

    for p in range(n_poses):
        for r in range(max_n_res):
            rt_ind = int(cf.res_types[p, r])
            if rt_ind < 0:
                continue
            name3 = co.restype_io_equiv_classes[rt_ind]
            at_map = co.restypes_atom_index_mapping[name3]
            for slot, at_name in enumerate(_BACKBONE_ATOM_NAMES):
                if at_name in at_map:
                    coords[p, r, slot] = cf.coords[p, r, at_map[at_name]]
            one = three2one(_CANONICALIZE_NAME3.get(name3, name3))
            res_types[p, r] = _DEFAULT_AA_ORDER.index(one)

    return coords, res_types, cf.chain_id.to(torch.int64)


def test_build_context_reuses_atomworks_chemistry(torch_device):
    context = _build_context_for_backbone_coords(torch_device)

    assert context.canonical_ordering is canonical_ordering_for_atomworks()
    assert context.packed_block_types is packed_block_types_for_atomworks(torch_device)
    assert context.parameter_database is _paramdb_for_atomworks()


def test_default_aa_order_matches_atomworks_protein_tokens():
    """AF2 one-letter order is alphabetical by name3, so it is a +1 shift."""
    tokens = _atomworks_tokens_for_aa_order(_DEFAULT_AA_ORDER, torch.device("cpu"))
    assert torch.equal(tokens, torch.arange(1, 21, dtype=torch.int64))


def test_backbone_slots_are_uniform_across_restypes():
    """The (L, 4, 3) reuse of the atomworks tables depends on this."""
    from tmol.io._pose_stack_from_atomworks import (
        ATOMWORKS_ATOM37_NAMES,
        ATOMWORKS_NAME3S,
    )

    for name3 in ATOMWORKS_NAME3S[1:21]:
        row = ATOMWORKS_ATOM37_NAMES[name3]
        slots = tuple(row.index(a) for a in _BACKBONE_ATOM_NAMES)
        assert slots == _BACKBONE_ATOM37_SLOTS


def test_aa_order_rejects_non_canonical_and_repeats():
    with pytest.raises(ValueError, match="not a canonical AA"):
        _atomworks_tokens_for_aa_order("ARNDCQEGHILKMFPSTWYX", torch.device("cpu"))
    with pytest.raises(ValueError, match="must not repeat"):
        _atomworks_tokens_for_aa_order("AARNDCQEGHILKMFPSTWY", torch.device("cpu"))


def test_packing_preserves_the_supplied_backbone(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)

    torch.manual_seed(0)
    pose_stack = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, torch_device
    )

    co = canonical_ordering_for_atomworks()
    round_tripped = canonical_form_from_pose_stack(co, pose_stack)

    n_res = int((res_types[0] >= 0).sum())
    for r in range(n_res):
        name3 = co.restype_io_equiv_classes[int(round_tripped.res_types[0, r])]
        at_map = co.restypes_atom_index_mapping[name3]
        for slot, at_name in enumerate(_BACKBONE_ATOM_NAMES):
            if at_name not in at_map:
                continue
            torch.testing.assert_close(
                round_tripped.coords[0, r, at_map[at_name]],
                coords[0, r, slot],
                rtol=0.0,
                atol=0.0,
            )


def test_sidechains_are_built_and_no_nans_remain(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)
    n_res = int((res_types[0] >= 0).sum())

    torch.manual_seed(0)
    packed = pose_stack_from_backbone_coords(coords, res_types, chain_id, torch_device)
    assert not torch.any(torch.isnan(packed.coords[packed.real_atoms]))
    assert int(packed.real_atoms.sum()) > 4 * n_res


def test_gradients_reach_the_input_backbone_through_packing(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)
    coords = coords[:, :8].clone()
    res_types = res_types[:, :8].clone()
    chain_id = chain_id[:, :8].clone()
    coords.requires_grad_(True)

    torch.manual_seed(0)
    pose_stack = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, torch_device
    )

    torch.nan_to_num(pose_stack.coords).sum().backward()
    assert coords.grad is not None
    finite = torch.isfinite(coords.detach()).all(dim=-1)
    assert torch.count_nonzero(coords.grad[finite]) > 0


def test_completion_none_leaves_sidechains_absent(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)

    unpacked = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, torch_device, sidechain_completion="none"
    )
    torch.manual_seed(0)
    packed = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, torch_device, sidechain_completion="pack"
    )

    real = unpacked.real_atoms
    assert torch.any(torch.isnan(unpacked.coords[real]))
    assert not torch.any(torch.isnan(packed.coords[packed.real_atoms]))


def test_completion_none_is_differentiable(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)
    coords = coords[:, :8].clone().requires_grad_(True)

    pose_stack = pose_stack_from_backbone_coords(
        coords,
        res_types[:, :8],
        chain_id[:, :8],
        torch_device,
        sidechain_completion="none",
    )
    torch.nan_to_num(pose_stack.coords).sum().backward()
    assert coords.grad is not None
    assert torch.count_nonzero(coords.grad) > 0


def test_invalid_completion_policy_is_rejected(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)
    with pytest.raises(ValueError, match="sidechain_completion"):
        pose_stack_from_backbone_coords(
            coords, res_types, chain_id, torch_device, sidechain_completion="ideal"
        )


def test_repeated_builds_are_stable_and_reuse_setup(ubq_pdb, torch_device):
    """Packed side chains are deliberately not compared: the CPU annealer draws
    from libc rand() (pack/compiled/compiled.cpu.cpp), which torch.manual_seed
    does not control. That reaches topology too, since the HIS tautomer is
    resolved from the packed side chain.
    """
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)

    first = pose_stack_from_backbone_coords(coords, res_types, chain_id, torch_device)
    context = _build_context_for_backbone_coords(torch_device)
    score_function = context._packing_score_function
    dunbrack_sampler = context._dunbrack_sampler

    second = pose_stack_from_backbone_coords(coords, res_types, chain_id, torch_device)

    assert _build_context_for_backbone_coords(torch_device) is context
    assert context._packing_score_function is score_function
    assert context._dunbrack_sampler is dunbrack_sampler

    assert first.coords.shape == second.coords.shape
    assert torch.equal(first.block_coord_offset, second.block_coord_offset)
    assert torch.equal(first.real_atoms, second.real_atoms)

    co = canonical_ordering_for_atomworks()
    cf_first = canonical_form_from_pose_stack(co, first)
    cf_second = canonical_form_from_pose_stack(co, second)
    n_res = int((res_types[0] >= 0).sum())
    for r in range(n_res):
        for cf, other in ((cf_first, cf_second), (cf_second, cf_first)):
            name3 = co.restype_io_equiv_classes[int(cf.res_types[0, r])]
            other3 = co.restype_io_equiv_classes[int(other.res_types[0, r])]
            at_map = co.restypes_atom_index_mapping[name3]
            other_map = co.restypes_atom_index_mapping[other3]
            for at_name in _BACKBONE_ATOM_NAMES:
                if at_name not in at_map or at_name not in other_map:
                    continue
                assert torch.equal(
                    cf.coords[0, r, at_map[at_name]],
                    other.coords[0, r, other_map[at_name]],
                )


@requires_cuda
def test_cpu_and_cuda_agree(ubq_pdb):
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, cpu)

    cf_cpu = canonical_form_from_backbone_coords(coords, res_types, chain_id)
    cf_cuda = canonical_form_from_backbone_coords(
        coords.to(cuda), res_types.to(cuda), chain_id.to(cuda)
    )
    assert torch.equal(torch.isnan(cf_cpu.coords), torch.isnan(cf_cuda.coords).cpu())
    finite = torch.isfinite(cf_cpu.coords)
    torch.testing.assert_close(
        cf_cpu.coords[finite], cf_cuda.coords.cpu()[finite], rtol=0.0, atol=0.0
    )
    assert torch.equal(cf_cpu.res_types, cf_cuda.res_types.cpu())

    ps_cpu = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, cpu, sidechain_completion="none"
    )
    ps_cuda = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, cuda, sidechain_completion="none"
    )
    assert ps_cpu.coords.shape == ps_cuda.coords.shape
    mask = ps_cpu.real_atoms & torch.isfinite(ps_cpu.coords).all(dim=-1)
    torch.testing.assert_close(
        ps_cpu.coords[mask], ps_cuda.coords.cpu()[mask], rtol=1e-5, atol=1e-4
    )


def test_single_pose_input_is_unsqueezed(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)

    batched = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, torch_device, sidechain_completion="none"
    )
    single = pose_stack_from_backbone_coords(
        coords[0],
        res_types[0],
        chain_id[0],
        torch_device,
        sidechain_completion="none",
    )

    assert single.n_poses == batched.n_poses == 1
    assert torch.equal(single.block_type_ind, batched.block_type_ind)
    finite = torch.isfinite(single.coords) & torch.isfinite(batched.coords)
    assert torch.equal(single.coords[finite], batched.coords[finite])


def test_output_is_on_the_requested_device(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch.device("cpu"))
    pose_stack = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, torch_device, sidechain_completion="none"
    )
    assert pose_stack.coords.device.type == torch_device.type


def test_padding_positions_are_excluded(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)
    n_keep = 5
    res_types = res_types.clone()
    res_types[0, n_keep:] = -1

    pose_stack = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, torch_device, sidechain_completion="none"
    )
    assert int((pose_stack.block_type_ind[0] >= 0).sum()) == n_keep


def test_two_chains_are_separated(ubq_pdb, torch_device):
    coords, res_types, chain_id = _backbone_from_pdb(ubq_pdb, torch_device)
    n_res = int((res_types[0] >= 0).sum())
    chain_id = chain_id.clone()
    chain_id[0, n_res // 2 : n_res] = 1

    pose_stack = pose_stack_from_backbone_coords(
        coords, res_types, chain_id, torch_device, sidechain_completion="none"
    )
    assert int((pose_stack.block_type_ind[0] >= 0).sum()) == n_res
    assert pose_stack.n_poses == 1
    assert int(pose_stack.chain_id[0, :n_res].max()) == 1


def test_shape_validation(torch_device):
    rt = torch.zeros((1, 4), dtype=torch.int64, device=torch_device)
    ci = torch.zeros((1, 4), dtype=torch.int64, device=torch_device)

    bad = torch.zeros((1, 4, 3, 3), dtype=torch.float32, device=torch_device)
    with pytest.raises(AssertionError, match="N, CA, C, O"):
        canonical_form_from_backbone_coords(bad, rt, ci)

    good = torch.zeros((1, 4, 4, 3), dtype=torch.float32, device=torch_device)
    with pytest.raises(AssertionError, match="res_types must be"):
        canonical_form_from_backbone_coords(good, rt[:, :3], ci)


def test_out_of_range_res_types_are_rejected(torch_device):
    coords = torch.zeros((1, 2, 4, 3), dtype=torch.float32, device=torch_device)
    rt = torch.tensor([[0, 20]], dtype=torch.int64, device=torch_device)
    ci = torch.zeros((1, 2), dtype=torch.int64, device=torch_device)
    with pytest.raises(ValueError, match="must be in range"):
        canonical_form_from_backbone_coords(coords, rt, ci)
