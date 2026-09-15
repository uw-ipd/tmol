import pytest
import torch


def _reference_get_children(parents):
    """Return the deterministic ordering produced by the legacy bubble sort."""
    parents = parents.cpu()
    n_atoms = parents.shape[0]
    n_children = torch.zeros(n_atoms + 1, dtype=torch.int32)
    is_atom_jump = torch.zeros(n_atoms, dtype=torch.bool)

    children = [[] for _ in range(n_atoms)]
    for child, parent in enumerate(parents.tolist()):
        if child == parent:
            continue
        n_children[parent] += 1
        children[parent].append(child)
        is_atom_jump[child] = parent == 0

    child_list_span = torch.zeros(n_atoms + 1, dtype=torch.int32)
    child_list_span[1:] = torch.cumsum(n_children[:-1], dim=0)
    child_list = torch.full((n_atoms,), -1, dtype=torch.int32)
    for parent, parent_children in enumerate(children):
        start = child_list_span[parent]
        child_list[start : start + len(parent_children)] = torch.tensor(
            sorted(parent_children), dtype=torch.int32
        )

    return n_children, child_list_span, child_list, is_atom_jump


@pytest.mark.parametrize(
    "parents",
    [
        pytest.param(
            [0, 0, 0, 1, 1, 2, 2, 2, 5, 5, 8, 8, 8, 8], id="mixed-tree"
        ),
        pytest.param([0] * 4097, id="high-fanout-root"),
    ],
)
def test_get_children_matches_legacy_sorted_order(torch_device, parents):
    from tmol.kinematics.compiled.compiled_ops import get_children

    parents = torch.tensor(parents, dtype=torch.int32, device=torch_device)
    n_atoms = parents.shape[0]

    # A single synthetic block is sufficient: root children are jumps and all
    # other parent/child pairs are within the block and therefore non-jumps.
    pose_stack_block_type = torch.zeros(
        (1, 1), dtype=torch.int32, device=torch_device
    )
    block_in_and_first_out = torch.zeros(
        (1, 1, 2), dtype=torch.int32, device=torch_device
    )
    kfo_2_orig_mapping = torch.zeros(
        (n_atoms, 3), dtype=torch.int32, device=torch_device
    )
    kfo_2_orig_mapping[0] = -1
    block_type_n_conn = torch.ones(1, dtype=torch.int32, device=torch_device)

    expected = _reference_get_children(parents)
    for _ in range(3):
        actual = get_children(
            pose_stack_block_type,
            block_in_and_first_out,
            kfo_2_orig_mapping,
            parents,
            block_type_n_conn,
        )
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor.cpu(), expected_tensor)


def test_get_children_sorts_across_jump_and_non_jump_children(torch_device):
    from tmol.kinematics.compiled.compiled_ops import get_children

    # Atom 1 has non-jump children 2 and 4 in block 0, plus jump child 3 in
    # block 1. The legacy second bubble-sort loop sorted the complete span, so
    # the final order is [2, 3, 4], not jump-first [3, 2, 4].
    pose_stack_block_type = torch.zeros(
        (1, 2), dtype=torch.int32, device=torch_device
    )
    block_in_and_first_out = torch.zeros(
        (1, 2, 2), dtype=torch.int32, device=torch_device
    )
    block_in_and_first_out[0, 1, 0] = 1
    kfo_2_orig_mapping = torch.tensor(
        [[-1, -1, -1], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 2]],
        dtype=torch.int32,
        device=torch_device,
    )
    parents = torch.tensor(
        [0, 0, 1, 1, 1], dtype=torch.int32, device=torch_device
    )
    block_type_n_conn = torch.ones(1, dtype=torch.int32, device=torch_device)

    actual = get_children(
        pose_stack_block_type,
        block_in_and_first_out,
        kfo_2_orig_mapping,
        parents,
        block_type_n_conn,
    )
    expected = (
        torch.tensor([1, 3, 0, 0, 0, 0], dtype=torch.int32),
        torch.tensor([0, 1, 4, 4, 4, 4], dtype=torch.int32),
        torch.tensor([1, 2, 3, 4, -1], dtype=torch.int32),
        torch.tensor([False, True, False, True, False]),
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor.cpu(), expected_tensor)
