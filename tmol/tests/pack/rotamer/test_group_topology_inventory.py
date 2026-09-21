"""Grouping must retain every covalent constraint, including non-tree edges."""

from types import SimpleNamespace

import pytest
import torch

from tmol.pose._conjugated_groups import find_conjugated_groups


def _pose_graph(n_blocks, conjugations, other_bonds, polymers, device):
    edges = [(a, b, True) for a, b in conjugations] + [
        (a, b, False) for a, b in other_bonds
    ]
    degree = [0] * n_blocks
    ports = []
    for a, b, conjugated in edges:
        ac = degree[a]
        degree[a] += 1
        bc = degree[b]
        degree[b] += 1
        ports.append((a, ac, b, bc, conjugated))
    n_conn = [n + int(b in polymers) for b, n in enumerate(degree)]
    width = max(n_conn)
    connections = torch.full(
        (2, n_blocks, width, 2), -1, dtype=torch.int32, device=device
    )
    conjugated = torch.zeros((n_blocks, width), dtype=torch.bool, device=device)
    for a, ac, b, bc, is_conjugated in ports:
        connections[:, a, ac] = torch.tensor([b, bc], device=device)
        connections[:, b, bc] = torch.tensor([a, ac], device=device)
        conjugated[a, ac] = conjugated[b, bc] = is_conjugated
    pbt = SimpleNamespace(
        active_block_types=[
            SimpleNamespace(
                properties=SimpleNamespace(
                    polymer=SimpleNamespace(is_polymer=b in polymers)
                )
            )
            for b in range(n_blocks)
        ],
        conjugation_conn=conjugated,
        n_conn=torch.tensor(n_conn, device=device),
        up_conn_inds=torch.tensor(
            [degree[b] if b in polymers else -1 for b in range(n_blocks)], device=device
        ),
        down_conn_inds=torch.full((n_blocks,), -1, device=device),
    )
    pose = SimpleNamespace(
        packed_block_types=pbt,
        n_poses=2,
        max_n_blocks=n_blocks,
        block_type_ind=torch.arange(n_blocks, device=device).repeat(2, 1),
        inter_residue_connections=connections,
    )
    return pose, ports


@pytest.mark.parametrize(
    "n_blocks,conjugations,other,polymers",
    [
        (4, [(0, 1), (1, 2), (1, 3)], [], {0}),
        (3, [(0, 1), (1, 2), (2, 0)], [], {0}),
        (2, [(0, 1), (0, 1)], [], {0}),
        (3, [(0, 1), (1, 2)], [(0, 2)], {0, 2}),
        (5, [(0, 1), (1, 2)], [(2, 3), (0, 4)], {0, 2}),
        (3, [(0, 1), (1, 2)], [], {0, 2}),
        (1, [(0, 0)], [], {0}),
    ],
)
def test_group_records_all_internal_and_external_bonds(
    n_blocks, conjugations, other, polymers, torch_device
):
    pose, ports = _pose_graph(n_blocks, conjugations, other, polymers, torch_device)
    groups = find_conjugated_groups(pose)
    assert len(groups) == 2
    for pose_index, group in enumerate(groups):
        assert group.pose == pose_index
        internal, external = set(), set()
        for a, ac, b, bc, _ in ports:
            if a in group.blocks and b in group.blocks:
                internal.add(tuple(sorted(((a, ac), (b, bc)))))
            elif a in group.blocks:
                external.add((a, ac, b, bc))
            elif b in group.blocks:
                external.add((b, bc, a, ac))
        actual = {
            tuple(sorted(((group.blocks[a], ac), (group.blocks[b], bc))))
            for a, ac, b, bc in group.links
        }
        assert actual == internal
        assert len(group.links) == len(internal)
        assert {
            (group.blocks[a], ac, b, bc) for a, ac, b, bc in group.external_links
        } == external
