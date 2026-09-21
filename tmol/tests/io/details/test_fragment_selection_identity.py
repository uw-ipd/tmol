"""Declared fragment bonds and alternative states retain their identity."""

import attr
import pytest
import torch

from tmol.pose import PackedBlockTypes
from tmol.io.details import _select_from_canonical as selection
from tmol.tests.ligand.test_fragmented_ligand_scoring import (
    _load_fixture,
    _annotate_at_bridge,
    _build,
)


@pytest.fixture
def fragment_case(torch_device):
    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    pose, context, _ = _build(annotated, params_path, torch_device, fragmented=True)
    return pose, context.canonical_ordering


def test_declared_connections_between_repeated_fragment_types(fragment_case):
    pose, co = fragment_case
    pbt = pose.packed_block_types
    indices = [
        i for i, bt in enumerate(pbt.active_block_types) if bt.is_ligand_fragment
    ]
    assert len(indices) == 2
    first, second = [pbt.active_block_types[i] for i in indices]
    assert first.base_name == second.base_name
    canonical = [
        co.restypes_ordered_atom_names[bt.io_equiv_class].index(bt.connections[0].atom)
        for bt in (first, second)
    ]
    blocks = torch.tensor(
        [indices * 2, indices + [-1, -1]], dtype=torch.int64, device=pbt.device
    )
    # Two copies share type names. The caller explicitly connects crossed
    # block instances, plus a separate jagged pose; every connection is used once.
    bonds = torch.tensor(
        [
            [0, 0, canonical[0], 3, canonical[1]],
            [0, 2, canonical[0], 1, canonical[1]],
            [1, 0, canonical[0], 1, canonical[1]],
        ],
        dtype=torch.int64,
        device=pbt.device,
    )
    original = blocks.clone()
    result = selection._apply_conjugated_variants(co, pbt, blocks, bonds)
    assert result == [(0, 0, 0, 3, 0), (0, 2, 0, 1, 0), (1, 0, 0, 1, 0)]
    torch.testing.assert_close(blocks, original)

    from tmol.io import canonical_form_from_pose_stack, pose_stack_from_canonical_form

    form = canonical_form_from_pose_stack(co, pose)
    source_blocks = [pose.block_type_ind[0].tolist().index(i) for i in indices]
    coords = form.coords[:, source_blocks * 2].repeat(2, 1, 1, 1)
    res_types = form.res_types[:, source_blocks * 2].repeat(2, 1).int()
    chains = torch.arange(4, dtype=torch.int32, device=pbt.device)[None, :].repeat(2, 1)
    coords[1, 2:] = torch.nan
    res_types[1, 2:] = -1
    chains[1, 2:] = -1
    rebuilt = pose_stack_from_canonical_form(
        co,
        pbt,
        chains,
        res_types,
        coords,
        res_labels=None,
        res_ins_codes=None,
        chain_labels=None,
        covalent_bonds=bonds,
        find_additional_disulfides=False,
        find_additional_cyclic_closures=False,
    )
    expected = torch.full_like(rebuilt.inter_residue_connections, -1)
    for pi, r1, c1, r2, c2 in result:
        expected[pi, r1, c1] = expected.new_tensor([r2, c2])
        expected[pi, r2, c2] = expected.new_tensor([r1, c1])
    torch.testing.assert_close(rebuilt.inter_residue_connections, expected)
    torch.testing.assert_close(rebuilt.block_type_ind.long(), original)
    assert torch.isfinite(rebuilt.coords).all()


@pytest.mark.parametrize("reverse", [False, True])
def test_all_fragment_states_remain_selection_candidates(fragment_case, reverse):
    pose, co = fragment_case
    source = pose.packed_block_types
    first = next(bt for bt in source.active_block_types if bt.is_ligand_fragment)
    second = attr.evolve(first, name=first.name + ":alternate")
    types = [first, second]
    if reverse:
        types.reverse()
    pbt = PackedBlockTypes.from_restype_list(
        source.chem_db, source.restype_set, types, source.device
    )
    selection._annotate_packed_block_types_w_canonical_res_order(co, pbt)
    annotation = pbt.canonical_ordering_annotation
    candidates = annotation.var_combo_candidate_bt_index[
        annotation.var_combo_is_real_candidate
    ].tolist()
    assert sorted(set(candidates)) == [0, 1]
