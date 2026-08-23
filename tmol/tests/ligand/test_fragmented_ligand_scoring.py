"""Parity tests for user-annotated connected ligand fragments."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io
import numpy as np
import pytest
import torch

from tmol.ligand import (
    FRAGMENT_ID_ANNOTATION,
    load_params_file,
    unsplit_pose_stack,
)
from tmol.ligand._fragmentation import (
    _unsplit_build_coords,
    _unsplit_chain_and_pdb,
    _unsplit_connections,
    _unsplit_group_entries,
    _unsplit_old_to_new,
    _unsplit_per_pose_blocks,
)
from tmol.pose import SplitBlockEntry, SplitBlockMapping

DATA_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
TARGET = "ace"
LIGAND_NAME = "LG1"
MULTI_CUTS = {
    "ace": (("C1", "C2"), ("C3", "C4"), ("C5", "C9"), ("C11", "C12"), ("C15", "C16")),
    "egfr": (("C5", "N1"), ("C10", "C11"), ("C14", "C15")),
}


def _load_fixture(target=TARGET):
    structure = biotite.structure.io.load_structure(
        str(DATA_DIR / f"{target}.tmol.nomin.cif"),
        model=1,
        include_bonds=True,
    )
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]
    params_path = DATA_DIR / f"{target}.xtal-lig.mmff94.tmol"
    preparation = load_params_file(params_path)[0]
    return structure, params_path, preparation


def _components_without_bonds(restype, removed):
    adjacency = {atom.name: set() for atom in restype.atoms}
    for bond in restype.bonds:
        edge = frozenset(bond[:2])
        if edge in removed:
            continue
        a, b = bond[:2]
        adjacency[a].add(b)
        adjacency[b].add(a)
    components = []
    unseen = set(adjacency)
    while unseen:
        queue = deque([next(iter(unseen))])
        component = set()
        while queue:
            atom = queue.popleft()
            if atom in component:
                continue
            component.add(atom)
            queue.extend(adjacency[atom] - component)
        unseen -= component
        components.append(component)
    return components


def _annotate_at_bridge(structure, preparation):
    result = structure.copy()
    ligand_mask = result.res_name == LIGAND_NAME
    input_ligand_names = set(str(name) for name in result.atom_name[ligand_mask])
    atom_by_name = {atom.name: atom for atom in preparation.residue_type.atoms}
    selected_components = None
    for bond in preparation.residue_type.bonds:
        components = _components_without_bonds(
            preparation.residue_type, {frozenset(bond[:2])}
        )
        if len(components) != 2:
            continue
        input_counts = [
            sum(name in input_ligand_names for name in component)
            for component in components
        ]
        heavy_counts = [
            sum(
                not atom_by_name[name].atom_type.upper().startswith("H")
                for name in component
            )
            for component in components
        ]
        if min(input_counts) > 0 and min(heavy_counts) >= 3:
            selected_components = components
            break
    assert selected_components is not None, "fixture ligand has no suitable bridge cut"

    fragment_ids = np.zeros(result.array_length(), dtype=np.int32)
    first_component = selected_components[0]
    for atom_index in np.flatnonzero(ligand_mask):
        fragment_ids[atom_index] = (
            1 if str(result.atom_name[atom_index]) in first_component else 2
        )
    result.set_annotation(FRAGMENT_ID_ANNOTATION, fragment_ids)
    return result


def _annotate_at_cuts(structure, preparation, cuts):
    result = structure.copy()
    ligand_mask = result.res_name == LIGAND_NAME
    restype = preparation.residue_type
    atom_order = {atom.name: index for index, atom in enumerate(restype.atoms)}
    fragments = _components_without_bonds(restype, {frozenset(cut) for cut in cuts})
    fragments.sort(key=lambda fragment: min(atom_order[name] for name in fragment))
    fragment_for_atom = {
        name: fragment_id
        for fragment_id, fragment in enumerate(fragments, start=1)
        for name in fragment
    }
    fragment_ids = np.zeros(result.array_length(), dtype=np.int32)
    for atom_index in np.flatnonzero(ligand_mask):
        fragment_ids[atom_index] = fragment_for_atom[str(result.atom_name[atom_index])]
    result.set_annotation(FRAGMENT_ID_ANNOTATION, fragment_ids)
    return result


def _build(structure, params_path, torch_device, *, fragmented):
    from tmol.database import ParameterDatabase
    from tmol.io import pose_stack_from_biotite

    pose, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=[str(params_path)],
        no_optH=True,
        sample_proton_chi=False,
        param_db=ParameterDatabase.get_default(),
        return_context=True,
    )
    if fragmented:
        return pose, context, pose.split_block_mapping
    return pose, context, None


def _fragment_block_mask(pose, mapping):
    mask = torch.zeros_like(pose.block_type_ind, dtype=torch.bool)
    for entry in mapping.entries:
        mask[entry.pose_ind, entry.block_ind] = True
    return mask


def test_fragment_definition_connections_icoors_and_mapping(torch_device):
    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    pose, _, mapping = _build(annotated, params_path, torch_device, fragmented=True)

    pbt = pose.packed_block_types
    frag_names = sorted(
        pbt.active_block_types[int(pose.block_type_ind[0, e.block_ind])].name
        for e in mapping.entries
        if e.pose_ind == 0
    )
    assert frag_names == ["LG1.1", "LG1.2"]
    # Find the cut-bond connection between the two fragment blocks.
    # After apply_fragment_connections the connection is encoded in inter_residue_connections.
    block_a, block_b = sorted(e.block_ind for e in mapping.entries if e.pose_ind == 0)
    # block_a connects to block_b via connection 0 (the first cut-bond connection)
    block_a_conn = pose.inter_residue_connections[0, block_a]
    connected = [
        (int(block_a_conn[c, 0]), c)
        for c in range(block_a_conn.shape[0])
        if int(block_a_conn[c, 0]) == block_b
    ]
    assert len(connected) == 1
    conn_a = (
        pbt.active_block_types[int(pose.block_type_ind[0, block_a])]
        .connections[connected[0][1]]
        .name
    )
    block_b_conn = pose.inter_residue_connections[0, block_b]
    connected_b = [
        (int(block_b_conn[c, 0]), c)
        for c in range(block_b_conn.shape[0])
        if int(block_b_conn[c, 0]) == block_a
    ]
    conn_b = (
        pbt.active_block_types[int(pose.block_type_ind[0, block_b])]
        .connections[connected_b[0][1]]
        .name
    )
    assert tuple(pose.inter_residue_connections[0, block_a, 0].tolist()) == (
        block_b,
        0,
    )
    assert tuple(pose.inter_residue_connections[0, block_b, 0].tolist()) == (
        block_a,
        0,
    )
    assert conn_a.startswith("conn_") and conn_b.startswith("conn_")
    for block_index in (block_a, block_b):
        block_type = pose.packed_block_types.active_block_types[
            int(pose.block_type_ind[0, block_index])
        ]
        connection_names = {connection.name for connection in block_type.connections}
        icoor_names = {icoor.name for icoor in block_type.icoors}
        assert block_type.is_ligand_fragment
        assert not block_type.hydrogens_regenerated
        assert connection_names <= icoor_names


def test_fragmented_ligand_export_restores_original_residue(torch_device):
    import attr

    from tmol.io import biotite_from_pose_stack
    from tmol.io import atom_records_from_pose_stack
    from tmol.ligand import recombine_fragmented_ligands

    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    pose, context, mapping = _build(
        annotated, params_path, torch_device, fragmented=True
    )
    pose = attr.evolve(pose, coords=pose.coords.clone())
    assert pose.split_block_mapping is mapping

    split = biotite_from_pose_stack(
        pose, context.canonical_ordering, merge_fragments=False
    )
    restored = recombine_fragmented_ligands(split, pose)
    merged = biotite_from_pose_stack(pose, context.canonical_ordering)

    np.testing.assert_array_equal(merged.res_name, restored.res_name)
    np.testing.assert_array_equal(merged.res_id, restored.res_id)
    assert not np.any(np.char.startswith(merged.res_name, f"{LIGAND_NAME}."))
    pbt = pose.packed_block_types
    expected_frag_res_ids = {
        (
            int(pose.pdb_info.residue_labels[e.pose_ind, e.block_ind]),
            pbt.active_block_types[
                int(pose.block_type_ind[e.pose_ind, e.block_ind])
            ].name,
        )
        for e in mapping.entries
        if e.pose_ind == 0
    }
    assert {
        (int(res_id), str(res_name))
        for res_id, res_name in zip(split.res_id, split.res_name)
        if str(res_name).startswith(f"{LIGAND_NAME}.")
    } == expected_frag_res_ids

    split_records = atom_records_from_pose_stack(pose, merge_fragments=False)
    merged_records = atom_records_from_pose_stack(pose)
    fragment_residue_labels = {
        int(pose.pdb_info.residue_labels[e.pose_ind, e.block_ind])
        for e in mapping.entries
        if e.pose_ind == 0
    }
    assert fragment_residue_labels <= set(split_records["resi"])
    assert fragment_residue_labels.isdisjoint(set(merged_records["resi"]))
    assert {e.orig_residue_label for e in mapping.entries if e.pose_ind == 0} <= set(
        merged_records["resi"]
    )


def test_fragmentation_uses_ligand_already_in_parameter_database(torch_device):
    from tmol.io import pose_stack_from_biotite

    structure, params_path, preparation = _load_fixture()
    _, whole_context, _ = _build(structure, params_path, torch_device, fragmented=False)
    annotated = _annotate_at_bridge(structure, preparation)

    pose = pose_stack_from_biotite(
        annotated,
        torch_device,
        param_db=whole_context.parameter_database,
        prepare_ligands=True,
        no_optH=True,
        sample_proton_chi=False,
    )

    pbt = pose.packed_block_types
    frag_names = sorted(
        pbt.active_block_types[int(pose.block_type_ind[e.pose_ind, e.block_ind])].name
        for e in pose.split_block_mapping.entries
        if e.pose_ind == 0
    )
    assert frag_names == ["LG1.1", "LG1.2"]


def test_fragment_interactions_validate_inputs(torch_device):
    from tmol.score import calculate_fragment_interactions

    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    pose, _, mapping = _build(annotated, params_path, torch_device, fragmented=True)

    with pytest.raises(TypeError, match="boolean"):
        calculate_fragment_interactions(
            pose,
            torch.zeros_like(pose.block_type_ind),
            sfxn=None,
        )

    partner_mask = ~_fragment_block_mask(pose, mapping) & (pose.block_type_ind >= 0)
    with pytest.raises(ValueError, match="sfxn is required"):
        calculate_fragment_interactions(
            pose,
            partner_mask,
            sfxn=None,
        )


def test_duplicate_ligand_names_require_same_fragment_layout():
    from tmol.ligand import LigandPreparationError, prepare_ligands

    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    ligand = annotated[annotated.res_name == LIGAND_NAME]
    duplicate = ligand.copy()
    duplicate.res_id[:] = ligand.res_id[0] + 1
    duplicate.tmol_fragment_id[:] = 3 - duplicate.tmol_fragment_id

    with pytest.raises(LigandPreparationError, match="same name"):
        prepare_ligands(
            ligand + duplicate,
            params_files=[str(params_path)],
            return_fragment_definitions=True,
        )


def test_prepare_ligands_rejects_too_small_fragment():
    from tmol.ligand import LigandPreparationError, prepare_ligands

    structure, params_path, _ = _load_fixture("egfr")
    annotated = structure.copy()
    ligand_mask = annotated.res_name == LIGAND_NAME
    fragment_ids = np.zeros(annotated.array_length(), dtype=np.int32)
    fragment_ids[ligand_mask] = 2
    fragment_ids[ligand_mask & np.isin(annotated.atom_name, ["C1", "C2", "H1"])] = 1
    annotated.set_annotation(FRAGMENT_ID_ANNOTATION, fragment_ids)

    with pytest.raises(
        LigandPreparationError,
        match=f"invalid {FRAGMENT_ID_ANNOTATION} annotation.*at least 3 heavy atoms",
    ):
        prepare_ligands(
            annotated,
            params_files=[str(params_path)],
            return_fragment_definitions=True,
        )


def test_prepare_ligands_rejects_unassigned_fragment_atoms_public_path():
    from tmol.ligand import LigandPreparationError, prepare_ligands

    structure, params_path, _ = _load_fixture("egfr")
    annotated = structure.copy()
    ligand_mask = annotated.res_name == LIGAND_NAME
    fragment_ids = np.zeros(annotated.array_length(), dtype=np.int32)
    fragment_ids[ligand_mask] = 1
    fragment_ids[ligand_mask & np.isin(annotated.atom_name, ["C1", "C2", "C3"])] = 2
    fragment_ids[ligand_mask & (annotated.atom_name == "H1")] = 0
    annotated.set_annotation(FRAGMENT_ID_ANNOTATION, fragment_ids)

    with pytest.raises(
        LigandPreparationError,
        match=f"invalid {FRAGMENT_ID_ANNOTATION} annotation.*positive",
    ):
        prepare_ligands(
            annotated,
            params_files=[str(params_path)],
            return_fragment_definitions=True,
        )


def test_fragment_validation_rejects_unsupported_layouts():
    from tmol.ligand import (
        _fragment_atom_tree,
        _validate_bonded_cut_layout,
        _validate_scoring_cut_layout,
    )

    with pytest.raises(ValueError, match="one connected component"):
        _fragment_atom_tree(
            ("A", "B"),
            (),
            {"A": np.zeros(3), "B": np.ones(3)},
        )

    _, _, preparation = _load_fixture()
    with pytest.raises(ValueError, match="hbond/lk_ball acceptor geometry"):
        _validate_scoring_cut_layout(preparation.residue_type, (("O1", "C1"),))

    adjacency = {
        "A": ("B",),
        "B": ("A", "C"),
        "C": ("B", "D"),
        "D": ("C",),
    }
    with pytest.raises(ValueError, match="torsions spanning three blocks"):
        _validate_bonded_cut_layout("LG1", adjacency, (("A", "B"), ("C", "D")))


def test_fragment_mapping_is_stable_for_atom_array_stack(torch_device):
    from tmol.io import pose_stack_from_biotite

    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    _, context, _ = _build(annotated, params_path, torch_device, fragmented=True)
    stack = struc.stack([annotated, annotated])
    pose = pose_stack_from_biotite(
        stack,
        torch_device,
        context=context,
        no_optH=True,
    )
    mapping = pose.split_block_mapping
    assert pose.n_poses == 2
    assert len(mapping.entries) == 4  # 2 poses × 2 fragments
    # Each distinct fragment block index appears in both poses
    pose0_block_inds = {e.block_ind for e in mapping.entries if e.pose_ind == 0}
    assert len(pose0_block_inds) == 2
    for block_ind in pose0_block_inds:
        assert {e.pose_ind for e in mapping.entries if e.block_ind == block_ind} == {
            0,
            1,
        }
    assert pose.clone().split_block_mapping is mapping
    split_mapping = pose.split(1).split_block_mapping
    assert len(split_mapping.entries) == 2
    assert {e.pose_ind for e in split_mapping.entries} == {0}
    assert len({e.block_ind for e in split_mapping.entries}) == 2


def test_fragmented_ligand_minimize_and_pack_e2e():
    from tmol import run_cart_min
    from tmol.ops import (
        build_coord_mask_for_mask_and_interacting_atoms,
        calculate_block_pair_ddg,
    )
    from tmol.score import (
        beta2016_score_function,
        calculate_fragment_interactions,
    )

    torch_device = torch.device("cpu")
    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    pose, context, mapping = _build(
        annotated, params_path, torch_device, fragmented=True
    )
    fragment_mask = _fragment_block_mask(pose, mapping)
    sfxn = beta2016_score_function(torch_device, param_db=context.parameter_database)

    wpsm = sfxn.render_whole_pose_scoring_module(pose)
    start_score = wpsm(pose.coords)
    coord_mask = build_coord_mask_for_mask_and_interacting_atoms(pose, fragment_mask)
    minimized = run_cart_min(
        pose.clone(), sfxn, coord_mask, optimizer_kwargs={"max_iter": 1}
    )
    end_score = wpsm(minimized.coords)
    assert torch.all(end_score < start_score)

    fragment_partner = ~fragment_mask & (pose.block_type_ind >= 0)
    packed_ddg, packed = calculate_block_pair_ddg(
        pose.clone(),
        fragment_mask,
        fragment_partner,
        sfxn=sfxn,
        sum_terms=True,
        minimize=False,
        pack=True,
        return_pose_stack=True,
    )
    assert torch.isfinite(packed_ddg).all()
    assert packed.split_block_mapping is not None

    packed_fragment_mask = _fragment_block_mask(packed, mapping)
    packed_partner = ~packed_fragment_mask & (packed.block_type_ind >= 0)
    attributed = calculate_fragment_interactions(
        packed,
        packed_partner,
        sfxn=sfxn,
        sum_terms=True,
    )
    torch.testing.assert_close(
        attributed.scores.sum(dim=1), packed_ddg, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    ("target", "fragmentation"),
    [
        ("ace", "bridge"),
        ("cox1", "bridge"),
        ("hsp90", "bridge"),
        ("ace", "multi"),
    ],
)
def test_fragmented_ligand_ddg_and_total_pose_parity(target, fragmentation):
    from tmol.ops import calculate_block_pair_ddg
    from tmol.score import (
        beta2016_score_function,
        calculate_fragment_interactions,
    )

    torch_device = torch.device("cpu")
    structure, params_path, preparation = _load_fixture(target)
    annotated = (
        _annotate_at_bridge(structure, preparation)
        if fragmentation == "bridge"
        else _annotate_at_cuts(structure, preparation, MULTI_CUTS[target])
    )
    whole, whole_context, _ = _build(
        structure, params_path, torch_device, fragmented=False
    )
    fragmented, fragment_context, mapping = _build(
        annotated, params_path, torch_device, fragmented=True
    )
    if fragmentation == "multi":
        pose0_entries = [e for e in mapping.entries if e.pose_ind == 0]
        assert len(pose0_entries) >= 3
        pbt_frag = fragmented.packed_block_types
        atom_by_name = {atom.name: atom for atom in preparation.residue_type.atoms}
        assert (
            min(
                sum(
                    not atom_by_name[
                        pbt_frag.active_block_types[
                            int(fragmented.block_type_ind[0, e.block_ind])
                        ]
                        .atoms[i]
                        .name
                    ]
                    .atom_type.upper()
                    .startswith("H")
                    for i in range(
                        int(
                            pbt_frag.n_atoms[
                                int(fragmented.block_type_ind[0, e.block_ind])
                            ]
                        )
                    )
                )
                for e in pose0_entries
            )
            == 3
        )

    whole_ligand = torch.zeros_like(whole.block_type_ind, dtype=torch.bool)
    for block_index in range(whole.max_n_blocks):
        block_type_index = int(whole.block_type_ind[0, block_index])
        if block_type_index < 0:
            continue
        block_type = whole.packed_block_types.active_block_types[block_type_index]
        whole_ligand[0, block_index] = block_type.name == LIGAND_NAME
    fragment_ligand = _fragment_block_mask(fragmented, mapping)
    whole_partner = ~whole_ligand & (whole.block_type_ind >= 0)
    fragment_partner = ~fragment_ligand & (fragmented.block_type_ind >= 0)

    whole_sfxn = beta2016_score_function(
        torch_device, param_db=whole_context.parameter_database
    )
    fragment_sfxn = beta2016_score_function(
        torch_device, param_db=fragment_context.parameter_database
    )
    whole_ddg = calculate_block_pair_ddg(
        whole,
        whole_ligand,
        whole_partner,
        sfxn=whole_sfxn,
        sum_terms=False,
        minimize=False,
    )
    fragment_ddg = calculate_block_pair_ddg(
        fragmented,
        fragment_ligand,
        fragment_partner,
        sfxn=fragment_sfxn,
        sum_terms=False,
        minimize=False,
    )
    attributed = calculate_fragment_interactions(
        fragmented,
        fragment_partner,
        mapping=mapping,
        sfxn=fragment_sfxn,
        sum_terms=False,
    )
    # Fragmentation changes floating-point accumulation order, increasingly so
    # for the seven-fragment case. The observed differences remain below 0.05%.
    torch.testing.assert_close(fragment_ddg, whole_ddg, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        attributed.scores.sum(dim=2), fragment_ddg, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        attributed.scores.sum(dim=2), whole_ddg, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        fragment_ddg.sum(dim=0),
        whole_ddg.sum(dim=0),
        rtol=1e-3,
        atol=1e-3,
    )

    whole_scores = whole_sfxn.render_whole_pose_scoring_module(whole)(
        whole.coords, sum_terms=False, apply_weights=False
    )
    fragment_scores = fragment_sfxn.render_whole_pose_scoring_module(fragmented)(
        fragmented.coords, sum_terms=False, apply_weights=False
    )
    mismatches = [
        (
            score_type.name,
            float(whole_scores[score_index, 0]),
            float(fragment_scores[score_index, 0]),
        )
        for score_index, score_type in enumerate(whole_sfxn.all_score_types())
        if not torch.allclose(
            fragment_scores[score_index],
            whole_scores[score_index],
            rtol=1e-3,
            atol=1e-3,
        )
    ]
    assert not mismatches, f"total-pose term mismatches: {mismatches}"

    whole_weighted = whole_sfxn.render_whole_pose_scoring_module(whole)(
        whole.coords, sum_terms=True, apply_weights=True
    )
    fragment_weighted = fragment_sfxn.render_whole_pose_scoring_module(fragmented)(
        fragmented.coords, sum_terms=True, apply_weights=True
    )
    torch.testing.assert_close(fragment_weighted, whole_weighted, rtol=1e-3, atol=1e-3)

    if target == "ace" and fragmentation == "bridge":
        whole_coords = whole.coords.detach().clone().requires_grad_(True)
        fragment_coords = fragmented.coords.detach().clone().requires_grad_(True)
        whole_score = whole_sfxn.render_whole_pose_scoring_module(whole)(
            whole_coords, sum_terms=True, apply_weights=True
        ).sum()
        fragment_score = fragment_sfxn.render_whole_pose_scoring_module(fragmented)(
            fragment_coords, sum_terms=True, apply_weights=True
        ).sum()
        (whole_gradient,) = torch.autograd.grad(whole_score, (whole_coords,))
        (fragment_gradient,) = torch.autograd.grad(fragment_score, (fragment_coords,))
        assert torch.isfinite(whole_gradient).all()
        assert torch.isfinite(fragment_gradient).all()

        # Atom order is preserved within each fragment block, but blocks are
        # reordered relative to the whole ligand. Compare by atom name.
        whole_block = int(torch.nonzero(whole_ligand[0], as_tuple=False)[0])
        whole_bt = whole.packed_block_types.active_block_types[
            int(whole.block_type_ind[0, whole_block])
        ]
        whole_offset = int(whole.block_coord_offset[0, whole_block])
        whole_by_name = {
            atom.name: whole_gradient[0, whole_offset + atom_index]
            for atom_index, atom in enumerate(whole_bt.atoms)
        }
        for entry in mapping.entries:
            if entry.pose_ind != 0:
                continue
            fragment_bt = fragmented.packed_block_types.active_block_types[
                int(fragmented.block_type_ind[0, entry.block_ind])
            ]
            fragment_offset = int(fragmented.block_coord_offset[0, entry.block_ind])
            for atom_index, atom in enumerate(fragment_bt.atoms):
                torch.testing.assert_close(
                    fragment_gradient[0, fragment_offset + atom_index],
                    whole_by_name[atom.name],
                    rtol=1e-3,
                    atol=1e-3,
                )


# ── helpers shared by _unsplit_* unit tests ───────────────────────────────────


class _MockPoseStack:
    """Minimal stand-in for the data-helper unit tests that only need block_type_ind64."""

    def __init__(self, block_type_ind64: torch.Tensor):
        self.block_type_ind64 = block_type_ind64

    def __len__(self) -> int:
        return int(self.block_type_ind64.shape[0])


def _make_split_entry(
    pose_ind: int,
    block_ind: int,
    group_ind: int,
    orig_bt_ind: int,
    n_atoms: int = 0,
) -> SplitBlockEntry:
    return SplitBlockEntry(
        pose_ind=pose_ind,
        block_ind=block_ind,
        group_ind=group_ind,
        orig_block_type_ind=orig_bt_ind,
        split_to_orig_atom_inds=np.arange(n_atoms, dtype=np.int32),
        orig_residue_label=100 + block_ind,
        orig_chain_label="A",
        orig_ins_code="",
    )


def _build_fragmented(torch_device):
    """Load the ACE fixture, annotate at the first bridge cut, and return the fragmented pose."""
    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    pose, _ctx, mapping = _build(annotated, params_path, torch_device, fragmented=True)
    return pose, mapping


def _make_unsplit_intermediates(pose):
    """Compute the intermediate data structures that unsplit_pose_stack builds internally."""
    from tmol.utility.tensor import exclusive_cumsum2d

    sbm = pose.split_block_mapping
    pbt = pose.packed_block_types
    device = pose.device
    n_poses = len(pose)

    groups, sbs, el = _unsplit_group_entries(sbm)
    per_pose = _unsplit_per_pose_blocks(pose, sbs, el)

    new_max_n_blocks = max(len(bl) for bl in per_pose)
    new_bt64 = torch.full(
        (n_poses, new_max_n_blocks), -1, dtype=torch.int64, device=device
    )
    for p, blocks in enumerate(per_pose):
        for new_b, (bt_idx, *_) in enumerate(blocks):
            new_bt64[p, new_b] = bt_idx

    real_new = new_bt64 >= 0
    n_atoms_blk = torch.zeros(
        (n_poses, new_max_n_blocks), dtype=torch.int32, device=device
    )
    n_atoms_blk[real_new] = pbt.n_atoms[new_bt64[real_new]]
    new_bco = exclusive_cumsum2d(n_atoms_blk)
    new_max_n_atoms = int(torch.max(torch.sum(n_atoms_blk, dim=1)).item())
    old_to_new = _unsplit_old_to_new(per_pose, pose, sbs, el)

    return groups, sbs, el, per_pose, new_bt64, new_bco, new_max_n_atoms, old_to_new


# ── _unsplit_group_entries ────────────────────────────────────────────────────


def test_unsplit_group_entries_basic_grouping():
    """Entries sharing (pose_ind, group_ind) land in the same group bucket."""
    e0 = _make_split_entry(0, 2, 0, 10)
    e1 = _make_split_entry(0, 4, 0, 10)
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=(e0, e1)))
    assert set(groups.keys()) == {(0, 0)}
    assert len(groups[(0, 0)]) == 2
    assert sbs[0] == {2, 4}
    assert el[(0, 2)] is e0
    assert el[(0, 4)] is e1


def test_unsplit_group_entries_groups_sorted_by_block_ind():
    """Within each group, entries are sorted by block_ind regardless of insertion order."""
    e_late = _make_split_entry(0, 7, 0, 10)
    e_early = _make_split_entry(0, 3, 0, 10)
    groups, _, _ = _unsplit_group_entries(SplitBlockMapping(entries=(e_late, e_early)))
    result = groups[(0, 0)]
    assert result[0].block_ind == 3
    assert result[1].block_ind == 7


def test_unsplit_group_entries_multiple_groups_and_poses():
    """Multiple (pose_ind, group_ind) pairs each populate their own bucket."""
    entries = (
        _make_split_entry(0, 2, 0, 20),  # pose 0, group 0 (ligand A)
        _make_split_entry(0, 4, 0, 20),  # pose 0, group 0
        _make_split_entry(0, 7, 1, 30),  # pose 0, group 1 (ligand B)
        _make_split_entry(1, 2, 0, 20),  # pose 1, group 0 (ligand A)
        _make_split_entry(1, 4, 0, 20),  # pose 1, group 0
    )
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=entries))
    assert len(groups[(0, 0)]) == 2
    assert len(groups[(0, 1)]) == 1
    assert len(groups[(1, 0)]) == 2
    assert sbs[0] == {2, 4, 7}
    assert sbs[1] == {2, 4}
    assert el[(0, 7)].group_ind == 1


# ── _unsplit_per_pose_blocks ──────────────────────────────────────────────────


def test_unsplit_per_pose_blocks_all_non_split():
    """Non-split blocks appear as ('orig', bt_idx, old_block_ind) triples in order."""
    bt64 = torch.tensor([[5, 3, 8]], dtype=torch.int64)
    ps = _MockPoseStack(bt64)
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=()))
    per_pose = _unsplit_per_pose_blocks(ps, sbs, el)
    assert per_pose[0] == [(5, "orig", 0), (3, "orig", 1), (8, "orig", 2)]


def test_unsplit_per_pose_blocks_padding_slots_skipped():
    """Block slots whose block_type_ind64 is negative are skipped."""
    bt64 = torch.tensor([[5, -1, 8]], dtype=torch.int64)
    ps = _MockPoseStack(bt64)
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=()))
    per_pose = _unsplit_per_pose_blocks(ps, sbs, el)
    assert per_pose[0] == [(5, "orig", 0), (8, "orig", 2)]


def test_unsplit_per_pose_blocks_first_fragment_is_group_rest_absorbed():
    """The first fragment block in a group emits a 'group' entry; later ones are dropped."""
    # Blocks: [non(5), frag_A(7), frag_B(8), non(6)] — frag_A and frag_B share group 0
    bt64 = torch.tensor([[5, 7, 8, 6]], dtype=torch.int64)
    ps = _MockPoseStack(bt64)
    entries = (
        _make_split_entry(0, 1, 0, 99),
        _make_split_entry(0, 2, 0, 99),
    )
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=entries))
    per_pose = _unsplit_per_pose_blocks(ps, sbs, el)
    assert per_pose[0] == [
        (5, "orig", 0),
        (99, "group", (0, 0)),
        (6, "orig", 3),
    ]


def test_unsplit_per_pose_blocks_two_poses_independent():
    """Each pose produces its own per-pose block list."""
    bt64 = torch.tensor([[5, 7, 8, 6], [5, 7, 8, 6]], dtype=torch.int64)
    ps = _MockPoseStack(bt64)
    entries = tuple(_make_split_entry(p, b, 0, 99) for p in range(2) for b in (1, 2))
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=entries))
    per_pose = _unsplit_per_pose_blocks(ps, sbs, el)
    assert len(per_pose) == 2
    for p in range(2):
        assert per_pose[p] == [
            (5, "orig", 0),
            (99, "group", (p, 0)),
            (6, "orig", 3),
        ]


# ── _unsplit_old_to_new ───────────────────────────────────────────────────────


def test_unsplit_old_to_new_non_split_get_sequential_indices():
    """Non-split blocks receive sequential new-block indices starting from 0."""
    bt64 = torch.tensor([[5, 3, 8]], dtype=torch.int64)
    ps = _MockPoseStack(bt64)
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=()))
    per_pose = _unsplit_per_pose_blocks(ps, sbs, el)
    otn = _unsplit_old_to_new(per_pose, ps, sbs, el)
    assert otn[0] == {0: 0, 1: 1, 2: 2}


def test_unsplit_old_to_new_first_fragment_mapped_rest_absorbed():
    """First fragment in a group maps to a positive new index; subsequent ones map to -1."""
    bt64 = torch.tensor([[5, 7, 8, 6]], dtype=torch.int64)
    ps = _MockPoseStack(bt64)
    entries = (
        _make_split_entry(0, 1, 0, 99),  # first in group → kept
        _make_split_entry(0, 2, 0, 99),  # second in group → absorbed
    )
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=entries))
    per_pose = _unsplit_per_pose_blocks(ps, sbs, el)
    otn = _unsplit_old_to_new(per_pose, ps, sbs, el)
    m = otn[0]
    assert m[0] == 0  # non-split
    assert m[1] == 1  # first fragment → new slot 1
    assert m[2] == -1  # second fragment → absorbed
    assert m[3] == 2  # non-split


def test_unsplit_old_to_new_padding_absent_from_result():
    """Padding slots (block_type_ind64 < 0) do not appear in old_to_new."""
    bt64 = torch.tensor([[5, -1, 8]], dtype=torch.int64)
    ps = _MockPoseStack(bt64)
    groups, sbs, el = _unsplit_group_entries(SplitBlockMapping(entries=()))
    per_pose = _unsplit_per_pose_blocks(ps, sbs, el)
    otn = _unsplit_old_to_new(per_pose, ps, sbs, el)
    assert 1 not in otn[0]  # padding block absent
    assert otn[0][0] == 0
    assert otn[0][2] == 1


# ── _unsplit_build_coords ─────────────────────────────────────────────────────


def test_unsplit_build_coords_fragment_atoms_at_original_positions(torch_device):
    """split_to_orig_atom_inds correctly routes each fragment atom to its slot."""
    pose, _ = _build_fragmented(torch_device)
    groups, sbs, el, per_pose, _, new_bco, new_max_n_atoms, _ = (
        _make_unsplit_intermediates(pose)
    )
    pbt = pose.packed_block_types
    n_poses = len(pose)

    coords_np, new_bco_np = _unsplit_build_coords(
        per_pose, pbt, groups, pose, new_bco, new_max_n_atoms, n_poses
    )

    old_coords = pose.coords.cpu().numpy()
    old_bco = pose.block_coord_offset.cpu().numpy()

    for p, blocks in enumerate(per_pose):
        for new_b, (_, kind, src) in enumerate(blocks):
            if kind != "group":
                continue
            new_off = int(new_bco_np[p, new_b])
            for entry in groups[src]:
                old_off = int(old_bco[p, entry.block_ind])
                for li, oi in enumerate(entry.split_to_orig_atom_inds):
                    np.testing.assert_array_equal(
                        coords_np[p, new_off + oi],
                        old_coords[p, old_off + li],
                    )


def test_unsplit_build_coords_non_fragment_blocks_unchanged(torch_device):
    """Atoms from non-split blocks are copied verbatim."""
    pose, _ = _build_fragmented(torch_device)
    groups, sbs, el, per_pose, _, new_bco, new_max_n_atoms, _ = (
        _make_unsplit_intermediates(pose)
    )
    pbt = pose.packed_block_types
    n_poses = len(pose)

    coords_np, new_bco_np = _unsplit_build_coords(
        per_pose, pbt, groups, pose, new_bco, new_max_n_atoms, n_poses
    )

    old_coords = pose.coords.cpu().numpy()
    old_bco = pose.block_coord_offset.cpu().numpy()

    for p, blocks in enumerate(per_pose):
        for new_b, (bt_idx, kind, old_b) in enumerate(blocks):
            if kind != "orig":
                continue
            new_off = int(new_bco_np[p, new_b])
            old_off = int(old_bco[p, old_b])
            n_at = int(pbt.n_atoms[bt_idx].item())
            np.testing.assert_array_equal(
                coords_np[p, new_off : new_off + n_at],
                old_coords[p, old_off : old_off + n_at],
            )


# ── _unsplit_connections ──────────────────────────────────────────────────────


def test_unsplit_connections_intra_fragment_bond_removed(torch_device):
    """The inter-fragment cut bond is absent from the output connection tensor."""
    pose, _ = _build_fragmented(torch_device)
    groups, sbs, el, per_pose, _, _, _, old_to_new = _make_unsplit_intermediates(pose)
    pbt = pose.packed_block_types
    n_poses = len(pose)
    device = pose.device
    new_max_n_blocks = max(len(bl) for bl in per_pose)

    new_irc64 = _unsplit_connections(
        pose, pbt, n_poses, new_max_n_blocks, sbs, el, old_to_new, device
    )

    # Identify the new index of the merged ligand block (first fragment → kept)
    sbm = pose.split_block_mapping
    frag_blocks = sorted(e.block_ind for e in sbm.entries if e.pose_ind == 0)
    new_lig_b = old_to_new[0][frag_blocks[0]]
    assert new_lig_b >= 0

    # In the ACE fixture the ligand is non-covalent: no external bonds.
    # After removing the intra-fragment bond, every connection slot must be -1.
    n_conn = int(new_irc64.shape[2])
    for c in range(n_conn):
        partner = int(new_irc64[0, new_lig_b, c, 0].item())
        assert partner == -1, (
            f"connection slot {c} of the unsplit ligand block should be empty "
            f"but points to block {partner}"
        )


def test_unsplit_connections_non_fragment_block_connections_preserved(torch_device):
    """Connections between non-split blocks are unchanged after unsplitting."""
    pose, _ = _build_fragmented(torch_device)
    groups, sbs, el, per_pose, _, _, _, old_to_new = _make_unsplit_intermediates(pose)
    pbt = pose.packed_block_types
    n_poses = len(pose)
    device = pose.device
    new_max_n_blocks = max(len(bl) for bl in per_pose)

    new_irc64 = _unsplit_connections(
        pose, pbt, n_poses, new_max_n_blocks, sbs, el, old_to_new, device
    )

    sbm = pose.split_block_mapping
    split_blocks = {e.block_ind for e in sbm.entries if e.pose_ind == 0}
    old_irc64 = pose.inter_residue_connections64

    for p in range(n_poses):
        for old_b, new_b in old_to_new[p].items():
            if new_b < 0 or old_b in split_blocks:
                continue
            bt_idx = int(pose.block_type_ind64[p, old_b].item())
            n_conn = len(pbt.active_block_types[bt_idx].connections)
            for c in range(n_conn):
                old_partner = int(old_irc64[p, old_b, c, 0].item())
                new_partner = int(new_irc64[p, new_b, c, 0].item())
                if old_partner == -1:
                    assert new_partner == -1
                else:
                    # Partner remapped via old_to_new
                    expected = old_to_new[p].get(old_partner, -1)
                    assert new_partner == expected


# ── _unsplit_chain_and_pdb ────────────────────────────────────────────────────


def test_unsplit_chain_and_pdb_shape_and_metadata(torch_device):
    """Output tensors have the right shape; chain IDs and residue labels are preserved."""
    pose, _ = _build_fragmented(torch_device)
    groups, sbs, el, per_pose, _, new_bco, new_max_n_atoms, old_to_new = (
        _make_unsplit_intermediates(pose)
    )
    pbt = pose.packed_block_types
    n_poses = len(pose)
    device = pose.device
    new_max_n_blocks = max(len(bl) for bl in per_pose)

    _, new_bco_np = _unsplit_build_coords(
        per_pose, pbt, groups, pose, new_bco, new_max_n_atoms, n_poses
    )
    new_chain_id, new_pdb = _unsplit_chain_and_pdb(
        pose,
        old_to_new,
        pbt,
        per_pose,
        new_bco_np,
        n_poses,
        new_max_n_blocks,
        new_max_n_atoms,
        device,
    )

    assert new_chain_id.shape == (n_poses, new_max_n_blocks)
    assert new_pdb.residue_labels.shape == (n_poses, new_max_n_blocks)

    for p in range(n_poses):
        for old_b, new_b in old_to_new[p].items():
            if new_b < 0:
                continue
            assert int(pose.chain_id[p, old_b].item()) == int(
                new_chain_id[p, new_b].item()
            )
            assert int(pose.pdb_info.residue_labels[p, old_b]) == int(
                new_pdb.residue_labels[p, new_b]
            )


# ── unsplit_pose_stack ────────────────────────────────────────────────────────


def test_unsplit_pose_stack_no_split_mapping_is_noop(torch_device):
    """With no split_block_mapping, unsplit_pose_stack is a near-identity."""
    structure, params_path, _ = _load_fixture()
    pose, _, _ = _build(structure, params_path, torch_device, fragmented=False)
    assert pose.split_block_mapping is None
    result = unsplit_pose_stack(pose)
    assert result.split_block_mapping is None
    assert result.n_poses == pose.n_poses


def test_unsplit_pose_stack_block_count_decreases_by_one(torch_device):
    """Two fragment blocks collapse into one original block, reducing the count by 1."""
    pose, _ = _build_fragmented(torch_device)
    n_before = int(torch.sum(pose.block_type_ind >= 0).item())
    result = unsplit_pose_stack(pose)
    n_after = int(torch.sum(result.block_type_ind >= 0).item())
    assert n_after == n_before - 1


def test_unsplit_pose_stack_original_block_type_restored(torch_device):
    """The original ligand type (LG1) is present; fragment types (LG1.1, LG1.2) are gone."""
    pose, _ = _build_fragmented(torch_device)
    result = unsplit_pose_stack(pose)
    pbt = result.packed_block_types
    names = {
        pbt.active_block_types[int(result.block_type_ind[0, b])].name
        for b in range(result.max_n_blocks)
        if int(result.block_type_ind[0, b]) >= 0
    }
    assert LIGAND_NAME in names
    assert f"{LIGAND_NAME}.1" not in names
    assert f"{LIGAND_NAME}.2" not in names


def test_unsplit_pose_stack_clears_split_block_mapping(torch_device):
    """The result always has split_block_mapping=None."""
    pose, _ = _build_fragmented(torch_device)
    result = unsplit_pose_stack(pose)
    assert result.split_block_mapping is None


def test_unsplit_pose_stack_fragment_atom_coords_preserved(torch_device):
    """Every atom in the unsplit ligand block carries coordinates from its fragment."""
    pose, _ = _build_fragmented(torch_device)
    result = unsplit_pose_stack(pose)

    pbt_r = result.packed_block_types
    lig_block = next(
        b
        for b in range(result.max_n_blocks)
        if int(result.block_type_ind[0, b]) >= 0
        and pbt_r.active_block_types[int(result.block_type_ind[0, b])].name
        == LIGAND_NAME
    )
    lig_off = int(result.block_coord_offset[0, lig_block])

    sbm = pose.split_block_mapping
    pbt_f = pose.packed_block_types
    for entry in sbm.entries:
        if entry.pose_ind != 0:
            continue
        frag_bt = pbt_f.active_block_types[int(pose.block_type_ind[0, entry.block_ind])]
        frag_off = int(pose.block_coord_offset[0, entry.block_ind])
        for atom_i in range(len(frag_bt.atoms)):
            orig_i = int(entry.split_to_orig_atom_inds[atom_i])
            torch.testing.assert_close(
                result.coords[0, lig_off + orig_i],
                pose.coords[0, frag_off + atom_i],
                rtol=1e-5,
                atol=1e-5,
            )


def test_unsplit_pose_stack_two_pose_stack(torch_device):
    """A two-pose fragmented stack is correctly unsplit pose-independently."""
    from tmol.io import pose_stack_from_biotite

    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    _, context, _ = _build(annotated, params_path, torch_device, fragmented=True)
    stack = struc.stack([annotated, annotated])
    fragmented = pose_stack_from_biotite(
        stack, torch_device, context=context, no_optH=True
    )
    assert len(fragmented) == 2
    result = unsplit_pose_stack(fragmented)
    assert len(result) == 2
    assert result.split_block_mapping is None
    pbt = result.packed_block_types
    for p in range(2):
        names = {
            pbt.active_block_types[int(result.block_type_ind[p, b])].name
            for b in range(result.max_n_blocks)
            if int(result.block_type_ind[p, b]) >= 0
        }
        assert LIGAND_NAME in names, f"Ligand missing in pose {p} after unsplitting"
