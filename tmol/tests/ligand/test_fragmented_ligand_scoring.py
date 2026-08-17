"""Parity tests for user-annotated connected ligand fragments."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io
import numpy as np
import pytest
import torch

from tmol.ligand import FRAGMENT_ID_ANNOTATION
from tmol.ligand import load_params_file

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
        return pose, context, pose.fragmented_ligand_mapping
    return pose, context, None


def _fragment_block_mask(pose, mapping):
    mask = torch.zeros_like(pose.block_type_ind, dtype=torch.bool)
    for record in mapping.blocks:
        mask[record.pose_index, record.block_index] = True
    return mask


def test_fragment_definition_connections_icoors_and_mapping(torch_device):
    structure, params_path, preparation = _load_fixture()
    annotated = _annotate_at_bridge(structure, preparation)
    pose, _, mapping = _build(annotated, params_path, torch_device, fragmented=True)

    assert [record.fragment_name for record in mapping.blocks] == ["LG1.1", "LG1.2"]
    assert len(mapping.connection_pairs) == 1
    block_a, conn_a, block_b, conn_b = mapping.connection_pairs[0]
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
    assert pose.fragmented_ligand_mapping == mapping

    split = biotite_from_pose_stack(
        pose, context.canonical_ordering, merge_fragments=False
    )
    restored = recombine_fragmented_ligands(split, mapping)
    merged = biotite_from_pose_stack(pose, context.canonical_ordering)

    np.testing.assert_array_equal(merged.res_name, restored.res_name)
    np.testing.assert_array_equal(merged.res_id, restored.res_id)
    assert not np.any(np.char.startswith(merged.res_name, f"{LIGAND_NAME}."))
    assert {
        (int(res_id), str(res_name))
        for res_id, res_name in zip(split.res_id, split.res_name)
        if str(res_name).startswith(f"{LIGAND_NAME}.")
    } == {
        (record.pose_residue_label, record.fragment_name) for record in mapping.blocks
    }

    split_records = atom_records_from_pose_stack(pose, merge_fragments=False)
    merged_records = atom_records_from_pose_stack(pose)
    fragment_residue_labels = {record.pose_residue_label for record in mapping.blocks}
    assert fragment_residue_labels <= set(split_records["resi"])
    assert fragment_residue_labels.isdisjoint(set(merged_records["resi"]))
    assert {record.residue_label for record in mapping.blocks} <= set(
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

    assert [
        record.fragment_name for record in pose.fragmented_ligand_mapping.blocks
    ] == [
        "LG1.1",
        "LG1.2",
    ]


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
    mapping = pose.fragmented_ligand_mapping
    assert pose.n_poses == 2
    assert len(mapping.blocks) == 4
    for fragment_id in (1, 2):
        block_indices = {
            record.block_index
            for record in mapping.blocks
            if record.fragment_id == fragment_id
        }
        assert len(block_indices) == 1
    assert pose.clone().fragmented_ligand_mapping == mapping
    split_mapping = pose.split(1).fragmented_ligand_mapping
    assert len(split_mapping.blocks) == 2
    assert {record.pose_index for record in split_mapping.blocks} == {0}
    assert {record.fragment_id for record in split_mapping.blocks} == {1, 2}


def test_fragmented_ligand_minimize_and_pack_e2e():
    from tmol import run_cart_min
    from tmol.score import beta2016_score_function
    from tmol.score import (
        build_coord_mask_for_mask_and_interacting_atoms,
        calculate_block_pair_ddg,
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
    assert packed.fragmented_ligand_mapping == mapping

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
    from tmol.score import beta2016_score_function
    from tmol.score import (
        calculate_block_pair_ddg,
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
        assert len(mapping.blocks) >= 3
        atom_by_name = {atom.name: atom for atom in preparation.residue_type.atoms}
        assert (
            min(
                sum(
                    not atom_by_name[name].atom_type.upper().startswith("H")
                    for name in record.atom_names
                )
                for record in mapping.blocks
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
        for record in mapping.blocks:
            if record.pose_index != 0:
                continue
            fragment_bt = fragmented.packed_block_types.active_block_types[
                int(fragmented.block_type_ind[0, record.block_index])
            ]
            fragment_offset = int(fragmented.block_coord_offset[0, record.block_index])
            for atom_index, atom in enumerate(fragment_bt.atoms):
                torch.testing.assert_close(
                    fragment_gradient[0, fragment_offset + atom_index],
                    whole_by_name[atom.name],
                    rtol=1e-3,
                    atol=1e-3,
                )
