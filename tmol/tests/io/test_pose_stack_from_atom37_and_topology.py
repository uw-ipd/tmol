from pathlib import Path

import biotite.structure as struc
import biotite.structure.io
import numpy as np
import pytest
import torch

from tmol.io import (
    Atom37MappingError,
    build_context_from_biotite,
    canonical_form_from_biotite,
    canonical_ordering_for_biotite,
    pose_stack_from_atom37_and_topology,
    pose_stack_from_biotite,
    prepare_atom37_pose_builder,
)
from tmol.tests.data import data_path


def _first_residues(structure: struc.AtomArray, count: int) -> struc.AtomArray:
    starts = struc.get_residue_starts(structure, add_exclusive_stop=True)
    return structure[: starts[min(count, len(starts) - 1)]].copy()


def _atomized_atom37(
    structure: struc.AtomArray,
    device: torch.device,
    n_poses: int = 1,
) -> tuple[struc.AtomArray, torch.Tensor]:
    """Encode every input atom as one AtomWorks-style atomized token."""
    structure = structure.copy()
    n_atoms = structure.array_length()
    structure.set_annotation("token_id", np.arange(n_atoms, dtype=np.int64))
    structure.set_annotation("atom37_slot", np.ones(n_atoms, dtype=np.int64))

    atom37 = torch.full(
        (n_poses, n_atoms, 37, 3),
        torch.nan,
        dtype=torch.float32,
        device=device,
    )
    reference = torch.as_tensor(structure.coord, dtype=torch.float32, device=device)
    atom37[:, :, 1] = reference.unsqueeze(0)
    if n_poses > 1:
        atom37[1:, :, 1] += torch.arange(
            1, n_poses, dtype=torch.float32, device=device
        ).reshape(-1, 1, 1)
    return structure, atom37


def _load_structure(path: Path) -> struc.AtomArray:
    structure = biotite.structure.io.load_structure(str(path), model=1)
    assert isinstance(structure, struc.AtomArray)
    return structure


def test_canonical_form_routes_batched_atom37_and_gradients(biotite_1ubq, torch_device):
    structure, atom37 = _atomized_atom37(
        _first_residues(biotite_1ubq, 2), torch_device, n_poses=2
    )
    atom37.requires_grad_(True)

    cf = canonical_form_from_biotite(
        structure,
        torch_device,
        atom37_coords=atom37,
    )

    assert cf.coords.shape[0] == 2
    assert cf.atom_b_factor is not None
    assert cf.atom_b_factor.shape[0] == 2
    co = canonical_ordering_for_biotite()
    n_index = co.restypes_atom_index_mapping[structure.res_name[0]]["N"]
    torch.testing.assert_close(cf.coords[:, 0, n_index], atom37[:, 0, 1])

    torch.nan_to_num(cf.coords).sum().backward()
    torch.testing.assert_close(
        atom37.grad[:, 0, 1], torch.ones_like(atom37.grad[:, 0, 1])
    )
    assert torch.count_nonzero(atom37.grad[:, :, 0]) == 0


def test_all_nan_atom37_coordinate_is_missing_not_reference_fallback(
    biotite_1ubq, torch_device
):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    oxygen = int(np.flatnonzero(structure.atom_name == "O")[0])
    atom37[:, oxygen, 1] = torch.nan
    atom37.requires_grad_(True)

    cf = canonical_form_from_biotite(
        structure,
        torch_device,
        atom37_coords=atom37,
    )

    co = canonical_ordering_for_biotite()
    oxygen_index = co.restypes_atom_index_mapping[structure.res_name[oxygen]]["O"]
    assert torch.isnan(cf.coords[0, 0, oxygen_index]).all()
    torch.nan_to_num(cf.coords).sum().backward()
    assert torch.count_nonzero(atom37.grad[:, oxygen, 1]) == 0


def _build_atom37_pose(adapter, atom37, structure, context):
    if adapter == "direct":
        return pose_stack_from_atom37_and_topology(
            atom37, structure, context, no_optH=True
        )
    return prepare_atom37_pose_builder(structure, context)(atom37, opt_h=False)


@pytest.mark.parametrize("adapter", ["direct", "prepared"])
def test_atom37_mapped_coordinates_are_reference_independent(
    biotite_1ubq, torch_device, adapter
):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    context = build_context_from_biotite(structure, torch_device)
    unresolved_reference = structure.copy()
    unresolved_reference.coord[:] = np.nan
    atom37 = atom37.requires_grad_(True)

    expected = _build_atom37_pose(adapter, atom37, structure, context)
    actual = _build_atom37_pose(adapter, atom37, unresolved_reference, context)

    assert actual.max_n_blocks == 1
    torch.testing.assert_close(actual.coords, expected.coords)
    actual.coords[actual.real_atoms].sum().backward()
    assert torch.count_nonzero(atom37.grad) > 0


@pytest.mark.parametrize("adapter", ["direct", "prepared"])
@pytest.mark.parametrize(
    ("triplet", "message"),
    [
        ([np.nan, 1.0, 2.0], "partial NaN"),
        ([np.inf, 1.0, 2.0], "contains infinity"),
    ],
)
def test_atom37_rejects_malformed_mapped_triplets(
    biotite_1ubq, torch_device, adapter, triplet, message
):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    context = build_context_from_biotite(structure, torch_device)
    atom37[0, 0, 1] = torch.tensor(triplet, device=torch_device)

    with pytest.raises(Atom37MappingError, match=message):
        _build_atom37_pose(adapter, atom37, structure, context)


@pytest.mark.parametrize("adapter", ["direct", "prepared"])
@pytest.mark.parametrize("reference_missing", [False, True])
def test_atom37_required_mainchain_missing_from_both_sources_errors(
    biotite_1ubq, torch_device, adapter, reference_missing
):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    context = build_context_from_biotite(structure, torch_device)
    nitrogen = int(np.flatnonzero(structure.atom_name == "N")[0])
    if reference_missing:
        structure.coord[nitrogen] = np.nan
    atom37[:, nitrogen, 1] = torch.nan

    with pytest.raises(
        Atom37MappingError,
        match="Required mainchain.*Atom37/Biotite.*never fall back",
    ):
        _build_atom37_pose(adapter, atom37, structure, context)


@pytest.mark.parametrize("adapter", ["direct", "prepared"])
@pytest.mark.parametrize("reference_missing", [False, True])
def test_atom37_missing_leaf_completion_is_deterministic_and_differentiable(
    biotite_1ubq, torch_device, adapter, reference_missing
):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    context = build_context_from_biotite(structure, torch_device)
    oxygen = int(np.flatnonzero(structure.atom_name == "O")[0])
    structure.coord[oxygen] = np.nan if reference_missing else [100.0, 100.0, 100.0]
    atom37[:, oxygen, 1] = torch.nan

    first_coords = atom37.detach().clone().requires_grad_(True)
    second_coords = atom37.detach().clone()
    first = _build_atom37_pose(adapter, first_coords, structure, context)
    second = _build_atom37_pose(adapter, second_coords, structure, context)
    torch.testing.assert_close(first.coords, second.coords)

    block_type = first.packed_block_types.active_block_types[
        int(first.block_type_ind[0, 0])
    ]
    pose_oxygen = int(first.block_coord_offset[0, 0]) + block_type.atom_to_idx["O"]
    assert torch.isfinite(first.coords[0, pose_oxygen]).all()
    if not reference_missing:
        assert not torch.all(first.coords[0, pose_oxygen] == 100)
    first.coords[0, pose_oxygen].sum().backward()
    assert torch.count_nonzero(first_coords.grad) > 0
    assert torch.count_nonzero(first_coords.grad[:, oxygen, 1]) == 0


@pytest.mark.parametrize("filename", ["1ubq.pdb", "1bna.pdb", "3zp8.pdb"])
def test_atom37_pose_supports_protein_dna_and_rna(filename, torch_device):
    structure = _first_residues(
        _load_structure(data_path("pdb", filename)),
        2,
    )
    structure, atom37 = _atomized_atom37(structure, torch_device, n_poses=2)
    atom37.requires_grad_(True)
    context = build_context_from_biotite(structure, torch_device)

    pose = pose_stack_from_atom37_and_topology(atom37, structure, context)

    assert pose.n_poses == 2
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    pose.coords[pose.real_atoms].sum().backward()
    assert atom37.grad is not None
    assert torch.count_nonzero(atom37.grad) > 0


@pytest.mark.parametrize(
    "filename,closed",
    [("1ubq.pdb", False), ("1bna.pdb", False), ("3zp8.pdb", False), ("1ubq.pdb", True)],
)
def test_prepared_atom37_builder_matches_direct_pose(filename, closed, torch_device):
    structure = _first_residues(_load_structure(data_path("pdb", filename)), 2)
    if closed:
        # Closing the peptide changes the terminal hydrogen complement.
        structure = structure[structure.element != "H"]
        structure.bonds = struc.BondList(structure.array_length())
        structure.bonds.add_bond(
            int(np.flatnonzero(structure.atom_name == "C")[-1]),
            int(np.flatnonzero(structure.atom_name == "N")[0]),
            struc.BondType.SINGLE,
        )
    structure, atom37 = _atomized_atom37(structure, torch_device, n_poses=2)
    context = build_context_from_biotite(structure, torch_device)

    expected = pose_stack_from_atom37_and_topology(
        atom37, structure, context, no_optH=True
    )
    builder = prepare_atom37_pose_builder(structure, context)
    if closed:
        bonds = builder._canonical_form(
            builder._canonical_coords(atom37)
        ).covalent_bonds
        assert bonds.shape == (2, 5)
        assert bonds[:, 0].tolist() == [0, 1]
    actual = builder(atom37, opt_h=False)

    torch.testing.assert_close(actual.coords, expected.coords)
    torch.testing.assert_close(actual.block_type_ind, expected.block_type_ind)
    torch.testing.assert_close(
        actual.inter_residue_connections, expected.inter_residue_connections
    )


def test_prepared_atom37_builder_is_reusable_and_differentiable(
    biotite_1ubq, torch_device
):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 2), torch_device)
    context = build_context_from_biotite(structure, torch_device)
    builder = prepare_atom37_pose_builder(structure, context)

    first_coords = atom37.detach().clone().requires_grad_(True)
    first_pose = builder(first_coords, opt_h=False)
    cached_pose = next(iter(builder._pose_topologies.values())).pose_stack
    assert cached_pose.coords.grad_fn is None
    assert first_pose.packed_block_types is context.packed_block_types
    assert cached_pose.packed_block_types is context.packed_block_types
    structural_tensors = (
        "coords",
        "block_coord_offset",
        "block_coord_offset64",
        "inter_residue_connections",
        "inter_residue_connections64",
        "inter_block_bondsep",
        "inter_block_bondsep64",
        "block_type_ind",
        "block_type_ind64",
        "chain_id",
        "chain_id64",
    )
    for name in structural_tensors:
        assert (
            getattr(first_pose, name).data_ptr()
            != getattr(cached_pose, name).data_ptr()
        )
    assert not np.shares_memory(
        first_pose.pdb_info.residue_labels, cached_pose.pdb_info.residue_labels
    )
    assert not np.shares_memory(
        first_pose.pdb_info.atom_occupancy, cached_pose.pdb_info.atom_occupancy
    )
    first_snapshot = first_pose.coords.detach().clone()
    first_pose.coords[first_pose.real_atoms].sum().backward()
    assert torch.count_nonzero(first_coords.grad) > 0

    first_pose.block_type_ind.fill_(-1)
    first_pose.block_coord_offset.add_(100)
    first_pose.inter_residue_connections.fill_(42)
    first_pose.pdb_info.residue_labels.fill(-100)
    first_pose.pdb_info.residue_insertion_codes.fill("X")
    first_pose.pdb_info.chain_labels.fill("Z")
    first_pose.pdb_info.atom_occupancy.fill(-1)
    first_pose.pdb_info.atom_b_factor.fill(-1)

    second_coords = atom37.detach().clone()
    second_coords[:, :, 1] += 1
    second_coords.requires_grad_(True)
    second_pose = builder(second_coords, opt_h=False)
    expected_second = pose_stack_from_atom37_and_topology(
        second_coords, structure, context, no_optH=True
    )
    assert second_pose.packed_block_types is context.packed_block_types
    torch.testing.assert_close(first_pose.coords, first_snapshot)
    for name in structural_tensors:
        torch.testing.assert_close(
            getattr(second_pose, name), getattr(expected_second, name)
        )
        assert (
            getattr(second_pose, name).data_ptr()
            != getattr(cached_pose, name).data_ptr()
        )
        assert (
            getattr(second_pose, name).data_ptr()
            != getattr(first_pose, name).data_ptr()
        )
    for name in (
        "residue_labels",
        "residue_insertion_codes",
        "chain_labels",
        "atom_occupancy",
        "atom_b_factor",
    ):
        actual_metadata = getattr(second_pose.pdb_info, name)
        np.testing.assert_array_equal(
            actual_metadata, getattr(expected_second.pdb_info, name)
        )
        assert not np.shares_memory(
            actual_metadata, getattr(cached_pose.pdb_info, name)
        )
        assert not np.shares_memory(actual_metadata, getattr(first_pose.pdb_info, name))
    second_pose.coords[second_pose.real_atoms].sum().backward()
    assert torch.count_nonzero(second_coords.grad) > 0

    batched_coords = second_coords.detach().expand(2, -1, -1, -1).clone()
    batched_pose = builder(batched_coords, opt_h=False)
    expected_batched = pose_stack_from_atom37_and_topology(
        batched_coords, structure, context, no_optH=True
    )
    torch.testing.assert_close(batched_pose.coords, expected_batched.coords)
    assert set(builder._pose_topologies) == {1, 2}

    oxygen = int(np.flatnonzero(structure.atom_name == "O")[0])
    nonfinite_coords = second_coords.detach().clone()
    nonfinite_coords[:, oxygen, 1] = torch.nan
    actual_nonfinite = builder(nonfinite_coords, opt_h=False)
    expected_nonfinite = pose_stack_from_atom37_and_topology(
        nonfinite_coords, structure, context, no_optH=True
    )
    torch.testing.assert_close(actual_nonfinite.coords, expected_nonfinite.coords)

    for batch_size in (3, 4, 5):
        builder(second_coords.detach().expand(batch_size, -1, -1, -1), opt_h=False)
    assert set(builder._pose_topologies) == {1, 3, 4, 5}


@pytest.mark.parametrize(
    ("residue_start", "residue_stop", "nhq_name"),
    [(0, 2, "GLN"), (66, 69, "HIS")],
)
def test_explicit_opth_flags_preserve_previous_results(
    biotite_1ubq, torch_device, residue_start, residue_stop, nhq_name
):
    starts = struc.get_residue_starts(biotite_1ubq, add_exclusive_stop=True)
    structure = biotite_1ubq[starts[residue_start] : starts[residue_stop]].copy()
    assert nhq_name in structure.res_name
    structure = structure[structure.element != "H"]
    structure, atom37 = _atomized_atom37(structure, torch_device)
    context = build_context_from_biotite(structure, torch_device)
    builder = prepare_atom37_pose_builder(structure, context)
    preserved = builder(atom37)
    coords = atom37.detach().requires_grad_(True)

    torch.manual_seed(0)
    expected = pose_stack_from_atom37_and_topology(
        coords, structure, context, no_optH=False
    )
    torch.manual_seed(0)
    actual = builder(coords, opt_h=True)

    torch.testing.assert_close(actual.block_type_ind, expected.block_type_ind)
    torch.testing.assert_close(actual.coords, expected.coords)
    assert not torch.equal(actual.block_type_ind, preserved.block_type_ind) or not (
        torch.allclose(actual.coords, preserved.coords)
    )
    actual.coords[actual.real_atoms].sum().backward()
    assert torch.count_nonzero(coords.grad) > 0


def test_prepared_atom37_builder_replays_variable_leaf_presence(
    biotite_1ubq, torch_device
):
    structure = _first_residues(biotite_1ubq, 2)
    oxygen = int(np.flatnonzero(structure.atom_name == "O")[0])
    structure.coord[oxygen] = np.nan
    structure, atom37 = _atomized_atom37(structure, torch_device)
    context = build_context_from_biotite(structure, torch_device)
    builder = prepare_atom37_pose_builder(structure, context)

    first = builder(atom37, opt_h=False)
    second_coords = atom37.clone()
    second_coords[0, oxygen, 1] = torch.tensor([1.0, 2.0, 3.0], device=torch_device)
    expected_second = pose_stack_from_atom37_and_topology(
        second_coords, structure, context, no_optH=True
    )
    actual_second = builder(second_coords, opt_h=False)

    assert torch.isfinite(first.coords[first.real_atoms]).all()
    assert builder._topology_cache_safe
    assert set(builder._pose_topologies) == {1}
    torch.testing.assert_close(actual_second.coords, expected_second.coords)


def test_prepared_atom37_builder_falls_back_for_ambiguous_histidine_hydrogen(
    biotite_1ubq, torch_device
):
    structure = biotite_1ubq[biotite_1ubq.res_name == "HIS"].copy()
    structure.atom_name[structure.atom_name == "HE2"] = "HN"
    structure, _ = _atomized_atom37(structure, torch_device)
    context = build_context_from_biotite(structure, torch_device)

    builder = prepare_atom37_pose_builder(structure, context)

    assert not builder._topology_cache_safe


@pytest.mark.parametrize("prebuilt", [True, False])
def test_atom37_pose_uses_ligand_context(torch_device, prebuilt):
    cif_path = data_path("protein_ligand_test", "cif_inputs", "ace.ligand.cif")
    params_path = data_path("protein_ligand_test", "ace.xtal-lig.mmff94.tmol")
    from tmol.io import atom_array_from_cif

    structure = atom_array_from_cif(cif_path)
    structure, atom37 = _atomized_atom37(structure, torch_device)
    atom37.requires_grad_(True)
    context = build_context_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=[str(params_path)] if prebuilt else None,
    )

    pose = pose_stack_from_atom37_and_topology(atom37, structure, context)

    assert any(
        block.name == "LG1" for block in pose.packed_block_types.active_block_types
    )
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    pose.coords[pose.real_atoms].sum().backward()
    assert torch.count_nonzero(atom37.grad) > 0

    if prebuilt:
        return

    from tmol.io.details._build_missing_leaf_atoms import _apply_h_geometric_completion

    pbt = pose.packed_block_types
    missing = pbt.h_completion_ann.eligible[pose.block_type_ind.clamp_min(0).long()]
    assert missing.any()
    assert torch.autograd.gradcheck(
        lambda coords: _apply_h_geometric_completion(
            pbt,
            coords,
            missing,
            pose.block_coord_offset,
            pose.block_type_ind,
            pose.inter_residue_connections,
        )[pose.real_atoms],
        (pose.coords.detach().double().requires_grad_(),),
        fast_mode=True,
    )


def test_pose_stack_from_biotite_accepts_atom37_directly(biotite_1ubq, torch_device):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    context = build_context_from_biotite(structure, torch_device)

    pose = pose_stack_from_biotite(
        structure,
        torch_device,
        context=context,
        atom37_coords=atom37,
        no_optH=True,
    )

    assert pose.n_poses == 1
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()


def test_atom37_pose_can_return_atom_mapping(biotite_1ubq, torch_device):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    context = build_context_from_biotite(structure, torch_device)

    pose, details = pose_stack_from_atom37_and_topology(
        atom37,
        structure,
        context,
        no_optH=True,
        return_atom_mapping=True,
    )

    n_real_atoms = int(pose.real_atoms.sum())
    assert details["can_atom_mapping"].shape[0] == n_real_atoms
    assert details["ps_atom_mapping"].shape[0] == n_real_atoms


def test_atom37_gradients_survive_hydrogen_optimization(biotite_1ubq, torch_device):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    atom37.requires_grad_(True)
    context = build_context_from_biotite(structure, torch_device)

    pose = pose_stack_from_atom37_and_topology(
        atom37,
        structure,
        context,
    )

    pose.coords[pose.real_atoms].sum().backward()
    assert pose.coords.requires_grad
    assert torch.count_nonzero(atom37.grad) > 0


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (
            lambda structure, atom37: structure.del_annotation("atom37_slot"),
            ValueError,
            "atom37_slot",
        ),
        (
            lambda structure, atom37: structure.set_annotation(
                "atom37_slot", np.full(structure.array_length(), 37, dtype=np.int64)
            ),
            ValueError,
            "less than 37",
        ),
        (
            lambda structure, atom37: structure.set_annotation(
                "token_id",
                np.full(structure.array_length(), atom37.shape[1], dtype=np.int64),
            ),
            ValueError,
            "exceeds atom37_coords token count",
        ),
        (
            lambda structure, atom37: structure.set_annotation(
                "token_id", np.zeros(structure.array_length(), dtype=np.int64)
            ),
            ValueError,
            "unique",
        ),
    ],
)
def test_atom37_coordinate_mapping_validation(
    biotite_1ubq, torch_device, mutate, error, message
):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    mutate(structure, atom37)

    with pytest.raises(error, match=message) as exc_info:
        canonical_form_from_biotite(
            structure,
            torch_device,
            atom37_coords=atom37,
        )
    assert isinstance(exc_info.value, Atom37MappingError)


def test_atom37_coordinate_shape_and_pose_count_validation(biotite_1ubq, torch_device):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)

    with pytest.raises(ValueError, match="must have shape"):
        canonical_form_from_biotite(
            structure,
            torch_device,
            atom37_coords=atom37[:, :, :36],
        )

    stack = struc.stack([structure, structure, structure])
    with pytest.raises(ValueError, match="has 3 poses.*has 1"):
        canonical_form_from_biotite(
            stack,
            torch_device,
            atom37_coords=atom37,
        )
