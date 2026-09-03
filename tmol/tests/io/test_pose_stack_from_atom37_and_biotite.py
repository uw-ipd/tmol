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
    pose_stack_from_atom37_and_biotite,
    pose_stack_from_biotite,
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


def test_nonfinite_atom37_coordinate_falls_back_to_biotite(biotite_1ubq, torch_device):
    structure, atom37 = _atomized_atom37(_first_residues(biotite_1ubq, 1), torch_device)
    atom37[:, 0, 1] = torch.nan
    atom37.requires_grad_(True)

    cf = canonical_form_from_biotite(
        structure,
        torch_device,
        atom37_coords=atom37,
    )

    co = canonical_ordering_for_biotite()
    n_index = co.restypes_atom_index_mapping[structure.res_name[0]]["N"]
    torch.testing.assert_close(
        cf.coords[0, 0, n_index],
        torch.as_tensor(structure.coord[0], device=torch_device),
    )
    torch.nan_to_num(cf.coords).sum().backward()
    assert torch.count_nonzero(atom37.grad[:, 0, 1]) == 0


@pytest.mark.parametrize("filename", ["1ubq.pdb", "1bna.pdb", "3zp8.pdb"])
def test_atom37_pose_supports_protein_dna_and_rna(filename, torch_device):
    structure = _first_residues(
        _load_structure(data_path("pdb", filename)),
        2,
    )
    structure, atom37 = _atomized_atom37(structure, torch_device, n_poses=2)
    atom37.requires_grad_(True)
    context = build_context_from_biotite(structure, torch_device)

    pose = pose_stack_from_atom37_and_biotite(atom37, structure, context)

    assert pose.n_poses == 2
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    pose.coords[pose.real_atoms].sum().backward()
    assert atom37.grad is not None
    assert torch.count_nonzero(atom37.grad) > 0


def test_atom37_pose_uses_ligand_context(torch_device):
    cif_path = data_path("protein_ligand_test", "cif_inputs", "ace.ligand.cif")
    params_path = data_path("protein_ligand_test", "ace.xtal-lig.mmff94.tmol")
    structure = _load_structure(cif_path)
    structure, atom37 = _atomized_atom37(structure, torch_device)
    atom37.requires_grad_(True)
    context = build_context_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=[str(params_path)],
        sample_proton_chi=False,
    )

    pose = pose_stack_from_atom37_and_biotite(atom37, structure, context)

    assert any(
        block.name == "LG1" for block in pose.packed_block_types.active_block_types
    )
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    pose.coords[pose.real_atoms].sum().backward()
    assert torch.count_nonzero(atom37.grad) > 0


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

    pose, details = pose_stack_from_atom37_and_biotite(
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

    pose = pose_stack_from_atom37_and_biotite(
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
