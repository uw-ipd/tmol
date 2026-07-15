"""Correctness of the ``context=`` reuse path in ``pose_stack_from_biotite``.

Scoring many structures that share one ligand can build the expensive,
structure-independent ``BiotitePoseBuildContext`` once and reuse it, recomputing
only the per-structure canonical form. These tests check that reusing a context
reproduces the ``prepare_ligands=True`` result and is stable across repeats.
"""

from pathlib import Path

import biotite.structure as struc
import biotite.structure.io
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io.pose_stack_from_biotite import (
    build_context_from_biotite,
    pose_stack_from_biotite,
)
from tmol.ligand.registry import clear_cache

PLI_DATA_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
TARGET = "ace"
N_REPEATS = 3


@pytest.fixture(autouse=True)
def _clear_ligand_cache() -> None:
    clear_cache()


def _load_complex_cif(target: str) -> struc.AtomArray:
    cif_path = PLI_DATA_DIR / f"{target}.tmol.nomin.cif"
    structure = biotite.structure.io.load_structure(
        str(cif_path), model=1, include_bonds=True
    )
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]
    return structure


def _params_files(target: str = TARGET) -> list[str]:
    return [str(PLI_DATA_DIR / f"{target}.xtal-lig.mmff94.tmol")]


def _build_context(structure, torch_device):
    return build_context_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=_params_files(),
        param_db=ParameterDatabase.get_default(),
    )


def _assert_pose_stacks_equal(a, b) -> None:
    assert a.coords.shape == b.coords.shape
    assert torch.equal(a.block_type_ind, b.block_type_ind)
    assert torch.equal(a.real_atoms, b.real_atoms)
    real = a.real_atoms
    assert torch.allclose(a.coords[real], b.coords[real], atol=1e-5, equal_nan=True)


def test_reused_context_matches_prepare_ligands(torch_device):
    """A pose built from a reused context equals one built with prepare_ligands."""
    structure = _load_complex_cif(TARGET)

    reference, _ = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        ligand_params_files=_params_files(),
        param_db=ParameterDatabase.get_default(),
        no_optH=True,
        return_context=True,
    )

    context = _build_context(structure, torch_device)
    pose_stack = pose_stack_from_biotite(
        structure, torch_device, context=context, no_optH=True
    )
    _assert_pose_stacks_equal(reference, pose_stack)


def test_repeated_calls_with_same_context_stable(torch_device):
    """Reusing one context across repeated calls is stable."""
    structure = _load_complex_cif(TARGET)
    context = _build_context(structure, torch_device)

    first = pose_stack_from_biotite(
        structure, torch_device, context=context, no_optH=True
    )
    for _ in range(N_REPEATS):
        pose_stack = pose_stack_from_biotite(
            structure, torch_device, context=context, no_optH=True
        )
        _assert_pose_stacks_equal(first, pose_stack)


def test_context_and_param_db_mutually_exclusive(torch_device):
    """Supplying both context= and param_db= is ambiguous and should raise."""
    structure = _load_complex_cif(TARGET)
    context = _build_context(structure, torch_device)

    with pytest.raises(ValueError):
        pose_stack_from_biotite(
            structure,
            torch_device,
            context=context,
            param_db=ParameterDatabase.get_default(),
        )


def test_context_and_prepare_ligands_mutually_exclusive(torch_device):
    """Supplying both context= and prepare_ligands=True should raise."""
    structure = _load_complex_cif(TARGET)
    context = _build_context(structure, torch_device)

    with pytest.raises(ValueError):
        pose_stack_from_biotite(
            structure,
            torch_device,
            context=context,
            prepare_ligands=True,
        )
