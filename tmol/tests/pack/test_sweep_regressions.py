import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.tests.data import data_path


@pytest.mark.parametrize(
    "fixture",
    [
        "two_atom_residue_1gj2",  # O (O, HO) on a DNA phosphate: no grandchild
    ],
)
def test_sweep_entry_builds_with_opth(fixture, torch_device):
    """A trimmed sweep entry builds and runs opt-H to finite coordinates."""
    structure = atom_array_from_cif(
        data_path("sweep_regressions", f"{fixture}.cif.zst")
    )
    pose_stack = pose_stack_from_biotite(
        structure, torch_device, prepare_ligands=True, no_optH=False
    )
    assert torch.isfinite(pose_stack.coords).all()
