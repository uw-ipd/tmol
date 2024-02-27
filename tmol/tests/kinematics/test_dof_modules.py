import pytest
import torch
import numpy

from tmol.system.packed import PackedResidueSystem

from tmol.tests.torch import requires_cuda
from tmol.kinematics.dof_modules import CartesianDOFs, KinematicDOFs, DOFMaskingFunc
from tmol.system.kinematic_module_support import (  # noqa: F401
    kinematic_operation_build_for,
)


@requires_cuda
def test_cartesian_coord_factory(ubq_system):
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda", torch.cuda.current_device())

    src = CartesianDOFs.build_from(ubq_system)

    # Coords are returned from forward
    assert src.coords.shape == (1, ubq_system.system_size, 3)
    torch.testing.assert_close(
        src.coords[0],
        torch.tensor(ubq_system.coords, dtype=src.coords.dtype),
        equal_nan=True,
    )
    torch.testing.assert_close(
        src()[0],
        torch.tensor(ubq_system.coords, dtype=src.coords.dtype),
        equal_nan=True,
    )

    # Device defaults and device clone
    clone = CartesianDOFs.build_from(src)
    assert clone.coords.device == src.coords.device
    assert clone.coords.device == cpu_device
    assert clone().device == cpu_device

    # Coords are copied, not referenced
    with torch.no_grad():
        torch.testing.assert_close(src.coords, clone.coords, equal_nan=True)
        clone.coords[0] += 1
        with pytest.raises(AssertionError):
            torch.testing.assert_close(src.coords, clone.coords, equal_nan=True)
        clone.coords[0] -= 1
        torch.testing.assert_close(src.coords, clone.coords, equal_nan=True)

    # Device can be overridden
    clone = clone.to(cuda_device)

    assert clone.coords.device != src.coords.device
    assert clone.coords.device == cuda_device
    assert clone().device == cuda_device

    # Coords are returned from forward
    torch.testing.assert_close(
        clone.coords.cpu()[0],
        torch.tensor(ubq_system.coords, dtype=clone.coords.dtype),
        equal_nan=True,
    )
    torch.testing.assert_close(
        clone().cpu()[0],
        torch.tensor(ubq_system.coords, dtype=clone.coords.dtype),
        equal_nan=True,
    )


@requires_cuda
def test_kinematic_dof_factory(ubq_system):
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda", torch.cuda.current_device())

    src = KinematicDOFs.build_from(ubq_system)

    torch.testing.assert_close(
        src()[0], torch.tensor(ubq_system.coords), equal_nan=True
    )

    # Device defaults and device clone
    clone: KinematicDOFs = KinematicDOFs.build_from(src)
    assert clone.dofs.device == src.dofs.device
    assert clone.dofs.device == cpu_device

    # dofs are copied, not referenced
    with torch.no_grad():
        torch.testing.assert_close(src.dofs, clone.dofs)
        clone.dofs[0] += 1
        with pytest.raises(AssertionError):
            torch.testing.assert_close(src.dofs, clone.dofs)

        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                clone()[0],
                torch.tensor(ubq_system.coords, dtype=clone()[0].dtype),
                equal_nan=True,
            )
        clone.dofs[0] -= 1

    torch.testing.assert_close(
        clone()[0],
        torch.tensor(ubq_system.coords, dtype=clone()[0].dtype),
        equal_nan=True,
    )

    # Device can be overridden
    clone = clone.to(cuda_device)

    assert clone.kinop.kin_module.gens_b.device == cpu_device
    assert clone.kinop.kin_module.gens_f.device == cpu_device

    assert clone.dofs.device != src.dofs.device
    assert clone.dofs.device == cuda_device
    assert clone().device == cuda_device

    # Coords are returned from forward
    torch.testing.assert_close(
        clone().cpu()[0],
        torch.tensor(ubq_system.coords, dtype=clone().dtype),
        equal_nan=True,
    )


@pytest.fixture
def gradcheck_test_system(ubq_res) -> PackedResidueSystem:
    system = PackedResidueSystem.from_residues(ubq_res[:2])
    system.coords = system.coords - numpy.mean(
        system.coords[system.atom_metadata["atom_type"].astype(bool)], axis=0
    )

    return system


def kdof_gradcheck_report(kdof, start_dofs, eps=1e-3, atol=1e-5, rtol=5e-3):
    def eval_kin(dofs_x):
        full_coords = kdof.kinop(
            DOFMaskingFunc.apply(dofs_x, tuple(kdof.dof_mask), kdof.full_dofs)
        )[None, ...]

        return full_coords[~torch.isnan(full_coords)]

    torch.autograd.gradcheck(eval_kin, start_dofs, atol=atol, rtol=rtol)


def test_kinematic_dofs_gradcheck_perturbed(gradcheck_test_system, torch_device):
    kdof: KinematicDOFs = KinematicDOFs.build_from(gradcheck_test_system)
    torch.random.manual_seed(1663)
    start_dofs = (
        (kdof.dofs + ((torch.rand_like(kdof.dofs) - 0.5) * 0.01))
        .clone()
        .detach()
        .requires_grad_(True)
    )

    kdof_gradcheck_report(kdof, start_dofs)


def test_kinematic_dofs_gradcheck(gradcheck_test_system, torch_device):
    kdof: KinematicDOFs = KinematicDOFs.build_from(gradcheck_test_system)
    start_dofs = kdof.dofs.clone().detach().requires_grad_(True)

    kdof_gradcheck_report(kdof, start_dofs)
