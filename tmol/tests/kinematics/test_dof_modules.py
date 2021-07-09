import pytest
import torch
import numpy

from tmol.system.packed import PackedResidueSystem

from tmol.tests.torch import requires_cuda
from tmol.kinematics.dof_modules import CartesianDOFs, KinematicDOFs


@requires_cuda
def test_cartesian_coord_factory(ubq_system):
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda", torch.cuda.current_device())

    src = CartesianDOFs.build_from(ubq_system)

    # Coords are returned from forward
    assert src.coords.shape == (1, ubq_system.system_size, 3)
    torch.testing.assert_allclose(src.coords[0], ubq_system.coords)
    torch.testing.assert_allclose(src()[0], ubq_system.coords)

    # Device defaults and device clone
    clone = CartesianDOFs.build_from(src)
    assert clone.coords.device == src.coords.device
    assert clone.coords.device == cpu_device
    assert clone().device == cpu_device

    # Coords are copied, not referenced
    torch.testing.assert_allclose(src.coords, clone.coords)
    clone.coords[0] += 1
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(src.coords, clone.coords)
    clone.coords[0] -= 1
    torch.testing.assert_allclose(src.coords, clone.coords)

    # Device can be overridden
    clone = clone.to(cuda_device)

    assert clone.coords.device != src.coords.device
    assert clone.coords.device == cuda_device
    assert clone().device == cuda_device

    # Coords are returned from forward
    torch.testing.assert_allclose(clone.coords.cpu()[0], ubq_system.coords)
    torch.testing.assert_allclose(clone().cpu()[0], ubq_system.coords)


@requires_cuda
def test_kinematic_dof_factory(ubq_system):
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda", torch.cuda.current_device())

    src = KinematicDOFs.build_from(ubq_system)

    torch.testing.assert_allclose(src()[0], ubq_system.coords)

    # Device defaults and device clone
    clone: KinematicDOFs = KinematicDOFs.build_from(src)
    assert clone.dofs.device == src.dofs.device
    assert clone.dofs.device == cpu_device

    # dofs are copied, not referenced
    torch.testing.assert_allclose(src.dofs, clone.dofs)
    clone.dofs[0] += 1
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(src.dofs, clone.dofs)

    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(clone()[0], ubq_system.coords)
    clone.dofs[0] -= 1

    # Device can be overridden

    clone = clone.to(cuda_device)

    assert clone.kinop.kin_module.gens_b.device == cpu_device
    assert clone.kinop.kin_module.gens_f.device == cpu_device

    assert clone.dofs.device != src.dofs.device
    assert clone.dofs.device == cuda_device
    assert clone().device == cuda_device

    # Coords are returned from forward
    torch.testing.assert_allclose(clone().cpu()[0], ubq_system.coords)


@pytest.fixture
def gradcheck_test_system(ubq_res) -> PackedResidueSystem:
    system = PackedResidueSystem.from_residues(ubq_res[:4])
    system.coords = system.coords - numpy.mean(
        system.coords[system.atom_metadata["atom_type"].astype(bool)], axis=0
    )

    return system


def kdof_gradcheck_report(kdof, start_dofs, eps=1e-3, atol=1e-5, rtol=5e-3):
    def eval_kin(dofs_x):
        kdof.dofs[:] = dofs_x
        full_coords = kdof()
        return full_coords[~torch.isnan(full_coords)]

    # we only minimize the "rbdel" dofs
    result = eval_kin(start_dofs)

    # Extract results from torch/autograd/gradcheck.py
    from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian

    (analytical,), reentrant, correct_grad_sizes = get_analytical_jacobian(
        (start_dofs,), result
    )
    numerical = get_numerical_jacobian(eval_kin, start_dofs, start_dofs, eps=eps)

    torch.testing.assert_allclose(analytical, numerical, atol=atol, rtol=rtol)


def test_kinematic_dofs_gradcheck_perturbed(gradcheck_test_system, torch_device):
    kdof: KinematicDOFs = KinematicDOFs.build_from(gradcheck_test_system)
    torch.random.manual_seed(1663)
    start_dofs = (
        (kdof.dofs + ((torch.rand_like(kdof.dofs) - .5) * .01))
        .clone()
        .detach()
        .requires_grad_(True)
    )

    kdof_gradcheck_report(kdof, start_dofs)


def test_kinematic_dofs_gradcheck(gradcheck_test_system, torch_device):
    kdof: KinematicDOFs = KinematicDOFs.build_from(gradcheck_test_system)
    torch.random.manual_seed(1663)
    start_dofs = kdof.dofs.clone().detach().requires_grad_(True)

    kdof_gradcheck_report(kdof, start_dofs)
