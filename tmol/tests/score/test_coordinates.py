import pytest

import torch

from tmol.tests.torch import requires_cuda
from tmol.score.coordinates import (
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@requires_cuda
def test_device_clone_factory(ubq_system):
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda", torch.cuda.current_device())

    src = CartesianAtomicCoordinateProvider.build_for(ubq_system)

    # Device defaults and device clone
    clone = CartesianAtomicCoordinateProvider.build_for(src)
    assert clone.device == src.device
    assert clone.device == cpu_device
    assert clone.coords.device == cpu_device

    # Device can be overridden
    clone = CartesianAtomicCoordinateProvider.build_for(src, device=cuda_device)
    assert clone.device != src.device
    assert clone.device == cuda_device
    assert clone.coords.device == cuda_device

    src = KinematicAtomicCoordinateProvider.build_for(ubq_system)

    # Device defaults and device clone
    clone = KinematicAtomicCoordinateProvider.build_for(src)
    assert clone.device == src.device
    assert clone.device == cpu_device
    assert clone.dofs.device == cpu_device

    # Can not chance device for kinematic providers
    with pytest.raises(ValueError):
        clone = KinematicAtomicCoordinateProvider.build_for(src, device=cuda_device)


def test_coord_clone_factory(ubq_system):
    src = CartesianAtomicCoordinateProvider.build_for(ubq_system)

    ### coords are copied, not referenced
    clone = CartesianAtomicCoordinateProvider.build_for(src)
    torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    # not reactive by write, need to assign
    clone.coords[0] += 1
    clone.coords = clone.coords

    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    ### Can't initialize kin from cart
    with pytest.raises(AttributeError):
        clone = KinematicAtomicCoordinateProvider.build_for(src)

    src = KinematicAtomicCoordinateProvider.build_for(ubq_system)

    ### dofs are copied, not referenced
    clone = KinematicAtomicCoordinateProvider.build_for(src)
    torch.testing.assert_allclose(src.dofs, clone.dofs, atol=0, rtol=0)
    torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    # not reactive by write, need to assign
    clone.dofs[10] += 1
    clone.dofs = clone.dofs

    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(src.dofs, clone.dofs, atol=0, rtol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    ### cart from kin copies coords
    clone = CartesianAtomicCoordinateProvider.build_for(src)
    torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    clone.coords[0] += 1
    clone.coords = clone.coords

    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    ### requires_grad is copied, but can be overridden
    src = CartesianAtomicCoordinateProvider.build_for(ubq_system)
    assert src.coords.requires_grad is True

    src = CartesianAtomicCoordinateProvider.build_for(ubq_system, requires_grad=False)
    assert src.coords.requires_grad is False

    clone = CartesianAtomicCoordinateProvider.build_for(src)
    assert clone.coords.requires_grad is src.coords.requires_grad
    assert clone.coords.requires_grad is False

    clone = CartesianAtomicCoordinateProvider.build_for(src, requires_grad=True)
    assert clone.coords.requires_grad is not src.coords.requires_grad
    assert clone.coords.requires_grad is True

    ### requires_grad is copied, but can be overridden
    src = KinematicAtomicCoordinateProvider.build_for(ubq_system)
    assert src.dofs.requires_grad is True

    src = KinematicAtomicCoordinateProvider.build_for(ubq_system, requires_grad=False)
    assert src.dofs.requires_grad is False

    clone = KinematicAtomicCoordinateProvider.build_for(src)
    assert clone.dofs.requires_grad is src.dofs.requires_grad
    assert clone.dofs.requires_grad is False

    clone = KinematicAtomicCoordinateProvider.build_for(src, requires_grad=True)
    assert clone.dofs.requires_grad is not src.dofs.requires_grad
    assert clone.dofs.requires_grad is True


def test_coord_clone_factory_from_stacked_systems(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    cacp = CartesianAtomicCoordinateProvider.build_for(twoubq)

    assert cacp.coords.shape == (2, cacp.system_size, 3)


def test_non_uniform_sized_stacked_system_coord_factory(ubq_res):
    sys1 = PackedResidueSystem.from_residues(ubq_res[:6])
    sys2 = PackedResidueSystem.from_residues(ubq_res[:8])
    sys3 = PackedResidueSystem.from_residues(ubq_res[:4])

    twoubq = PackedResidueSystemStack((sys1, sys2, sys3))
    cacp = CartesianAtomicCoordinateProvider.build_for(twoubq)

    assert cacp.coords.shape == (3, sys2.coords.shape[0], 3)
