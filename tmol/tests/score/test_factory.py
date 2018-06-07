import pytest

import copy

import numpy
import torch

from tmol.database import ParameterDatabase
from tmol.utility.reactive import reactive_attrs
from tmol.tests.torch import requires_cuda
from tmol.score import (
    TotalScoreGraph,
    CartesianAtomicCoordinateProvider,
    KinematicAtomicCoordinateProvider,
)


@reactive_attrs
class TCart(CartesianAtomicCoordinateProvider, TotalScoreGraph):
    """Cart total."""
    pass


@reactive_attrs
class TKin(KinematicAtomicCoordinateProvider, TotalScoreGraph):
    pass


@requires_cuda
def test_device_clone_factory(ubq_system):
    cpu_device = torch.device("cpu")
    cuda_device = torch.device("cuda", torch.cuda.current_device())

    src = TCart.build_for(ubq_system)

    # Device defaults and device clone
    clone = TCart.build_for(src)
    assert clone.device == src.device
    assert clone.device == cpu_device
    assert clone.coords.device == cpu_device

    # Device can be overridden
    clone = TCart.build_for(src, device=cuda_device)
    assert clone.device != src.device
    assert clone.device == cuda_device
    assert clone.coords.device == cuda_device

    src = TKin.build_for(ubq_system)

    # Device defaults and device clone
    clone = TKin.build_for(src)
    assert clone.device == src.device
    assert clone.device == cpu_device
    assert clone.dofs.device == cpu_device

    # Can not chance device for kinematic providers
    with pytest.raises(ValueError):
        clone = TKin.build_for(src, device=cuda_device)


def test_coord_clone_factory(ubq_system):
    src = TCart.build_for(ubq_system)

    ### coords are copied, not referenced
    clone = TCart.build_for(src)
    torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    # not reactive by write, need to assign
    clone.coords[0] += 1
    clone.coords = clone.coords

    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    ### Can't initialize kin from cart
    with pytest.raises(AttributeError):
        clone = TKin.build_for(src)

    src = TKin.build_for(ubq_system)

    ### dofs are copied, not referenced
    clone = TKin.build_for(src)
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
    clone = TCart.build_for(src)
    torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    clone.coords[0] += 1
    clone.coords = clone.coords

    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(src.coords, clone.coords, atol=0, rtol=0)

    ### requires_grad is copied, but can be overridden
    src = TCart.build_for(ubq_system)
    assert src.coords.requires_grad is True

    src = TCart.build_for(ubq_system, requires_grad=False)
    assert src.coords.requires_grad is False

    clone = TCart.build_for(src)
    assert clone.coords.requires_grad is src.coords.requires_grad
    assert clone.coords.requires_grad is False

    clone = TCart.build_for(src, requires_grad=True)
    assert clone.coords.requires_grad is not src.coords.requires_grad
    assert clone.coords.requires_grad is True

    ### requires_grad is copied, but can be overridden
    src = TKin.build_for(ubq_system)
    assert src.dofs.requires_grad is True

    src = TKin.build_for(ubq_system, requires_grad=False)
    assert src.dofs.requires_grad is False

    clone = TKin.build_for(src)
    assert clone.dofs.requires_grad is src.dofs.requires_grad
    assert clone.dofs.requires_grad is False

    clone = TKin.build_for(src, requires_grad=True)
    assert clone.dofs.requires_grad is not src.dofs.requires_grad
    assert clone.dofs.requires_grad is True


def test_bonded_atom_clone_factory(ubq_system):
    src: TCart = TCart.build_for(ubq_system)

    # Bond graph is referenced
    clone = TCart.build_for(src)
    assert clone.bonds is src.bonds
    numpy.testing.assert_allclose(src.bonds, clone.bonds)
    numpy.testing.assert_allclose(
        src.bonded_path_length, clone.bonded_path_length
    )

    clone.bonds = clone.bonds[:len(clone.bonds) // 2]
    assert clone.bonds is not src.bonds
    with pytest.raises(AssertionError):
        numpy.testing.assert_allclose(src.bonds, clone.bonds)
    with pytest.raises(AssertionError):
        numpy.testing.assert_allclose(
            src.bonded_path_length, clone.bonded_path_length
        )

    # Atom types are referenced
    assert clone.atom_types is src.atom_types


def test_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default())

    # Parameter database defaults
    src: TCart = TCart.build_for(ubq_system)
    assert src.parameter_database is ParameterDatabase.get_default()

    # Parameter database is overridden via kwarg
    src: TCart = TCart.build_for(ubq_system, parameter_database=clone_db)
    assert src.parameter_database is clone_db

    # Parameter database is referenced on clone
    clone: TCart = TCart.build_for(src)
    assert clone.parameter_database is src.parameter_database

    # Parameter database is overriden on clone via kwarg
    clone: TCart = TCart.build_for(
        src, parameter_database=ParameterDatabase.get_default()
    )
    assert clone.parameter_database is not src.parameter_database
    assert clone.parameter_database is ParameterDatabase.get_default()


def test_ljlk_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.ljlk)

    src: TCart = TCart.build_for(ubq_system)
    assert src.ljlk_database is ParameterDatabase.get_default().scoring.ljlk

    # Parameter database is overridden via kwarg
    src: TCart = TCart.build_for(ubq_system, ljlk_database=clone_db)
    assert src.ljlk_database is clone_db

    # Parameter database is referenced on clone
    clone: TCart = TCart.build_for(src)
    assert clone.ljlk_database is src.ljlk_database

    # Parameter database is overriden on clone via kwarg
    clone: TCart = TCart.build_for(
        src, ljlk_database=ParameterDatabase.get_default().scoring.ljlk
    )
    assert clone.ljlk_database is not src.ljlk_database
    assert clone.ljlk_database is ParameterDatabase.get_default().scoring.ljlk


def test_hbond_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default().scoring.hbond)

    src: TCart = TCart.build_for(ubq_system)
    assert src.hbond_database is ParameterDatabase.get_default().scoring.hbond

    # Parameter database is overridden via kwarg
    src: TCart = TCart.build_for(ubq_system, hbond_database=clone_db)
    assert src.hbond_database is clone_db

    # Parameter database is referenced on clone
    clone: TCart = TCart.build_for(src)
    assert clone.hbond_database is src.hbond_database

    # Parameter database is overriden on clone via kwarg
    clone: TCart = TCart.build_for(
        src, hbond_database=ParameterDatabase.get_default().scoring.hbond
    )
    assert clone.hbond_database is not src.hbond_database
    assert clone.hbond_database is ParameterDatabase.get_default().scoring.hbond
