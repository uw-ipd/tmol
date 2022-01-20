import pytest
import typing

import torch

from tmol.types.torch import Tensor

from tmol.kinematics.datatypes import KinTree
from tmol.kinematics.script_modules import KinematicModule
from tmol.kinematics.operations import inverseKin

from tmol.system.packed import PackedResidueSystem
from tmol.chemical.restypes import Residue
from tmol.system.kinematics import KinematicDescription

from tmol.tests.torch import requires_cuda


@pytest.mark.benchmark(group="kinematic_forward_op")
def test_kinematic_torch_op_forward(benchmark, ubq_system, torch_device):
    tsys = ubq_system
    tkin = KinematicDescription.for_system(
        tsys.system_size, tsys.bonds, (tsys.torsion_metadata,)
    )
    kincoords = tkin.extract_kincoords(tsys.coords).to(torch_device)
    tkintree = tkin.kintree.to(torch_device)

    tdofs = inverseKin(tkintree, kincoords, requires_grad=True)
    kop = KinematicModule(tkintree, torch_device)

    @benchmark
    def refold_kincoords():
        return kop(tdofs.raw)

    torch.testing.assert_allclose(refold_kincoords, kincoords)
    assert refold_kincoords.device.type == torch_device.type


@pytest.mark.benchmark(group="kinematic_backward_op")
def test_kinematic_torch_op_backward_benchmark(benchmark, ubq_system, torch_device):
    tsys = ubq_system
    tkin = KinematicDescription.for_system(
        tsys.system_size, tsys.bonds, (tsys.torsion_metadata,)
    )
    kincoords = tkin.extract_kincoords(tsys.coords).to(torch_device)
    tkintree = tkin.kintree.to(torch_device)

    tdofs = inverseKin(tkintree, kincoords, requires_grad=True)
    kop = KinematicModule(tkintree, torch_device)

    refold_kincoords = kop(tdofs.raw)
    total = refold_kincoords.sum()

    @benchmark
    def refold_grad():
        total.backward(retain_graph=True)

    torch.testing.assert_allclose(refold_kincoords, kincoords)
    assert refold_kincoords.device.type == torch_device.type


@pytest.fixture
def gradcheck_test_system(
    ubq_res: typing.Sequence[Residue],
) -> typing.Tuple[KinTree, Tensor[torch.float64][:, 3]]:

    tsys = PackedResidueSystem.from_residues(ubq_res[:4])
    tkin = KinematicDescription.for_system(
        tsys.system_size, tsys.bonds, (tsys.torsion_metadata,)
    )

    return (tkin.kintree, tkin.extract_kincoords(tsys.coords))


def kop_gradcheck_report(kop, start_dofs, eps=2e-3, atol=1e-5, rtol=1e-3):
    # we only minimize the "rbdel" dofs
    minimizable_dofs = start_dofs[:, :6]

    def eval_kin(dofs_x):
        dofsfull = start_dofs.clone()
        dofsfull[:, :6] = dofs_x
        return kop(dofsfull)

    result = eval_kin(minimizable_dofs)

    # Extract results from torch/autograd/gradcheck.py
    from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian

    (analytical,), reentrant, correct_grad_sizes = get_analytical_jacobian(
        (minimizable_dofs,), result
    )
    numerical = get_numerical_jacobian(
        eval_kin, minimizable_dofs, minimizable_dofs, eps=eps
    )

    torch.testing.assert_allclose(analytical, numerical, atol=atol, rtol=rtol)


def test_kinematic_torch_op_gradcheck_perturbed(gradcheck_test_system, torch_device):
    kintree, kincoords = gradcheck_test_system
    tkintree = kintree.to(torch_device)
    tkincoords = kincoords.to(torch_device)

    tdofs = inverseKin(tkintree, tkincoords, requires_grad=True)
    kop = KinematicModule(tkintree, torch_device)

    torch.random.manual_seed(1663)
    start_dofs = (
        (tdofs.raw + ((torch.rand_like(tdofs.raw) - .5) * .01))
        .clone()
        .detach()
        .requires_grad_(True)
    )

    kop_gradcheck_report(kop, start_dofs)


def test_kinematic_torch_op_gradcheck(gradcheck_test_system, torch_device):
    kintree, kincoords = gradcheck_test_system
    tkintree = kintree.to(torch_device)
    tkincoords = kincoords.to(torch_device)

    tdofs = inverseKin(tkintree, tkincoords, requires_grad=True)
    kop = KinematicModule(tkintree, torch_device)

    kop_gradcheck_report(kop, tdofs.raw)


def test_kinematic_torch_op_smoke(
    gradcheck_test_system, torch_backward_coverage, torch_device
):
    """Smoke test of kinematic operation with backward-pass code coverage."""
    kintree, kincoords = gradcheck_test_system
    tkintree = kintree.to(torch_device)
    tkincoords = kincoords.to(torch_device)

    tdofs = inverseKin(tkintree, tkincoords, requires_grad=True)
    kop = KinematicModule(tkintree, torch_device)

    coords = kop(tdofs.raw)
    coords.register_hook(torch_backward_coverage)

    total = coords.sum()
    total.backward()

    assert tdofs.raw.grad is not None


@requires_cuda
def test_kinematic_op_device(gradcheck_test_system):
    kintree, kincoords = gradcheck_test_system
    assert kincoords.device.type == "cpu"

    tdofs = inverseKin(kintree, kincoords, requires_grad=True)

    cpu_kop = KinematicModule(kintree, torch.device("cpu"))
    assert cpu_kop.kintree.device.type == "cpu"
    cpu_kop(tdofs.raw.to(torch.device("cpu")))

    cuda_kop = KinematicModule(kintree, torch.device("cuda"))
    assert cuda_kop.kintree.device.type == "cuda"
    cuda_kop(tdofs.raw.to(torch.device("cuda")))

    # Passing tensors of incorrect device for op errors
    with pytest.raises(RuntimeError):
        cpu_kop(tdofs.raw.to(torch.device("cuda")))

    with pytest.raises(RuntimeError):
        cuda_kop(tdofs.raw.to(torch.device("cpu")))
