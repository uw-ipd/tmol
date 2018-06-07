import pytest
import typing

import pandas
import torch
import numpy

from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian

from tmol.types.torch import Tensor

from tmol.kinematics.operations import SegScanStrategy
from tmol.kinematics.datatypes import KinTree
from tmol.kinematics.metadata import DOFMetadata, DOFTypes
from tmol.kinematics.torch_op import KinematicOp

from tmol.system.packed import PackedResidueSystem
from tmol.system.restypes import Residue
from tmol.system.kinematics import KinematicDescription


@pytest.fixture(params=["efficient", "min_depth"])
def scan_strategy(request):
    return SegScanStrategy(request.param)


@pytest.mark.benchmark(
    group="kinematic_forward_op",
)
def test_torsion_refold_ubq(
        benchmark,
        ubq_system,
        torch_device,
        scan_strategy,
):
    tsys = ubq_system
    tkin = KinematicDescription.for_system(tsys.bonds, tsys.torsion_metadata)

    torsion_dofs = tkin.dof_metadata[
        (tkin.dof_metadata.dof_type == DOFTypes.bond_torsion)
    ]

    kincoords = tkin.extract_kincoords(tsys.coords).to(torch_device)

    kop = KinematicOp.from_coords(
        tkin.kintree,
        torsion_dofs,
        kincoords,
        scan_strategy=scan_strategy,
    )

    @benchmark
    def refold_kincoords():
        return kop.apply(kop.src_mobile_dofs)

    torch.testing.assert_allclose(refold_kincoords, kincoords)


@pytest.fixture
def gradcheck_test_system(
        ubq_res: typing.Sequence[Residue],
) -> typing.Tuple[KinTree,
                  DOFMetadata,
                  Tensor("f8")[:, 3],
                  ]:
    tsys = PackedResidueSystem.from_residues(ubq_res[:4])
    tkin = KinematicDescription.for_system(tsys.bonds, tsys.torsion_metadata)

    return (
        tkin.kintree,
        tkin.dof_metadata,
        tkin.extract_kincoords(tsys.coords),
    )


def kop_gradcheck_report(
        kop, dofs, start_dofs, eps=1e-6, atol=1e-5, rtol=1e-3
):
    initial_gradcheck = torch.autograd.gradcheck(
        kop, (start_dofs, ), raise_exception=False
    )

    if initial_gradcheck:
        # Initial exhausive gradcheck succeeded.
        return

    # Intiial gradcheck failed, generate a more specific failure report.

    result = kop(start_dofs)

    # Extract results from torch/autograd/gradcheck.py
    (analytical, ), reentrant, correct_grad_sizes = get_analytical_jacobian(
        (start_dofs, ), result
    )
    numerical = get_numerical_jacobian(kop, start_dofs, start_dofs, eps)

    a = analytical
    n = numerical

    grad_match = ((a - n).abs() <= (atol + rtol * n.abs()))
    if grad_match.all():
        return

    failures = torch.nonzero(~grad_match)
    nfailures = len(failures)
    failures = pandas.DataFrame({
        "dof_input": failures[:, 0],
        "node_idx": failures[:, 1] / 3,
        "node_coord": numpy.array(list("xyz"))[failures[:, 1] % 3],
        "analytical": a[~grad_match],
        "numerical": n[~grad_match],
    }).drop_duplicates()[[
        "dof_input", "node_idx", "node_coord", "analytical", "numerical"
    ]]

    dof_summary = dofs.to_frame()
    failing_dofs = failures["dof_input"].drop_duplicates().values

    assert nfailures == 0, (
        f"DOFs failed grad check:\n{failures}\n\n"
        f"Failed DOF metadata:\n{dof_summary.iloc[failing_dofs]}\n\n"
    )

    # Rerun gradcheck to w/ exception to return full error.
    torch.autograd.gradcheck(kop, (start_dofs, ), raise_exception=True)


def test_kinematic_torch_op_gradcheck_perturbed(gradcheck_test_system):
    kintree, dofs, kincoords = gradcheck_test_system

    # Temporary workaround for #45, disable theta for post-jump siblings
    post_root_siblings = ((dofs.parent_id == 0) &
                          (dofs.dof_type == DOFTypes.bond_angle))
    dofs = dofs[~post_root_siblings]

    kop = KinematicOp.from_coords(
        kintree,
        dofs,
        kincoords,
    )

    torch.random.manual_seed(1663)
    start_dofs = torch.tensor(
        kop.src_mobile_dofs +
        ((torch.rand_like(kop.src_mobile_dofs) - .5) * .01),
        requires_grad=True
    )

    kop_gradcheck_report(kop, dofs, start_dofs)


def test_kinematic_torch_op_gradcheck(gradcheck_test_system):
    kintree, dofs, kincoords = gradcheck_test_system

    # Temporary workaround for #45, disable theta for post-jump siblings
    post_root_siblings = ((dofs.parent_id == 0) &
                          (dofs.dof_type == DOFTypes.bond_angle))
    dofs = dofs[~post_root_siblings]

    kop = KinematicOp.from_coords(
        kintree,
        dofs,
        kincoords,
    )

    start_dofs = torch.tensor(kop.src_mobile_dofs, requires_grad=True)

    kop_gradcheck_report(kop, dofs, start_dofs)


def test_kinematic_torch_op_smoke(
        gradcheck_test_system, torch_backward_coverage
):
    kintree, dofs, kincoords = gradcheck_test_system

    kop = KinematicOp.from_coords(
        kintree,
        dofs,
        kincoords,
    )

    start_dofs = torch.tensor(kop.src_mobile_dofs, requires_grad=True)

    coords = kop(start_dofs)
    coords.register_hook(torch_backward_coverage)

    total = coords.sum()
    total.backward()

    assert start_dofs.grad is not None
