import pytest
import typing
import numpy

import torch

from tmol import PoseStack, canonical_form_from_pdb, pose_stack_from_canonical_form
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
)
from tmol.types.torch import Tensor

from tmol.kinematics.datatypes import KinForest
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.script_modules import KinematicModule, PoseStackKinematicModule
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
    tkinforest = tkin.kinforest.to(torch_device)

    tdofs = inverseKin(tkinforest, kincoords, requires_grad=True)
    kop = KinematicModule(tkinforest, torch_device)

    @benchmark
    def refold_kincoords():
        return kop(tdofs.raw)

    torch.testing.assert_close(refold_kincoords, kincoords)
    assert refold_kincoords.device.type == torch_device.type

    print("tkinforest.id[:10]", tkinforest.id[:10])
    print("tkinforest.parent[:10]", tkinforest.parent[:10])
    print("tkinforest.doftype[:10]", tkinforest.doftype[:10])
    print("scans", kop.scans_f[:10])
    print("gens", kop.gens_f)
    print("nodes", kop.nodes_f[:10])


@pytest.mark.benchmark(group="kinematic_backward_op")
def test_kinematic_torch_op_backward_benchmark(benchmark, ubq_system, torch_device):
    tsys = ubq_system
    tkin = KinematicDescription.for_system(
        tsys.system_size, tsys.bonds, (tsys.torsion_metadata,)
    )
    kincoords = tkin.extract_kincoords(tsys.coords).to(torch_device)
    tkinforest = tkin.kinforest.to(torch_device)

    tdofs = inverseKin(tkinforest, kincoords, requires_grad=True)
    kop = KinematicModule(tkinforest, torch_device)

    refold_kincoords = kop(tdofs.raw)
    total = torch.sum(refold_kincoords[:, :])

    @benchmark
    def refold_grad():
        total.backward(retain_graph=True)

    torch.testing.assert_close(refold_kincoords, kincoords)
    assert refold_kincoords.device.type == torch_device.type


@pytest.fixture
def gradcheck_test_system(
    ubq_res: typing.Sequence[Residue],
) -> typing.Tuple[KinForest, Tensor[torch.float64][:, 3]]:
    tsys = PackedResidueSystem.from_residues(ubq_res[:4])
    tkin = KinematicDescription.for_system(
        tsys.system_size, tsys.bonds, (tsys.torsion_metadata,)
    )

    return (tkin.kinforest, tkin.extract_kincoords(tsys.coords))


def kop_gradcheck_report(kop, start_dofs, eps=2e-3, atol=1e-5, rtol=1e-3):
    # we only minimize the "rbdel" dofs
    minimizable_dofs = start_dofs[:, :6]

    def eval_kin(dofs_x):
        dofsfull = start_dofs.clone()
        dofsfull[:, :6] = dofs_x
        return kop(dofsfull)

    torch.autograd.gradcheck(eval_kin, minimizable_dofs, atol=atol, rtol=rtol)


def test_kinematic_torch_op_gradcheck_perturbed(gradcheck_test_system, torch_device):
    kinforest, kincoords = gradcheck_test_system
    tkinforest = kinforest.to(torch_device)
    tkincoords = kincoords.to(torch_device)

    tdofs = inverseKin(tkinforest, tkincoords, requires_grad=True)
    kop = KinematicModule(tkinforest, torch_device)

    torch.random.manual_seed(1663)
    start_dofs = (
        (tdofs.raw + ((torch.rand_like(tdofs.raw) - 0.5) * 0.01))
        .clone()
        .detach()
        .requires_grad_(True)
    )

    kop_gradcheck_report(kop, start_dofs)


def test_kinematic_torch_op_gradcheck(gradcheck_test_system, torch_device):
    kinforest, kincoords = gradcheck_test_system
    tkinforest = kinforest.to(torch_device)
    tkincoords = kincoords.to(torch_device)

    tdofs = inverseKin(tkinforest, tkincoords, requires_grad=True)
    kop = KinematicModule(tkinforest, torch_device)

    kop_gradcheck_report(kop, tdofs.raw)


def test_kinematic_torch_op_smoke(
    gradcheck_test_system, torch_backward_coverage, torch_device
):
    """Smoke test of kinematic operation with backward-pass code coverage."""
    kinforest, kincoords = gradcheck_test_system
    tkinforest = kinforest.to(torch_device)
    tkincoords = kincoords.to(torch_device)

    tdofs = inverseKin(tkinforest, tkincoords, requires_grad=True)
    kop = KinematicModule(tkinforest, torch_device)

    coords = kop(tdofs.raw)
    coords.register_hook(torch_backward_coverage)

    total = torch.sum(coords[:, :])
    total.backward()

    assert tdofs.raw.grad is not None


@pytest.fixture
def pose_stack_gradcheck_test_system1(
    ubq_pdb: str, torch_device: torch.device
) -> typing.Tuple[PoseStack, PoseStackKinematicModule]:
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=0, residue_end=6
    )
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

    # capital letter H fold forest
    # 0       3
    # ^       ^
    # |       |
    # 1* ---> 4
    # |       |
    # v       v
    # 2       5
    ff_roots = numpy.full((1,), 1, dtype=int)  # residue 1 is the root
    ff_n_edges = numpy.full(
        (1, 1), 5, dtype=int
    )  # five edges for the single Pose in the PoseStack
    ff_edges = numpy.zeros((1, 5, 3), dtype=int)
    ff_edges[0, 0, 0] = 0
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = 0
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = 1
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 4

    ff_edges[0, 3, 0] = 0
    ff_edges[0, 3, 1] = 4
    ff_edges[0, 3, 2] = 3

    ff_edges[0, 4, 0] = 0
    ff_edges[0, 4, 1] = 4
    ff_edges[0, 4, 2] = 5

    fold_forest = FoldForest(
        max_n_edges=5,
        n_edges=ff_n_edges,
        edges=ff_edges,
        roots=ff_roots,
    )

    kinematics_module = PoseStackKinematicModule(
        pose_stack,
        fold_forest,
    )

    return (pose_stack, kinematics_module)


def test_pose_stack_kinematics_module_smoke(
    pose_stack_gradcheck_test_system1, torch_backward_coverage, torch_device
):
    """Smoke test of kinematic operation with backward-pass code coverage."""
    pose_stack, kinematics_module = pose_stack_gradcheck_test_system1
    kinforest = kinematics_module.kmd.forest

    kincoords = torch.zeros(
        (kinematics_module.kmd.forest.id.shape[0], 3),
        dtype=torch.float64,
        device=torch_device,
    )
    kincoords[1:] = pose_stack.coords.view(-1, 3)[
        kinematics_module.kmd.forest.id[1:]
    ].to(torch.float64)

    dofs = inverseKin(kinforest, kincoords, requires_grad=True)

    coords = kinematics_module(dofs.raw)
    coords.register_hook(torch_backward_coverage)

    total = torch.sum(coords[:, :])
    total.backward()

    assert dofs.raw.grad is not None


@requires_cuda
def test_kinematic_op_device(gradcheck_test_system):
    kinforest, kincoords = gradcheck_test_system
    assert kincoords.device.type == "cpu"

    tdofs = inverseKin(kinforest, kincoords, requires_grad=True)

    cpu_kop = KinematicModule(kinforest, torch.device("cpu"))
    assert cpu_kop.kinforest.device.type == "cpu"
    cpu_kop(tdofs.raw.to(torch.device("cpu")))

    cuda_kop = KinematicModule(kinforest, torch.device("cuda"))
    assert cuda_kop.kinforest.device.type == "cuda"
    cuda_kop(tdofs.raw.to(torch.device("cuda")))

    # Passing tensors of incorrect device for op errors
    with pytest.raises(RuntimeError):
        cpu_kop(tdofs.raw.to(torch.device("cuda")))

    with pytest.raises(RuntimeError):
        cuda_kop(tdofs.raw.to(torch.device("cpu")))
