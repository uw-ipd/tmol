import pytest
import typing
import numpy

import torch

from tmol import (
    PoseStack,
    PackedBlockTypes,
    canonical_form_from_pdb,
    pose_stack_from_canonical_form,
)
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
)
from tmol.types.torch import Tensor

from tmol.kinematics.fold_forest import FoldForest, EdgeType

from tmol.kinematics.script_modules import PoseStackKinematicsModule
from tmol.kinematics.operations import inverseKin

from tmol.tests.torch import requires_cuda


def kop_gradcheck_report(kop, start_dofs, eps=2e-3, atol=1e-5, rtol=1e-3):
    # we only minimize the "rbdel" dofs
    minimizable_dofs = start_dofs[:, :6]

    def eval_kin(dofs_x):
        dofsfull = start_dofs.clone()
        dofsfull[:, :6] = dofs_x
        return kop(dofsfull)

    torch.autograd.gradcheck(eval_kin, minimizable_dofs, atol=atol, rtol=rtol)


def kincoords_and_dofs_for_pose_stack_system(
    pose_stack: PoseStack, kinematics_module: PoseStackKinematicsModule, torch_device
):
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
    return kincoords, dofs


@pytest.fixture
def pose_stack_system1(
    ubq_pdb: str, torch_device: torch.device
) -> typing.Tuple[PoseStack, FoldForest]:
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=0, residue_end=2
    )
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)
    # ff_roots = numpy.full((1,), 0, dtype=int)  # residue 0 is the root
    ff_n_edges = numpy.full(
        (1,), 1, dtype=int
    )  # one edge for the single Pose in the PoseStack
    ff_edges = numpy.zeros((1, 2, 4), dtype=int)
    ff_edges[0, 0, 0] = EdgeType.root_jump
    ff_edges[0, 0, 1] = -1
    ff_edges[0, 0, 2] = 0
    ff_edges[0, 1, 0] = EdgeType.polymer
    ff_edges[0, 1, 1] = 0
    ff_edges[0, 1, 2] = 1

    fold_forest = FoldForest(
        max_n_edges=1,
        n_edges=ff_n_edges,
        edges=ff_edges,
    )
    return pose_stack, fold_forest


@pytest.fixture
def pose_stack_gradcheck_test_system1(
    pose_stack_system1: typing.Tuple[PoseStack, FoldForest], torch_device: torch.device
) -> typing.Tuple[
    PoseStack,
    PoseStackKinematicsModule,
    Tensor[torch.float64][:, 3],
    Tensor[torch.float64],
]:
    pose_stack, fold_forest = pose_stack_system1

    kinematics_module = PoseStackKinematicsModule(
        pose_stack,
        fold_forest,
    )
    kincoords, dofs = kincoords_and_dofs_for_pose_stack_system(
        pose_stack, kinematics_module, torch_device
    )

    return (pose_stack, kinematics_module, kincoords, dofs)


@pytest.fixture
def pose_stack_system2(
    ubq_pdb: str, torch_device: torch.device
) -> typing.Tuple[PoseStack, FoldForest]:
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=0, residue_end=6
    )
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

    # capital letter H fold forest
    # "*" designates the root-jump residue
    #
    # 0       3
    # ^       ^
    # |       |
    # 1* ---> 4
    # |       |
    # v       v
    # 2       5
    ff_edges = numpy.zeros((1, 6, 4), dtype=int)
    ff_edges[0, 0, 0] = EdgeType.polymer
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = EdgeType.polymer
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = EdgeType.jump
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 4

    ff_edges[0, 3, 0] = EdgeType.polymer
    ff_edges[0, 3, 1] = 4
    ff_edges[0, 3, 2] = 3

    ff_edges[0, 4, 0] = EdgeType.polymer
    ff_edges[0, 4, 1] = 4
    ff_edges[0, 4, 2] = 5

    ff_edges[0, 5, 0] = EdgeType.root_jump
    ff_edges[0, 5, 1] = -1
    ff_edges[0, 5, 2] = 1

    fold_forest = FoldForest.from_edges(ff_edges)
    return pose_stack, fold_forest


@pytest.fixture
def pose_stack_gradcheck_test_system2(
    pose_stack_system2: typing.Tuple[PoseStack, FoldForest], torch_device: torch.device
) -> typing.Tuple[
    PoseStack,
    PoseStackKinematicsModule,
    Tensor[torch.float64][:, 3],
    Tensor[torch.float64],
]:
    pose_stack, fold_forest = pose_stack_system2

    kinematics_module = PoseStackKinematicsModule(
        pose_stack,
        fold_forest,
    )
    kincoords, dofs = kincoords_and_dofs_for_pose_stack_system(
        pose_stack, kinematics_module, torch_device
    )

    return (pose_stack, kinematics_module, kincoords, dofs)


def test_pose_stack_kinematics_module_smoke(
    pose_stack_gradcheck_test_system1, torch_backward_coverage, torch_device
):
    """Smoke test of kinematic operation with backward-pass code coverage."""
    _1, kinematics_module, _2, dofs = pose_stack_gradcheck_test_system1

    coords = kinematics_module(dofs.raw)
    coords.register_hook(torch_backward_coverage)

    total = torch.sum(coords[:, :])
    total.backward()

    assert dofs.raw.grad is not None


def test_pose_stack_kinematic_torch_op_gradcheck_perturbed(
    pose_stack_gradcheck_test_system1,
):
    pose_stack, kinematics_module, kincoords, dofs = pose_stack_gradcheck_test_system1

    torch.random.manual_seed(1663)
    start_dofs = (
        (dofs.raw + ((torch.rand_like(dofs.raw) - 0.5) * 0.01))
        .clone()
        .detach()
        .requires_grad_(True)
    )

    def func(dofs):
        return torch.sum(kinematics_module(dofs)[:, :])

    kop_gradcheck_report(func, start_dofs)


def test_pose_stack_kinematic_torch_op_gradcheck(
    pose_stack_gradcheck_test_system1, torch_device
):
    pose_stack, kinematics_module, kincoords, dofs = pose_stack_gradcheck_test_system1
    kop_gradcheck_report(kinematics_module, dofs.raw)


@requires_cuda
def test_pose_stack_kinematics_op_device(pose_stack_system1, torch_device):
    if torch_device.type != "cpu":
        return
    cpu_device = torch_device
    cuda_device = torch.device("cuda")

    cpu_pose_stack, fold_forest = pose_stack_system1
    cpu_kinematics_module = PoseStackKinematicsModule(
        cpu_pose_stack,
        fold_forest,
    )

    cpu_pbt = cpu_pose_stack.packed_block_types
    cuda_packed_block_types = PackedBlockTypes.from_restype_list(
        cpu_pbt.chem_db,
        cpu_pbt.restype_set,
        cpu_pbt.active_block_types,
        cuda_device,
    )

    def _to_cuda(x):
        return x.to(cuda_device)

    # TO DO: make moving a PoseStack to the device more efficient!
    cuda_pose_stack = PoseStack(
        packed_block_types=cuda_packed_block_types,
        coords=_to_cuda(cpu_pose_stack.coords),
        block_coord_offset=_to_cuda(cpu_pose_stack.block_coord_offset),
        block_coord_offset64=_to_cuda(cpu_pose_stack.block_coord_offset64),
        inter_residue_connections=_to_cuda(cpu_pose_stack.inter_residue_connections),
        inter_residue_connections64=_to_cuda(
            cpu_pose_stack.inter_residue_connections64
        ),
        inter_block_bondsep=_to_cuda(cpu_pose_stack.inter_block_bondsep),
        inter_block_bondsep64=_to_cuda(cpu_pose_stack.inter_block_bondsep64),
        block_type_ind=_to_cuda(cpu_pose_stack.block_type_ind),
        block_type_ind64=_to_cuda(cpu_pose_stack.block_type_ind64),
        device=cuda_device,
    )
    cuda_kinematics_module = PoseStackKinematicsModule(
        cuda_pose_stack,
        fold_forest,
    )

    cpu_kincoords, cpu_dofs = kincoords_and_dofs_for_pose_stack_system(
        cpu_pose_stack, cpu_kinematics_module, cpu_device
    )
    cuda_kincoords, cuda_dofs = kincoords_and_dofs_for_pose_stack_system(
        cuda_pose_stack, cuda_kinematics_module, cuda_device
    )

    assert cpu_kinematics_module.kmd.forest.id.device.type == "cpu"
    assert cuda_kinematics_module.kmd.forest.id.device.type == "cuda"

    # backwards scans/nodes/gens:
    torch.testing.assert_close(
        cpu_kinematics_module.nodes_b, cuda_kinematics_module.nodes_b.to(cpu_device)
    )
    torch.testing.assert_close(
        cpu_kinematics_module.scans_b, cuda_kinematics_module.scans_b.to(cpu_device)
    )
    torch.testing.assert_close(
        cpu_kinematics_module.gens_b, cuda_kinematics_module.gens_b
    )

    # Passing tensors of incorrect device for op errors
    with pytest.raises(RuntimeError):
        cpu_kinematics_module(cuda_dofs.raw)

    with pytest.raises(RuntimeError):
        cuda_kinematics_module(cpu_dofs.raw)

    # let's assert that the coordinates are the same for CPU and CUDA calculations:
    cpu_coords = cpu_kinematics_module(cpu_dofs.raw)
    cuda_coords = cuda_kinematics_module(cuda_dofs.raw)
    torch.testing.assert_close(cpu_coords, cuda_coords.to(cpu_device))

    # let's trigger a call to backwards on both the CPU and GPU and
    # make sure the calculated gradients are the same

    cpu_total = torch.sum(cpu_coords[:, :])
    cpu_total.backward()
    cpu_grads = cpu_dofs.raw.grad

    cuda_total = torch.sum(cuda_coords[:, :])
    cuda_total.backward()
    cuda_grads = cuda_dofs.raw.grad

    torch.testing.assert_close(cpu_grads, cuda_grads.to(cpu_device))
