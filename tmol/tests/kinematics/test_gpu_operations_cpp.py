import pytest

import attr
import torch
import numpy

from tmol.kinematics.operations import backwardKin, forwardKin
from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.scan_ordering import KinTreeScanOrdering
from tmol.tests.torch import requires_cuda

from tmol.kinematics.compiled import compiled


def system_kintree(target_system):
    return (
        KinematicBuilder()
        .append_connected_component(
            *KinematicBuilder.bonds_to_connected_component(0, target_system.bonds)
        )
        .kintree
    )


@pytest.mark.benchmark(group="score_setup")
def test_refold_data_construction(benchmark, ubq_system):
    kintree = system_kintree(ubq_system)

    @benchmark
    def tree_reordering_cpp() -> KinTreeScanOrdering:
        return KinTreeScanOrdering.calculate_from_kintree(kintree)

    kinorder = tree_reordering_cpp

    # ensure the tree is reasonable:
    #   - dimensionality match
    #   - all connected nodes are parent->child
    natms = kintree.id.shape[0]
    generated = numpy.zeros(natms, dtype=numpy.int32)
    ngens = len(kinorder.forward_scan_paths.nodes)
    assert ngens == len(kinorder.forward_scan_paths.scans)

    for i in range(ngens):
        nscans = kinorder.forward_scan_paths.scans[i].shape[0]
        assert kinorder.forward_scan_paths.scans[i][0] == 0
        for j in range(nscans):
            # tag the root(s) as generated
            if i == 0:
                generated[kinorder.forward_scan_paths.nodes[i][0]] += 1

            scanstart = kinorder.forward_scan_paths.scans[i][j]
            scanstop = kinorder.forward_scan_paths.nodes[i].shape[0]
            if j != nscans - 1:
                scanstop = kinorder.forward_scan_paths.scans[i][j + 1]
            for k in range(scanstart, scanstop - 1):
                parent = kinorder.forward_scan_paths.nodes[i][k]
                child = kinorder.forward_scan_paths.nodes[i][k + 1]
                assert kintree.parent[child].to(dtype=torch.int) == parent

                # tag the child as visited
                generated[child] += 1

    #  ensure all nodes are generated exactly once
    for i in range(1, natms):
        assert generated[i] == 1


@requires_cuda
@pytest.mark.benchmark(group="kinematic_op_micro_refold")
def test_refold_values_cpp(benchmark, big_system):
    target_device = torch.device("cuda")
    kintree = system_kintree(big_system)

    tcoords = torch.tensor(big_system.coords[kintree.id]).to(device=target_device)
    tkintree = kintree.to(device=target_device)
    bkin = backwardKin(tkintree, tcoords)

    KinTreeScanOrdering.calculate_from_kintree(tkintree)

    @benchmark
    def parallel_refold_hts_cpp():
        return forwardKin(tkintree, bkin.dofs)

    # fold via cpu and gpu, ensuring results match
    hts_cuda = parallel_refold_hts_cpp

    bkin = backwardKin(kintree, tcoords.cpu())

    hts_cpu = forwardKin(kintree, bkin.dofs)

    assert hts_cuda.hts.device.type == "cuda"
    assert hts_cpu.hts.device.type == "cpu"
    torch.testing.assert_allclose(hts_cuda.hts.cpu(), hts_cpu.hts)


@requires_cuda
@pytest.mark.benchmark(group="kinematic_op_micro_derivsum")
def test_derivsum_values_cpp(benchmark, big_system):
    target_device = torch.device("cuda")
    torch.manual_seed(1663)

    kintree_cpu = system_kintree(big_system)

    coords_cuda = torch.tensor(big_system.coords[kintree_cpu.id]).to(
        device=target_device
    )
    kintree_cuda = kintree_cpu.to(device=target_device)

    dsc_dx = (torch.rand_like(coords_cuda) * 2) - 1
    f1s = torch.cross(coords_cuda, coords_cuda - dsc_dx)
    f2s = dsc_dx.clone()
    f1f2s_cuda = torch.cat((f1s, f2s), 1)
    f1f2s_cpu = f1f2s_cuda.cpu()

    ordering_cuda = KinTreeScanOrdering.for_kintree(kintree_cuda)

    @benchmark
    def parallel_f1f2_sums_cpp():
        f1f2s_cuda_copy = f1f2s_cuda.clone()
        compiled.segscan_f1f2s(
            f1f2s_cuda_copy, **attr.asdict(ordering_cuda.backward_scan_paths)
        )
        return f1f2s_cuda_copy

    f1f2s_cuda = parallel_f1f2_sums_cpp

    # same calc on CPU
    ordering_cpu = KinTreeScanOrdering.for_kintree(kintree_cpu)
    compiled.segscan_f1f2s(f1f2s_cpu, **attr.asdict(ordering_cpu.backward_scan_paths))

    assert f1f2s_cuda.device.type == "cuda"
    assert f1f2s_cpu.device.type == "cpu"
    torch.testing.assert_allclose(f1f2s_cuda.cpu(), f1f2s_cpu)
