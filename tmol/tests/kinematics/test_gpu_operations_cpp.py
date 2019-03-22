import pytest

import numpy
import torch
import numba

from tmol.kinematics.operations import DOFTransforms, backwardKin, forwardKin

from tmol.kinematics.builder import KinematicBuilder

from tmol.kinematics.scan_ordering import KinTreeScanOrdering

from tmol.tests.torch import requires_cuda


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

    refold_ordering = tree_reordering_cpp


@requires_cuda
@pytest.mark.benchmark(group="kinematic_op_micro_refold")
def test_refold_values_cpp(benchmark, big_system):
    target_device = torch.device("cuda")
    kintree = system_kintree(big_system)

    tcoords = torch.tensor(big_system.coords[kintree.id]).to(device=target_device)
    tkintree = kintree.to(device=target_device)
    bkin = backwardKin(tkintree, tcoords)

    refold_ordering = KinTreeScanOrdering.calculate_from_kintree(tkintree)

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
