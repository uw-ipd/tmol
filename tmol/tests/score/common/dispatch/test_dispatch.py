import numpy
import torch

from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available


def test_complex_dispatch():
    compiled = load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["test.cpp", "test.pybind.cpp", "test.cu"])
        ),
    )
    vals = torch.arange(20, dtype=torch.int32)
    boundaries = torch.zeros(2, dtype=torch.int32)
    boundaries[0] = 0
    boundaries[1] = 9

    res = compiled.complex_dispatch(vals, boundaries)
    forall, reduction, ex_scan, inc_scan, ex_scan2, final, ex_seg_scan = res

    numpy.testing.assert_equal(forall.numpy(), (vals * 2).numpy())

    reduction_gold = 20 * 19 / 2
    assert reduction == reduction_gold

    ex_scan_gold = numpy.array(
        [
            0,
            0,
            1,
            3,
            6,
            10,
            15,
            21,
            28,
            36,
            45,
            55,
            66,
            78,
            91,
            105,
            120,
            136,
            153,
            171,
        ],
        dtype=numpy.int32,
    )
    numpy.testing.assert_equal(ex_scan_gold, ex_scan.cpu())

    inc_scan_gold = numpy.array(
        [
            0,
            1,
            3,
            6,
            10,
            15,
            21,
            28,
            36,
            45,
            55,
            66,
            78,
            91,
            105,
            120,
            136,
            153,
            171,
            190,
        ],
        dtype=numpy.int32,
    )
    numpy.testing.assert_equal(inc_scan_gold, inc_scan.cpu())

    numpy.testing.assert_equal(ex_scan_gold, ex_scan2.cpu())

    ex_seg_scan_gold = numpy.array(
        [0, 0, 1, 3, 6, 10, 15, 21, 28, 0, 9, 19, 30, 42, 55, 69, 84, 100, 117, 135],
        dtype=numpy.int32,
    )
    numpy.testing.assert_equal(ex_seg_scan_gold, ex_seg_scan.cpu())
