"""Paired dense CPU spline timings across the stride-correctness change."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from tmol.utility import load

BASELINE = "3709b04f05ac796aa32e3745f40af9b31ec4e802"
SOURCE = "tmol/numeric/bspline_compiled/bspline.hh"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.parent / f"{args.output.stem}-source"
    directory.mkdir(exist_ok=True)
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{SOURCE}"], text=True)
    assert source.count("struct ndspline {") == 1
    (directory / "baseline.hh").write_text(
        source.replace("struct ndspline {", "struct ndspline_baseline {")
    )
    cpp = r"""
#include <tmol/numeric/bspline_compiled/bspline.hh>
#include "baseline.hh"
#include <tmol/utility/tensor/pybind.h>
using namespace tmol;
using namespace tmol::numeric::bspline;
template<int N> void bind(pybind11::module& m) {
  using Input = TView<float, N, Device::CPU>;
  using Points = TView<Eigen::Matrix<float, N, 1>, 1, Device::CPU>;
  m.def(("before" + std::to_string(N)).c_str(),
      pybind11::overload_cast<Input, Points>(&ndspline_baseline<N, 3, Device::CPU, float, int32_t>::interpolate_tv));
  m.def(("after" + std::to_string(N)).c_str(),
      pybind11::overload_cast<Input, Points>(&ndspline<N, 3, Device::CPU, float, int32_t>::interpolate_tv));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { bind<2>(m); bind<3>(m); }
"""
    path = directory / "paired.cpp"
    path.write_text(cpp)
    module = load("pr503_spline_layout_profile", [str(path.resolve())])
    generator = torch.Generator().manual_seed(503)
    results = []
    for ndim in (2, 3):
        coeffs = torch.randn((36,) * ndim, generator=generator)
        for count in (1, 10000):
            points = torch.rand((count, ndim), generator=generator) * 36
            funcs = {
                side: getattr(module, f"{side}{ndim}") for side in ("before", "after")
            }
            left, right = [func(coeffs, points) for func in funcs.values()]
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            times = {side: [] for side in funcs}
            repetitions = 100 if count == 1 else 3
            for round_index in range(7):
                for side in list(funcs)[:: -1 if round_index % 2 else 1]:
                    start = time.perf_counter()
                    for _ in range(repetitions):
                        funcs[side](coeffs, points)
                    times[side].append(
                        (time.perf_counter() - start) * 1000 / repetitions
                    )
            medians = {
                side: statistics.median(values) for side, values in times.items()
            }
            results.append(
                {
                    "dimensions": ndim,
                    "points": count,
                    "milliseconds": times,
                    "median_milliseconds": medians,
                    "speedup": medians["before"] / medians["after"],
                }
            )
    result = {
        "baseline_commit": BASELINE,
        "source_sha256": {
            "before": hashlib.sha256(source.encode()).hexdigest(),
            "after": hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
        },
        "torch": torch.__version__,
        "device": "cpu",
        "results": results,
        "limits": "Single-thread native evaluation of dense 36-bin coefficients, including output allocation and Python binding overhead. Values and derivatives agree exactly. Seven alternating warm rounds; construction and compilation excluded. This is not a CUDA or whole-pose benchmark. No reduced allocation is claimed.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
