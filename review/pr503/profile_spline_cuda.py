"""Evaluate an external spline index prototype before integrating it."""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
import torch
from tmol.utility import load

BASELINE = "ea4e906bc6e5aa881f5a56c06549c568ed10dfea"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--candidate-header",
    type=Path,
    default=Path("tmol/numeric/bspline_compiled/bspline.hh"),
)
args = parser.parse_args()
source_path = Path("tmol/numeric/bspline_compiled/bspline.hh")
baseline = subprocess.check_output(
    ["git", "show", f"{BASELINE}:{source_path}"], text=True
)
candidate = args.candidate_header.read_text()
directory = args.output.parent / (args.output.stem + "-source")
directory.mkdir(exist_ok=True)
(directory / "before.hh").write_text(
    baseline.replace("struct ndspline {", "struct ndspline_before {")
)
(directory / "after.hh").write_text(candidate)
(directory / "bind.cpp").write_text("""
#include <torch/extension.h>
at::Tensor evaluate(at::Tensor, at::Tensor, bool);
std::vector<int64_t> attributes(int, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("evaluate", &evaluate);
 m.def("attributes", &attributes);
}
""")
(directory / "kernels.cu").write_text(r"""
#include "before.hh"
#include "after.hh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
using namespace tmol;
using namespace tmol::numeric::bspline;
template<int N, bool After>
__global__ void kernel(TView<float, N, Device::CUDA> coeffs,
    TView<Eigen::Matrix<float,N,1>,1,Device::CUDA> points,
    TView<float,2,Device::CUDA> output) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=points.size(0)) return;
  using Spline = typename std::conditional<After,
    ndspline<N,3,Device::CUDA,float,int32_t>,
    ndspline_before<N,3,Device::CUDA,float,int32_t>>::type;
  auto values=Spline::interpolate(TensorAccessor<float,N,Device::CUDA>(coeffs),points[i]);
  output[i][0]=tmol::score::common::get<0>(values);
  for(int dim=0;dim<N;++dim) output[i][dim+1]=tmol::score::common::get<1>(values)[dim];
}
template<int N, bool After>
at::Tensor launch(at::Tensor coeffs,at::Tensor points) {
  auto output=at::empty({points.size(0),N+1},points.options());
  if(points.size(0)) {
    kernel<N,After><<<(points.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
      view_tensor<float,N,Device::CUDA>(coeffs),
      view_tensor<Eigen::Matrix<float,N,1>,1,Device::CUDA>(points),
      view_tensor<float,2,Device::CUDA>(output));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
at::Tensor evaluate(at::Tensor coeffs,at::Tensor points,bool after) {
  TORCH_CHECK(coeffs.is_cuda() && points.is_cuda() && coeffs.device()==points.device());
  c10::cuda::CUDAGuard guard(coeffs.device());
  TORCH_CHECK(points.dim()==2 && points.size(1)==coeffs.dim());
  if(coeffs.dim()==2) return after?launch<2,true>(coeffs,points):launch<2,false>(coeffs,points);
  TORCH_CHECK(coeffs.dim()==3);
  return after?launch<3,true>(coeffs,points):launch<3,false>(coeffs,points);
}
template<int N,bool After> std::vector<int64_t> get_attributes() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel<N,After>));
  return {a.numRegs,static_cast<int64_t>(a.localSizeBytes),a.maxThreadsPerBlock};
}
std::vector<int64_t> attributes(int n,bool after) {
 if(n==2) return after?get_attributes<2,true>():get_attributes<2,false>();
 TORCH_CHECK(n==3);
 return after?get_attributes<3,true>():get_attributes<3,false>();
}
""")
module = load(
    "pr503_spline_offsets_cuda",
    [str(directory / "bind.cpp"), str(directory / "kernels.cu")],
)
torch.manual_seed(503)
small_grid_cases = []
for shape in ((1, 2), (2, 3), (1, 2, 3), (2, 3, 4)):
    coeffs = torch.randn(shape, device="cuda")
    axes = [
        torch.arange(-2 * n, 2 * n + 1, device="cuda", dtype=torch.float32)
        for n in shape
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
        -1, len(shape)
    )
    points = torch.cat(
        (
            grid,
            torch.nextafter(grid, torch.full_like(grid, -torch.inf)),
            torch.nextafter(grid, torch.full_like(grid, torch.inf)),
            torch.rand((64, len(shape)), device="cuda")
            * torch.tensor(shape, device="cuda"),
        )
    )
    before, after = [module.evaluate(coeffs, points, flag) for flag in (False, True)]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    small_grid_cases.append(
        {"shape": shape, "points": len(points), "values_and_derivatives_exact": True}
    )
results = []
for ndim in (2, 3):
    for padded in (False, True):
        storage = torch.randn((40 if padded else 36,) * ndim, device="cuda")
        coeffs = storage[tuple(slice(0, 36) for _ in range(ndim))]
        for count in (1000, 100000):
            points = torch.rand((count, ndim), device="cuda") * 36
            left, right = [
                module.evaluate(coeffs, points, after) for after in (False, True)
            ]
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            del left, right
            times = {side: [] for side in ("before", "after")}
            for _ in range(5):
                for after in (False, True):
                    module.evaluate(coeffs, points, after)
            for rnd in range(7):
                for side in list(times)[:: -1 if rnd % 2 else 1]:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    for _ in range(100):
                        module.evaluate(coeffs, points, side == "after")
                    end.record()
                    end.synchronize()
                    times[side].append(start.elapsed_time(end) / 100)
            peaks = {}
            for side in times:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                initial = torch.cuda.memory_allocated()
                out = module.evaluate(coeffs, points, side == "after")
                torch.cuda.synchronize()
                peaks[side] = torch.cuda.max_memory_allocated() - initial
                del out
            medians = {
                side: statistics.median(values) for side, values in times.items()
            }
            results.append(
                {
                    "dimensions": ndim,
                    "padded": padded,
                    "points": count,
                    "milliseconds": times,
                    "median_milliseconds": medians,
                    "speedup": medians["before"] / medians["after"],
                    "peak_allocated_bytes_above_inputs": peaks,
                    "kernel_attributes": {
                        side: dict(
                            zip(
                                ("registers", "local_bytes_per_thread", "max_threads"),
                                module.attributes(ndim, side == "after"),
                            )
                        )
                        for side in times
                    },
                }
            )
result = {
    "baseline_commit": BASELINE,
    "small_grid_cases": small_grid_cases,
    "checkout_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_sha256": {
        "before": hashlib.sha256(baseline.encode()).hexdigest(),
        "after": hashlib.sha256(candidate.encode()).hexdigest(),
    },
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "results": results,
    "limits": "External float32 kernel harness, 128 threads/block, actual padded strides, values and all derivatives exactly equal. Seven alternating warm CUDA-event rounds of 100 launches. Includes output allocation/launch sequence; no full-pose claim. Allocator peak excludes source tensors and CUDA-managed register/local-memory backing; kernel attributes report local bytes separately.",
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
