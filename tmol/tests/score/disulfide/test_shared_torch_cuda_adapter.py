import importlib.util
from pathlib import Path
import sys

import torch


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("_shared_disulfide_torch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(dtype, requires_grad=False):
    coords = torch.tensor(
        [
            [-1.0, 1.0, 0.2],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.05, 0.1, 0.1],
            [2.2, 1.0, 0.3],
            [3.0, 1.3, 0.9],
        ],
        dtype=dtype,
        requires_grad=requires_grad,
    )
    pairs = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]], dtype=torch.int32)
    values = torch.tensor(
        [
            2.03,
            0.15,
            -0.2,
            0.1,
            1.2,
            1.8,
            0.2,
            1.1,
            -1.5,
            -0.1,
            0.8,
            1.4,
            0.1,
            -1.2,
            0.7,
            -0.2,
            0.1,
            0.9,
            0.05,
            1.3,
            1.0,
            0.6,
            0.4,
            0.8,
            1.1,
            0.3,
            0.15,
            0.9,
            -1.0,
            -0.05,
            1.0,
            1.0,
            1.0,
        ],
        dtype=dtype,
    )
    params = torch.stack((values, values.clone()))
    params[1, [13, 16, 19]] *= -1
    params[1, 32] = -1
    return coords, pairs, params


def score_and_gradient(coords, pairs, params, recompute):
    energy = torch.ops.tmol_disulfide_shared.prepared(coords, pairs, params, recompute)
    (gradient,) = torch.autograd.grad(
        energy, coords, torch.tensor([0.7], device=coords.device, dtype=coords.dtype)
    )
    return energy.detach(), gradient


def main():
    adapter = load_adapter(Path(sys.argv[1]))
    common_cuda = Path(sys.argv[2])
    operator = "tmol_disulfide_shared::prepared"
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(operator, "CUDA")
    try:
        adapter._register_cuda_common("")
    except RuntimeError as error:
        assert "path is empty" in str(error)
    else:
        raise AssertionError("empty CUDA common-library path was accepted")
    adapter._register_cuda_common(str(common_cuda))
    adapter._register_cuda_common(str(common_cuda))
    assert torch._C._dispatch_has_kernel_for_dispatch_key(operator, "CUDA")
    assert torch._C._dispatch_has_kernel_for_dispatch_key(operator, "AutogradCUDA")

    stream = torch.cuda.Stream()
    for dtype in (torch.float32, torch.float64):
        cpu_coords, cpu_pairs, cpu_params = inputs(dtype, True)
        reference = [
            score_and_gradient(
                cpu_coords.detach().clone().requires_grad_(True),
                cpu_pairs,
                cpu_params,
                recompute,
            )
            for recompute in (False, True)
        ]
        with torch.cuda.stream(stream):
            coords = cpu_coords.detach().cuda().requires_grad_(True)
            pairs = cpu_pairs.cuda()
            params = cpu_params.cuda()
            torch.cuda._sleep(10_000_000)
            coords = coords.clone()
            coords[2, 0] += 0.125
            results = [
                score_and_gradient(
                    coords.detach().clone().requires_grad_(True),
                    pairs,
                    params,
                    recompute,
                )
                for recompute in (False, True)
            ]
        stream.synchronize()

        shifted = cpu_coords.detach().clone()
        shifted[2, 0] += 0.125
        shifted_reference = [
            score_and_gradient(
                shifted.clone().requires_grad_(True),
                cpu_pairs,
                cpu_params,
                recompute,
            )
            for recompute in (False, True)
        ]
        tolerance = (
            dict(rtol=3e-5, atol=3e-5)
            if dtype == torch.float32
            else dict(rtol=2e-12, atol=2e-12)
        )
        for result, expected in zip(results, shifted_reference):
            torch.testing.assert_close(result[0].cpu(), expected[0], **tolerance)
            torch.testing.assert_close(result[1].cpu(), expected[1], **tolerance)
        torch.testing.assert_close(results[0][0], results[1][0], **tolerance)
        torch.testing.assert_close(results[0][1], results[1][1], **tolerance)
        assert not torch.allclose(results[0][0].cpu(), reference[0][0])


if __name__ == "__main__":
    main()
