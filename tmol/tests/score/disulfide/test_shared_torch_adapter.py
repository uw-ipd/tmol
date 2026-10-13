import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys

import torch


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("_shared_disulfide_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(requires_grad):
    coords = torch.tensor(
        [
            [-1.0, 1.0, 0.2],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.05, 0.1, 0.1],
            [2.2, 1.0, 0.3],
            [3.0, 1.3, 0.9],
        ],
        dtype=torch.float64,
        requires_grad=requires_grad,
    )
    pairs = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1]], dtype=torch.int32)
    values = [
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
    ]
    params = torch.tensor([values, values], dtype=torch.float64)
    params[1, [13, 16, 19]] *= -1
    params[1, 32] = -1
    return coords, pairs, params


def gradient(recompute):
    coords, pairs, params = inputs(True)
    energy = torch.ops.tmol_disulfide_shared.prepared(coords, pairs, params, recompute)
    saved_count = len(energy.grad_fn.saved_tensors)
    (grad,) = torch.autograd.grad(energy, coords, torch.tensor([0.7]))
    return energy.detach(), grad, saved_count


def main():
    module_path = Path(sys.argv[1])
    common_path = Path(sys.argv[2])
    load_adapter(module_path)

    common = ctypes.cdll.LoadLibrary(str(common_path))
    common.tmol_disulfide_cpu_abi_version.argtypes = []
    common.tmol_disulfide_cpu_abi_version.restype = ctypes.c_uint32
    assert common.tmol_disulfide_cpu_abi_version() == 1

    coords, pairs, params = inputs(False)
    inference = torch.ops.tmol_disulfide_shared.prepared(coords, pairs, params, False)
    assert inference.shape == (1,)
    assert inference.grad_fn is None
    assert torch.isfinite(inference).all()

    no_grad_coords, pairs, params = inputs(True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(saved.append, lambda tensor: tensor):
        with torch.no_grad():
            no_grad_energy = torch.ops.tmol_disulfide_shared.prepared(
                no_grad_coords, pairs, params, False
            )
    assert no_grad_energy.grad_fn is None
    assert not no_grad_energy.requires_grad
    assert saved == []

    tracked_energy, tracked_grad, tracked_saved = gradient(False)
    recompute_energy, recompute_grad, recompute_saved = gradient(True)
    assert tracked_saved == 2  # pairs plus retained pair derivatives
    assert recompute_saved == 3  # coords, pairs, and params
    torch.testing.assert_close(tracked_energy, inference, rtol=0, atol=1e-12)
    torch.testing.assert_close(recompute_energy, inference, rtol=0, atol=1e-12)
    torch.testing.assert_close(recompute_grad, tracked_grad, rtol=0, atol=1e-12)

    epsilon = 1e-6
    plus_coords, pairs, params = inputs(False)
    minus_coords = plus_coords.clone()
    plus_coords[2, 0] += epsilon
    minus_coords[2, 0] -= epsilon
    plus = torch.ops.tmol_disulfide_shared.prepared(plus_coords, pairs, params, False)
    minus = torch.ops.tmol_disulfide_shared.prepared(minus_coords, pairs, params, False)
    finite_difference = 0.7 * (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(
        tracked_grad[2, 0], finite_difference[0], rtol=1e-5, atol=2e-5
    )

    exported = subprocess.check_output(
        ["nm", "-D", "--defined-only", common_path], text=True
    )
    symbols = [line.split()[-1] for line in exported.splitlines() if line]
    assert symbols
    assert all(symbol.startswith("tmol_disulfide_cpu_") for symbol in symbols)

    undefined = subprocess.check_output(
        ["nm", "-D", "--undefined-only", module_path], text=True
    )
    assert "tmol_disulfide_cpu_abi_version" in undefined
    assert "_ZN2at" not in undefined
    assert "_ZN3c10" not in undefined
    linkage = subprocess.check_output(["ldd", common_path], text=True)
    assert "libtorch" not in linkage


if __name__ == "__main__":
    main()
