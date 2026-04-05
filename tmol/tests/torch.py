import threading

import pytest

import torch
import torch.cuda

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()

requires_cuda = pytest.mark.skipif(not cuda_available, reason="Requires cuda.")
requires_mps = pytest.mark.skipif(not mps_available, reason="Requires MPS (Apple Silicon).")


def zero_padded_counts(counts):
    from math import log10, floor

    max_count = max(counts)
    width = int(floor(log10(max_count))) + 1
    return [str(x).zfill(width) for x in counts]


def _device_params():
    """Build the parameter list for the torch_device fixture."""
    params = ["cpu"]
    if cuda_available:
        params.append(pytest.param("cuda", marks=requires_cuda))
    if mps_available:
        params.append(pytest.param("mps", marks=requires_mps))
    return params


@pytest.fixture(params=_device_params())
def torch_device(request):
    """Parametrized test fixture covering cpu, cuda, and mps torch devices."""

    if request.param == "cpu":
        device = torch.device("cpu")
    elif request.param == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    elif request.param == "mps":
        device = torch.device("mps", 0)
    else:
        raise NotImplementedError(f"Unknown device param: {request.param}")

    # Perform a "warmup" computation on the device, ensuring that it is
    # initialized and available.
    torch.arange(100, device=device).sum()

    return device


def cuda_not_implemented(f):
    """Parametrize 'torch_device' as an xfail via NotImplementedError."""
    return pytest.mark.parametrize(
        "torch_device",
        [
            (torch.device("cpu")),
            pytest.param(
                torch.device("cuda"),
                marks=[
                    requires_cuda,
                    pytest.mark.xfail(strict=True, raises=NotImplementedError),
                ],
            ),
        ],
    )(f)


def mps_not_implemented(f):
    """Parametrize 'torch_device' as an xfail for MPS via NotImplementedError."""
    return pytest.mark.parametrize(
        "torch_device",
        [
            (torch.device("cpu")),
            pytest.param(
                torch.device("mps", 0),
                marks=[
                    requires_mps,
                    pytest.mark.xfail(strict=True, raises=NotImplementedError),
                ],
            ),
        ],
    )(f)


@pytest.fixture
def torch_backward_coverage(cov):
    """Torch hook to enable coverage in backward pass.

    Returns a hook function used to enable coverage tracing during
    pytorch backward passes. Torch runs all backward passes in a
    non-main thread, not spawned by the standard 'threading'
    interface, so coverage does not trace the thread.

    Example:

    result = custom_func(input)

    # enable the hook
    result.register_hook(torch_backward_coverage)

    # call backward via sum so hook fires before custom_op backward
    result.sum().backward()
    """

    if cov:
        print("cov collector???", hasattr(cov, "collector"))
        cov._collector.added_tracers = {threading.get_ident()}

        def add_tracer(_):
            tid = threading.get_ident()
            if tid not in cov._collector.added_tracers:
                print(f"pytorch backward trace: {tid}")
                cov._collector.added_tracers.add(tid)
                cov._collector._start_tracer()

    else:

        def add_tracer(_):
            pass

    return add_tracer
