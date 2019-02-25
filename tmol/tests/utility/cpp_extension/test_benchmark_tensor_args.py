import torch
from tmol.tests.benchmark import make_subfixture

from tmol.utility.cpp_extension import load, modulename


def test_tview_args(benchmark):
    extension = load(modulename(__name__), __file__.replace(".py", ".cpp"))

    for narg, f in extension.scalar_args.items():
        args = [torch.empty(10) for _ in range(narg)]
        kwargs = {f"t{n}": torch.empty(10) for n in range(narg)}

        make_subfixture(benchmark, f".args.scalar.{narg}")(lambda: f(*args))
        make_subfixture(benchmark, f".kwargs.scalar.{narg}")(lambda: f(**kwargs))

    for narg, f in extension.vec_args.items():
        args = [torch.empty((10, 3)) for _ in range(narg)]
        kwargs = {f"t{n}": torch.empty((10, 3)) for n in range(narg)}

        make_subfixture(benchmark, f".args.vec.{narg}")(lambda: f(*args))
        make_subfixture(benchmark, f".kwargs.vec.{narg}")(lambda: f(**kwargs))
