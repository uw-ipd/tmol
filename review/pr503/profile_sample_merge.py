"""Paired prefix-merge timings and CUDA peak tensor storage, with exact parity."""

import argparse
import ast
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import time

BASELINE = "9d49606e08cb95871c7dfb59731e4231586f428e"


def baseline_merge():
    import toolz

    module = importlib.import_module("tmol.pack.rotamer._build_rotamers")
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:tmol/pack/rotamer/_build_rotamers.py"], text=True
    )
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "merge_conformer_samples"
    )
    namespace = dict(vars(module), toolz=toolz)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "baseline_merge", "exec"),
        namespace,
    )
    return namespace["merge_conformer_samples"]


def main():
    import torch
    from tmol.pack.rotamer._build_rotamers import merge_conformer_samples
    from tmol.tests.pack.rotamer.test_merge_sample_plan import samples_from_counts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    functions = dict(baseline=baseline_merge(), candidate=merge_conformer_samples)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def tensors(result):
        for item in result:
            if isinstance(item, torch.Tensor):
                yield item
            else:
                yield from tensors(item)

    results = []
    for n_types, width in ((32, 8), (1024, 32), (8192, 128)):
        generator = torch.Generator().manual_seed(503)
        counts = torch.randint(0, width, (3, n_types), generator=generator).tolist()
        samples = samples_from_counts(counts, device)
        # IncludeCurrent produces int64 indices alongside int32 chi-sampler rows.
        c, index, extra = samples[-1]
        samples[-1] = (c, index.long(), extra)
        del c, index, extra
        expected = functions["baseline"](samples)
        actual = functions["candidate"](samples)
        for a, b in zip(tensors(expected), tensors(actual), strict=True):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a, b)
        del expected, actual, a, b
        timings = {name: [] for name in functions}
        peaks = {name: [] for name in functions}
        for round_index in range(5):
            order = (
                list(functions) if round_index % 2 == 0 else list(reversed(functions))
            )
            for name in order:
                function = functions[name]
                function(samples)
                sync()
                measured = []
                for _ in range(5):
                    if device.type == "cuda":
                        initial = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                    sync()
                    start = time.perf_counter()
                    result = function(samples)
                    sync()
                    measured.append(time.perf_counter() - start)
                    if device.type == "cuda":
                        peaks[name].append(torch.cuda.max_memory_allocated() - initial)
                    del result
                timings[name].append(statistics.median(measured))
        medians = {name: statistics.median(values) for name, values in timings.items()}
        row = dict(
            n_types=n_types,
            n_rotamers=sum(sample[1].numel() for sample in samples),
            exact_parity=True,
            round_medians_seconds=timings,
            median_seconds=medians,
            baseline_over_candidate=medians["baseline"] / medians["candidate"],
            extra_peak_allocated_cuda_bytes=peaks,
        )
        results.append(row)
        print(n_types, row["n_rotamers"], medians, flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                baseline=BASELINE,
                device=str(device),
                torch=torch.__version__,
                results=results,
                limits="Merge only; inputs, chemistry, sampling, coordinates and scoring are outside timing. CUDA peaks are allocated tensor storage above the held input baseline, not reserved allocator memory or total process RSS. CPU memory is not measured. Five alternating warm rounds, five samples each.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
