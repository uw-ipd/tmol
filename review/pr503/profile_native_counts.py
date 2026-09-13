"""Measure native Dunbrack count checks against an isolated preceding extension."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

BASELINE = "0db5eb5d79d98250e0f3ec84955231c60fc42c51"
ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tmol/tests/pack/rotamer/dunbrack"


def baseline_module(directory):
    from tmol._load_ext import load_module

    directory.mkdir(parents=True, exist_ok=True)
    relative = "tmol/pack/rotamer/dunbrack/dispatch.impl.hh"
    original = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT, text=True
    )
    sources = {
        "dispatch.impl.hh": original.replace(
            "namespace dunbrack {", "namespace pr503_count_baseline {"
        )
    }
    for name in (
        "compiled.pybind.cpp",
        "test_cpu.cpp",
        "test_cuda.cu",
        "test.hh",
        "test.impl.hh",
    ):
        source = (TESTS / name).read_text()
        source = source.replace(
            "DunbrackChiSamplerTester", "Pr503BaselineDunbrackTester"
        )
        source = source.replace(
            "pack::rotamer::dunbrack::", "pack::rotamer::pr503_count_baseline::"
        )
        source = source.replace(
            "<tmol/pack/rotamer/dunbrack/dispatch.impl.hh>", '"dispatch.impl.hh"'
        )
        source = source.replace(
            "<tmol/tests/pack/rotamer/dunbrack/test.hh>", '"test.hh"'
        )
        sources[name] = source
    for name, source in sources.items():
        path = directory / name
        if not path.exists() or path.read_text() != source:
            path.write_text(source)
    module = load_module(
        "tmol.pr503.baseline_dun_counts_0db5",
        str(directory / "loader.py"),
        ["compiled.pybind.cpp", "test_cpu.cpp", "test_cuda.cu"],
        "unused_in_jit_mode",
    )
    provenance = dict(
        original_header_sha256=hashlib.sha256(original.encode()).hexdigest(),
        generated_sources_sha256={
            name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in sources.items()
        },
        directory=str(directory),
        changes="Only the baseline namespace, test bridge class names and local include paths are changed to prevent native symbol interposition; numerical function bodies are unchanged.",
    )
    return module, provenance


def main():  # noqa: C901
    import torch
    from tmol.tests.pack.rotamer.dunbrack.test_dunbrack_chi_sampler import get_compiled

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    baseline, provenance = baseline_module(args.baseline_dir)
    candidate = get_compiled()
    functions = dict(
        baseline=baseline.count_expanded_rotamers,
        candidate=candidate.count_expanded_rotamers,
    )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def inputs(n, overflow=False):
        def t(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        if overflow:
            base = t([1])
            n = 1
            table = t([[0, -1]])
            chi = t([2])
            library_chi = t([0])
            expanded = t([[0, 0]])
            extra = t([[65536, 65536]])
        else:
            base = (torch.arange(n, device=device, dtype=torch.int32) % 7) + 1
            table = t([[0, 0]] * n)
            chi = t([4] * n)
            library_chi = t([2])
            expanded = t([[1, 0, 0, 0]] * n)
            extra = t([[0, 0, 3, 2]] * n)
        expansions = torch.zeros_like(base)
        products = torch.zeros_like(extra)
        counts = base.clone()
        offsets = torch.zeros_like(base)
        return base, (
            chi,
            table,
            library_chi,
            expanded,
            extra,
            expansions,
            products,
            counts,
            offsets,
        )

    # Prove these are distinct native implementations, not interposed symbols.
    _, witness = inputs(1, True)
    assert functions["baseline"](*witness) == 0
    witness[-2].fill_(1)
    try:
        functions["candidate"](*witness)
    except RuntimeError as error:
        assert "Dunbrack sampling count" in str(error)
    else:
        raise AssertionError("Candidate must reject the baseline's wrapped count")

    rows = []
    for n in (1, 128, 4096):
        base, call_args = inputs(n)
        expected = None
        for function in functions.values():
            call_args[-2].copy_(base)
            call_args[-1].zero_()
            total = function(*call_args)
            result = [total, *[x.clone() for x in call_args[-4:]]]
            if expected is None:
                expected = result
            else:
                assert expected[0] == result[0]
                for a, b in zip(expected[1:], result[1:]):
                    assert torch.equal(a, b)
        timings = {name: [] for name in functions}
        for round_index in range(5):
            order = (
                list(functions) if round_index % 2 == 0 else list(reversed(functions))
            )
            for name in order:
                function = functions[name]
                measured = []
                for repetition in range(21):
                    call_args[-2].copy_(base)
                    call_args[-1].zero_()
                    sync()
                    start = time.perf_counter()
                    function(*call_args)
                    sync()
                    elapsed = time.perf_counter() - start
                    if repetition:
                        measured.append(elapsed)
                timings[name].append(statistics.median(measured))
        medians = {name: statistics.median(values) for name, values in timings.items()}
        rows.append(
            dict(
                n_buildable_types=n,
                exact_parity=True,
                round_medians_seconds=timings,
                median_seconds=medians,
                candidate_over_baseline=medians["candidate"] / medians["baseline"],
            )
        )
        print(n, medians, flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                baseline=BASELINE,
                torch=torch.__version__,
                device=str(device),
                distinct_implementations_verified=True,
                provenance=provenance,
                rows=rows,
                limits="Native count/offset stage only. Compilation, inputs, count reset, synchronization before timing and parity verification are outside timing. Completion synchronization is included. Five alternating warm rounds with twenty samples per round. No memory or end-to-end sampling speed claim.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
