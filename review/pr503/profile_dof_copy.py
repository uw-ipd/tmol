"""Compare exact DOF-copy plans, latency and extra allocated CUDA memory."""

import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from tmol.database import ParameterDatabase
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.tests.pack.rotamer import test_build_rotamers as fixtures
from tmol.utility.tensor import exclusive_cumsum1d, stretch

BASELINE = "5060998cf9c20191a25dd44f73d8c5f89e346447"
SOURCE = "tmol/pack/rotamer/_chi_sampler.py"
NAME = "create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    files = (
        SOURCE,
        "tmol/pack/rotamer/_build_rotamers.py",
        "tmol/tests/pack/rotamer/test_build_rotamers.py",
    )
    hashes = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files
    }
    old_source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{SOURCE}"], text=True
    )
    module = importlib.import_module("tmol.pack.rotamer._chi_sampler")
    namespace = dict(
        vars(module), exclusive_cumsum1d=exclusive_cumsum1d, stretch=stretch
    )
    node = next(
        n
        for n in ast.parse(old_source).body
        if isinstance(n, ast.FunctionDef) and n.name == NAME
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "baseline_dof_copy", "exec"),
        namespace,
    )
    old, new = namespace[NAME], getattr(module, NAME)
    recorded = []

    def compare(*values):
        expected, actual = old(*values), new(*values)
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        recorded.append(values)
        return actual

    setattr(fixtures, NAME, compare)
    database = ParameterDatabase.get_default()
    sampler = create_dunbrack_sampler_from_database(database, device)
    pdb = Path("tmol/tests/data/pdb/1ubq.pdb").read_text()
    for test in (
        fixtures.test_create_dof_inds_to_copy_from_orig_to_rotamers,
        fixtures.test_create_dof_inds_to_copy_from_orig_to_rotamers2,
    ):
        test(database, pdb, device, sampler)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    results = []
    for case, inputs in enumerate(recorded):
        for factor in (1, 1000):
            values = list(inputs)
            for index in (3, 4):
                values[index] = values[index].repeat(factor)
            values[5] = torch.arange(len(values[3]), device=device)
            values[6] = values[6] * factor
            values[7] = values[3].to(torch.int32)
            counts = values[0].packed_block_types.n_atoms[values[4]].long()
            values[8] = torch.cumsum(counts, 0) - counts
            first, second = old(*values), new(*values)
            for x, y in zip(first, second):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            pairs = len(first[0])
            del first, second, x, y
            functions = {"baseline": old, "candidate": new}
            timings = {name: [] for name in functions}
            for round_index in range(5):
                for name in list(functions)[:: -1 if round_index % 2 else 1]:
                    samples = []
                    for _ in range(5):
                        sync()
                        start = time.perf_counter()
                        _result = functions[name](*values)
                        sync()
                        samples.append(time.perf_counter() - start)
                        del _result
                    timings[name].append(statistics.median(samples))
            peaks = {}
            if device.type == "cuda":
                for name, fn in functions.items():
                    sync()
                    before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    # Keep output alive until its peak allocation is recorded.
                    _result = fn(*values)  # noqa: F841
                    sync()
                    peaks[name] = torch.cuda.max_memory_allocated() - before
                    del _result
            from tmol.pack.rotamer._build_rotamers import _chi4_and_kfo_device_tables

            pbt = values[0].packed_block_types
            kfo = pbt._kinforest_device_indices
            assert _chi4_and_kfo_device_tables(pbt, device)[1] is kfo
            results.append(
                dict(
                    case=case,
                    conformers=len(values[3]),
                    exact_copy_pairs=pairs,
                    seconds={k: statistics.median(v) for k, v in timings.items()},
                    round_seconds=timings,
                    additional_cuda_peak_bytes=peaks,
                    shared_kfo_bytes=kfo.numel() * kfo.element_size(),
                )
            )
    assert all(
        hashlib.sha256(Path(name).read_bytes()).hexdigest() == value
        for name, value in hashes.items()
    )
    report = dict(
        baseline=BASELINE,
        baseline_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
        candidate_source_sha256=hashes,
        device=str(device),
        torch=torch.__version__,
        results=results,
        limits="DOF index construction only. Synthetic repeated conformers from two corrected real-chemistry fixtures; both baseline and candidate pass named-atom and offset oracles. Five alternating rounds of five samples. Exact index/order equality. First device-table construction excluded; the shared KFO tensor aliases the existing chi-correction table. Peak is additional allocated CUDA tensor memory above inputs, not reserved/process memory. Excludes setup, actual sampling, DOF transfer and scoring; no whole-packer speed claim.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
