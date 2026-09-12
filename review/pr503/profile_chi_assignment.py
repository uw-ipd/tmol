"""Paired chi-assignment timings with exact DOF parity and CUDA memory peaks."""

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

from tmol.chemical._restypes import ResidueTypeSet
from tmol.database import ParameterDatabase
from tmol.tests.pack.rotamer.test_chi_assignment import chi_assignment_inputs

BASELINE = "458feb22dc571c02960b3c93ff9e67a3edd246bd"
SOURCE = "tmol/pack/rotamer/_chi_sampler.py"
BUILDER = "tmol/pack/rotamer/_build_rotamers.py"
NAME = "assign_chi_dofs_from_samples"


def previous_function():
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{SOURCE}"], text=True)
    builder = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{BUILDER}"], text=True
    )
    namespace = dict(vars(importlib.import_module("tmol.pack.rotamer._build_rotamers")))
    namespace.update(vars(importlib.import_module("tmol.pack.rotamer._chi_sampler")))
    old_builder = ast.parse(
        builder.replace("_ring_chi_phi_c_corrections", "_baseline_ring_corrections")
    )
    node = next(
        n
        for n in old_builder.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_build_baseline_ring_corrections"
    )
    node.name = "_build_ring_chi_phi_c_corrections"
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "previous_ring_table", "exec"
        ),
        namespace,
    )
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == NAME
    )
    node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "previous_chi_assignment", "exec"
        ),
        namespace,
    )
    return namespace[NAME], {
        SOURCE: hashlib.sha256(source.encode()).hexdigest(),
        BUILDER: hashlib.sha256(builder.encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if args.device == "cuda"
        else torch.device("cpu")
    )
    paths = (SOURCE, BUILDER, "tmol/tests/pack/rotamer/test_chi_assignment.py")
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    old, old_hashes = previous_function()
    module = importlib.import_module("tmol.pack.rotamer._chi_sampler")
    new = getattr(module, NAME)
    database = ParameterDatabase.get_default()
    restypes = ResidueTypeSet.from_database(database.chemical)
    original = chi_assignment_inputs(database, restypes, device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    results = []
    for factor in (1, 1000):
        values = list(original)
        values[1] = values[1].repeat(factor)
        values[2] = (
            original[2][None, :]
            + torch.arange(factor, device=device)[:, None] * len(original[1])
        ).flatten()
        values[3] = values[3] * factor
        values[4] = values[4].repeat(factor)
        counts = values[0].n_atoms[values[1]].long()
        values[5] = counts.cumsum(0) - counts
        for index in (6, 7):
            values[index] = values[index].repeat(factor, 1)
        values[8] = torch.zeros((int(counts.sum()) + 1, 9), device=device)
        old(*values)
        expected = values[8].clone()
        values[8].zero_()
        new(*values)
        torch.testing.assert_close(values[8], expected, rtol=0, atol=0)
        del expected
        timings = {name: [] for name in ("baseline", "candidate")}
        functions = {"baseline": old, "candidate": new}
        for round_index in range(5):
            for name in list(functions)[:: -1 if round_index % 2 else 1]:
                samples = []
                for _ in range(10):
                    sync()
                    start = time.perf_counter()
                    functions[name](*values)
                    sync()
                    samples.append(time.perf_counter() - start)
                timings[name].append(statistics.median(samples))
        peaks = {}
        if device.type == "cuda":
            for name, fn in functions.items():
                sync()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                fn(*values)
                sync()
                peaks[name] = torch.cuda.max_memory_allocated() - before
        table = values[0]._ring_chi_phi_c_corrections
        results.append(
            dict(
                conformers=len(values[2]),
                chi_entries=int((values[6] != -1).sum()),
                seconds={k: statistics.median(v) for k, v in timings.items()},
                round_seconds=timings,
                additional_cuda_peak_bytes=peaks,
                ring_table_bytes=table.numel() * table.element_size(),
            )
        )
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()
    )
    report = dict(
        baseline=BASELINE,
        baseline_source_sha256=old_hashes,
        candidate_source_sha256=hashes,
        device=str(device),
        torch=torch.__version__,
        results=results,
        limits="Assignment only; includes sparse indexing and in-place DOF writes. Five alternating rounds of ten synchronized samples. Real ILE/PRO gapped/reordered samples and 1000 synthetic repeats; full DOF arrays match exactly. First table construction excluded. Ring offsets replace a same-size retained host array with one device tensor; KFO storage is shared with normal rotamer construction. CUDA peak is additional allocated tensor memory above warmed tables and prepared inputs, not reserved/process memory. No whole-packer speed claim.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
