"""Pair Dunbrack annotation setup against the exact preceding sampler class."""

import argparse
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import subprocess
import time

BASELINE = "35350d6e3"
SOURCE = "tmol/pack/rotamer/dunbrack/_dunbrack_chi_sampler.py"


def baseline_class():
    module = importlib.import_module("tmol.pack.rotamer.dunbrack._dunbrack_chi_sampler")
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{SOURCE}"], text=True)
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "DunbrackChiSampler"
    )
    namespace = dict(vars(module))
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "baseline_sampler", "exec"),
        namespace,
    )
    return namespace["DunbrackChiSampler"], hashlib.sha256(source.encode()).hexdigest()


def fresh_pbt(template):
    pbt = copy.copy(template)
    pbt.active_block_types = [copy.copy(bt) for bt in template.active_block_types]
    for owner in [pbt, *pbt.active_block_types]:
        for name in tuple(vars(owner)):
            if name.startswith(("dun_sampler_", "_dun_sampler_")):
                delattr(owner, name)
    return pbt


def main():
    import attr
    import numpy
    import torch
    from tmol.database import ParameterDatabase
    from tmol.io import default_packed_block_types
    from tmol.pack.rotamer.dunbrack import DunbrackChiSampler
    from tmol.score.dunbrack import DunbrackParamResolver

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    template = default_packed_block_types(device)
    resolver = DunbrackParamResolver.from_database(
        ParameterDatabase.get_default().scoring.dun, device
    )
    baseline, baseline_sha = baseline_class()
    classes = dict(baseline=baseline, candidate=DunbrackChiSampler)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def setup(cls, pbt):
        sampler = cls(resolver)
        for bt in pbt.active_block_types:
            sampler.annotate_residue_type(bt)
        sampler.annotate_packed_block_types(pbt)
        return sampler

    prepared = {}
    for name, cls in classes.items():
        pbt = fresh_pbt(template)
        setup(cls, pbt)
        prepared[name] = pbt
    tensor_bytes = dict(baseline=0, candidate=0)
    for a, b in zip(
        [prepared["baseline"], *prepared["baseline"].active_block_types],
        [prepared["candidate"], *prepared["candidate"].active_block_types],
        strict=True,
    ):
        for field in attr.fields(type(a.dun_sampler_cache)):
            old = getattr(a.dun_sampler_cache, field.name)
            new = getattr(b.dun_sampler_cache, field.name)
            if isinstance(new, int):
                assert int(old) == new
            elif isinstance(old, torch.Tensor):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            elif isinstance(old, numpy.ndarray):
                numpy.testing.assert_array_equal(old, new)
            else:
                assert int(old) == int(new)
            for name, value in (("baseline", old), ("candidate", new)):
                if isinstance(value, torch.Tensor):
                    tensor_bytes[name] += value.numel() * value.element_size()
                elif isinstance(value, numpy.ndarray):
                    tensor_bytes[name] += value.nbytes

    timings = {name: [] for name in classes}
    for round_index in range(5):
        order = list(classes)[:: -1 if round_index % 2 else 1]
        for name in order:
            measured = []
            for _ in range(5):
                pbt = fresh_pbt(template)
                sync()
                start = time.perf_counter()
                sampler = setup(classes[name], pbt)
                sync()
                measured.append(time.perf_counter() - start)
                del sampler, pbt
            timings[name].append(statistics.median(measured))
    medians = {name: statistics.median(values) for name, values in timings.items()}
    result = dict(
        baseline=subprocess.check_output(
            ["git", "rev-parse", BASELINE], text=True
        ).strip(),
        baseline_source_sha256=baseline_sha,
        candidate_source_sha256=hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest(),
        device=str(device),
        torch=torch.__version__,
        n_types=template.n_types,
        n_atoms=sum(bt.n_atoms for bt in template.active_block_types),
        all_annotation_fields_exact=True,
        annotation_tensor_and_array_bytes=tensor_bytes,
        round_medians_seconds=timings,
        median_seconds=medians,
        baseline_over_candidate=medians["baseline"] / medians["candidate"],
        limits=(
            "Sampler construction plus first RT/PBT annotation only; five alternating "
            "warm rounds of five samples. Database/resolver construction and fresh "
            "chemical-object copies are outside timing. Tensor/array byte totals "
            "sum annotation fields; they exclude Python metadata, allocator overhead "
            "and process RSS. No end-to-end sampling or peak-memory claim."
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
