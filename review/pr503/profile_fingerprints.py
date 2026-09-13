"""Compare cold/warm mainchain-copy setup and all canonical transfer mappings."""

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

BASELINE = "dd99ab29e6823c731c25a6662cfa5c3b4cffecd0"
SOURCE = "tmol/pack/rotamer/_mainchain_fingerprint.py"


def baseline_functions():
    module = importlib.import_module("tmol.pack.rotamer._mainchain_fingerprint")
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{SOURCE}"], text=True)
    selected = {
        "MCFingerprints",
        "annotate_residue_type_with_sampler_fingerprints",
        "find_max_length_fp_among_res_samplers",
        "find_unique_fingerprints",
        "create_mainchain_fingerprint",
        "create_non_sidechain_fingerprint",
        "_mc_inds_for_chiral_mc_atom",
    }
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected
    ]
    namespace = dict(vars(module))
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "baseline_fingerprints", "exec"
        ),
        namespace,
    )
    return (
        namespace["annotate_residue_type_with_sampler_fingerprints"],
        namespace["find_unique_fingerprints"],
    ), hashlib.sha256(source.encode()).hexdigest()


def fresh_pbt(template):
    pbt = copy.copy(template)
    pbt.active_block_types = [copy.copy(bt) for bt in template.active_block_types]
    for owner in [pbt, *pbt.active_block_types]:
        for name in tuple(vars(owner)):
            if name.startswith(("mc_", "_mc_")):
                delattr(owner, name)
    return pbt


def main():
    source_sha = hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()
    import attr
    import numpy
    import torch
    from tmol.database import ParameterDatabase
    from tmol.io import default_packed_block_types
    from tmol.pack.rotamer import FixedAAChiSampler, construct_single_residue_kinforest
    from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
    from tmol.pack.rotamer._mainchain_fingerprint import (
        annotate_residue_type_with_sampler_fingerprints,
        find_unique_fingerprints,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--include-fallback", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    database = ParameterDatabase.get_default()
    template = default_packed_block_types(device)
    for bt in template.active_block_types:
        construct_single_residue_kinforest(bt)
    samplers = (
        create_dunbrack_sampler_from_database(database, device),
        FixedAAChiSampler(),
    )
    if args.include_fallback:
        from tmol.pack.rotamer import FallbackSampler

        samplers = (*samplers, FallbackSampler())
    old, baseline_sha = baseline_functions()
    functions = {
        "baseline": old,
        "candidate": (
            annotate_residue_type_with_sampler_fingerprints,
            find_unique_fingerprints,
        ),
    }
    candidate_path = args.candidate_source or Path(SOURCE)
    if args.candidate_source:
        text = candidate_path.read_text()
        source_sha = hashlib.sha256(text.encode()).hexdigest()
        module = importlib.import_module("tmol.pack.rotamer._mainchain_fingerprint")
        namespace = dict(vars(module))
        exec(compile(text, str(candidate_path), "exec"), namespace)
        functions["candidate"] = (
            namespace["annotate_residue_type_with_sampler_fingerprints"],
            namespace["find_unique_fingerprints"],
        )

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def setup(name, pbt):
        annotate, pack = functions[name]
        for bt in pbt.active_block_types:
            annotate(bt, samplers, database.chemical)
        pack(pbt)

    prepared = {}
    for name in functions:
        prepared[name] = fresh_pbt(template)
        setup(name, prepared[name])
    old = prepared["baseline"].mc_fingerprints
    new = prepared["candidate"].mc_fingerprints
    old_maps = old.atom_mapping.cpu().numpy()
    new_maps = new.atom_mapping.cpu().numpy()
    old_samplers = old.max_sampler.cpu().tolist()
    old_sources = old.max_fingerprint.cpu().tolist()
    new_sources = new.source_fingerprint.cpu().tolist()
    source_atoms = new.source_atom_mapping.cpu().numpy()
    checks = 0
    for source, (old_slot, old_fp, new_fp) in enumerate(
        zip(old_samplers, old_sources, new_sources)
    ):
        if old_fp < 0:
            assert new_fp < 0
            continue
        src_old = old_maps[old_slot, old_fp, source]
        src_new = source_atoms[source]
        for sampler in samplers:
            old_si = old.sampler_mapping[sampler.sampler_name()]
            new_si = new.sampler_mapping[id(sampler)]
            for target in range(template.n_types):
                dst_old = old_maps[old_si, old_fp, target]
                dst_new = new_maps[new_si, new_fp, target]
                valid_old = (src_old >= 0) & (dst_old >= 0)
                valid_new = (src_new >= 0) & (dst_new >= 0)
                numpy.testing.assert_array_equal(
                    numpy.stack((src_old[valid_old], dst_old[valid_old]), axis=1),
                    numpy.stack((src_new[valid_new], dst_new[valid_new]), axis=1),
                )
                checks += 1
    storage = {}
    for name, pbt in prepared.items():
        packed = pbt.mc_fingerprints
        storage[name] = sum(
            value.numel() * value.element_size()
            for f in attr.fields(type(packed))
            if isinstance(value := getattr(packed, f.name), torch.Tensor)
        )

    results = {}
    for mode in ("cold", "warm"):
        timings = {name: [] for name in functions}
        for round_index in range(5):
            for name in list(functions)[:: -1 if round_index % 2 else 1]:
                measured = []
                for _ in range(5):
                    pbt = fresh_pbt(template) if mode == "cold" else prepared[name]
                    sync()
                    start = time.perf_counter()
                    setup(name, pbt)
                    sync()
                    measured.append(time.perf_counter() - start)
                timings[name].append(statistics.median(measured))
        medians = {name: statistics.median(values) for name, values in timings.items()}
        results[mode] = dict(
            round_medians_seconds=timings,
            median_seconds=medians,
            baseline_over_candidate=medians["baseline"] / medians["candidate"],
        )
        print(mode, medians, flush=True)
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == source_sha
    result = dict(
        baseline=BASELINE,
        baseline_source_sha256=baseline_sha,
        candidate_source_sha256=source_sha,
        candidate_source_path=str(candidate_path),
        samplers=[s.sampler_name() for s in samplers],
        device=str(device),
        torch=torch.__version__,
        n_types=template.n_types,
        exact_transfer_maps=checks,
        packed_tensor_bytes=storage,
        results=results,
        limits="Fingerprint annotation and packing only. Chemical/kinforest/resolver construction and fresh object copies are excluded. Five alternating warm-process rounds of five samples. Every source/target transfer for the listed samplers is compared exactly where a source exists. Tensor bytes exclude Python metadata, allocator overhead and process RSS; no whole-packer speed or peak-memory claim.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
