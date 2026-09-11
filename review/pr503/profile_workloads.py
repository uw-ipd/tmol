"""Stage timings, memory, Python profiles, and correctness for real fixtures.

Run from the checkout being measured with PYTHONPATH=.; the script may live in
another checkout. CUDA timings synchronize the selected device. Compilation
and first execution are recorded separately from steady-state measurements.
"""

import argparse
import cProfile
import importlib.metadata
import json
from pathlib import Path
import resource
import statistics
import subprocess
import time
import traceback

import torch
import tmol

from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.io._pose_stack_from_biotite import build_context_from_biotite
from tmol.pack import PackerPalette, PackerTask, SetPackerTask, pack_rotamers
from tmol.pack.rotamer import FixedAAChiSampler, NaChiRotamerSampler, build_rotamers
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.pack.rotamer._conjugated_groups import add_conjugated_group_sampler
from tmol.pose import PoseStackBuilder
from tmol.score import beta2016_score_function


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device, repeats=1):
    times = []
    result = None
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return result, dict(median=statistics.median(times), samples=times)


def task_for(pose, database, device):
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(create_dunbrack_sampler_from_database(database, device))
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(NaChiRotamerSampler.from_database(database, device))
    add_conjugated_group_sampler(task, pose)
    return task


def read_structure(path, reader):
    if reader == "tmol":
        return atom_array_from_cif(path)
    from atomworks.io.parser import parse
    from atomworks.io.config import ParseConfig

    config = ParseConfig(
        model=1,
        build_assembly=None,
        remove_ccds=(),
        remove_waters=False,
        fix_arginines=False,
        fix_ligands_at_symmetry_centers=False,
        long_bond_policy="keep",
        struct_conn_distance_policy="keep",
        add_bond_types_from_struct_conn=("covale", "disulf"),
        hydrogen_policy="remove",
        ccd_mirror_path=None,
        add_id_and_entity_annotations=False,
    )
    array = parse(path, config=config)["asym_unit"]
    return array[0] if array.coord.ndim == 3 else array


def measure(path, device, args, output):
    data = Path(tmol.__file__).parent / "tests/data"
    row = dict(fixture=str(path.relative_to(data)), device=str(device), stages={})
    stages = row["stages"]
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        array, stages["read"] = timed(lambda: read_structure(path, args.reader), device)
        profiler = cProfile.Profile()
        if args.python_profile:
            profiler.enable()
        context, stages["prepare"] = timed(
            lambda: build_context_from_biotite(
                array, device, prepare_ligands=True, ligand_seed=20260909
            ),
            device,
        )
        pose, stages["construct"] = timed(
            lambda: pose_stack_from_biotite(
                array, device, context=context, no_optH=True
            ),
            device,
        )
        profiler.disable()
        if args.python_profile:
            profiler.dump_stats(str(output / f"{path.stem}-{device}-prepare.pstats"))
        db = context.parameter_database
        if args.batch != 1:
            pose = PoseStackBuilder.from_poses([pose] * args.batch, device)
        row.update(
            batch=pose.n_poses, blocks=pose.max_n_blocks, atoms=pose.coords.shape[1]
        )
        score = beta2016_score_function(device, param_db=db)
        module, stages["score_setup"] = timed(
            lambda: score.render_whole_pose_scoring_module(pose), device
        )
        coords = pose.coords.detach().clone().requires_grad_(True)

        def forward():
            with torch.no_grad():
                return module(coords)

        def backward():
            coords.grad = None
            energy = module(coords)
            energy.sum().backward()
            return energy.detach()

        _, stages["first_forward_backward"] = timed(backward, device)
        energy, stages["score"] = timed(forward, device, args.repeats)
        _, stages["score_backward"] = timed(backward, device, args.repeats)
        assert torch.isfinite(energy).all(), "non-finite energy"
        assert torch.isfinite(coords.grad).all(), "non-finite gradient"
        row.update(
            score=energy.cpu().tolist(), max_abs_gradient=float(coords.grad.abs().max())
        )
        if args.rotamers or args.pack:
            task = task_for(pose, db, device)
            profiler = cProfile.Profile()
            if args.python_profile:
                profiler.enable()
            (rot_pose, rotamers), stages["rotamers"] = timed(
                lambda: build_rotamers(
                    pose, SetPackerTask.from_packer_task(task), db.chemical
                ),
                device,
            )
            profiler.disable()
            if args.python_profile:
                profiler.dump_stats(
                    str(output / f"{path.stem}-{device}-rotamers.pstats")
                )
            row["rotamers"] = int(rotamers.n_rots_for_pose.sum())
            assert torch.isfinite(
                rotamers.coords
            ).all(), "non-finite rotamer coordinates"
        if args.pack:
            packed, stages["pack"] = timed(
                lambda: pack_rotamers(pose, score, task_for(pose, db, device)), device
            )
            packed_module = score.render_whole_pose_scoring_module(packed)
            assert torch.isfinite(packed.coords).all(), "non-finite packed coordinates"
            assert torch.isfinite(
                packed_module(packed.coords)
            ).all(), "non-finite packed score"
        row["status"] = "passed"
    except Exception:
        row.update(status="failed", error=traceback.format_exc())
    row["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if device.type == "cuda":
        row["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", action="append", help="Fixture stem, repeatable")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reader", choices=("tmol", "atomworks"), default="tmol")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--rotamers", action="store_true")
    parser.add_argument("--pack", action="store_true")
    parser.add_argument(
        "--python-profile",
        action="store_true",
        help="Instrument preparation/rotamers; use a separate uninstrumented run for timing comparisons",
    )
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(tmol.__file__).parent.parent
    data = root / "tmol/tests/data"
    paths = [data / "cif/1UBQ.cif"]
    paths += sorted((data / "ncaa_fixtures").glob("*.cif"))
    paths += sorted((data / "covalent_fixtures").glob("*.cif"))
    paths += [data / "cif/cyclic_peptide_1jbl.cif"]
    if args.fixture:
        paths = [p for p in paths if p.stem in args.fixture]
        missing = set(args.fixture) - {p.stem for p in paths}
        if missing:
            parser.error(f"unknown fixtures: {sorted(missing)}")
    results = dict(
        checkout=str(root),
        reader=args.reader,
        diff=subprocess.check_output(
            ["git", "-C", str(root), "diff", "HEAD"], text=True
        ),
        commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        packages={
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "biotite", "rdkit")
        },
        device=str(device),
        instrumented=args.python_profile,
        threads=torch.get_num_threads(),
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        runs=[],
    )
    for path in paths:
        row = measure(path, device, args, args.output)
        results["runs"].append(row)
        (args.output / "measurements.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(json.dumps(row), flush=True)
    return 0 if all(r["status"] == "passed" for r in results["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
