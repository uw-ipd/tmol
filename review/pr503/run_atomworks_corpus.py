"""Isolated scoring/minimization trials for local AtomWorks structural fixtures.

Inventory includes ignored test data and records byte-identical aliases. No
input residues are cropped by this harness; tmol's default water and incomplete
backbone filtering is recorded explicitly in each trial.
Compressed/text conversion preserves CIF categories; each reader gets the same
text file. Each trial runs in a fresh process with a wall-clock timeout.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inventory(root):
    groups = {}
    for path in sorted((root / "tests").rglob("*")):
        if not path.is_file() or not path.name.endswith((".cif", ".cif.gz", ".bcif")):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in groups:
            groups[digest]["aliases"].append(str(path.relative_to(root)))
        else:
            groups[digest] = {
                "path": str(path.relative_to(root)),
                "sha256": digest,
                "bytes": path.stat().st_size,
                "aliases": [],
            }
    # Exercise the small custom chemistries before large protein controls.
    return sorted(groups.values(), key=lambda row: row["bytes"])


def normalize(path, destination):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            destination.write_text(handle.read())
        return destination
    if path.suffix == ".bcif":
        from biotite.structure.io import pdbx

        source = pdbx.BinaryCIFFile.read(path)
        target = pdbx.CIFFile()
        for name, block in source.items():
            target[name] = pdbx.CIFBlock()
            for category_name, category in block.items():
                target[name][category_name] = pdbx.CIFCategory(
                    {
                        field: pdbx.CIFColumn(
                            column.as_array(str),
                            mask=None if column.mask is None else column.mask.array,
                        )
                        for field, column in category.items()
                    }
                )
        target.write(destination)
        return destination
    return path


def trial(args):  # noqa: C901 - keep stage/error recording in one diagnostic trial
    import numpy as np
    import torch
    import biotite.structure as struc
    from tmol.io import pose_stack_from_biotite
    from tmol.io._pose_stack_from_biotite import build_context_from_biotite
    from tmol.ligand import chem_comp_types_from_cif
    from tmol.optimization import CartesianSfxnNetwork, LBFGS_Armijo
    from tmol.score import beta2016_score_function
    from profile_workloads import read_structure

    device = torch.device(args.device)
    output = Path(args.output)
    row = dict(path=args.case, reader=args.reader, device=args.device, stages={})
    start = time.perf_counter()
    stage = "normalize"

    def mark(name, function):
        nonlocal stage
        stage = name
        row.update(status="running", stage=stage)
        write(output, row)
        begin = time.perf_counter()
        result = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row["stages"][name] = time.perf_counter() - begin
        return result

    try:
        path = mark(
            "normalize", lambda: normalize(Path(args.case), output.with_suffix(".cif"))
        )
        array = mark("read", lambda: read_structure(path, args.reader))
        comp_types = mark("component_types", lambda: chem_comp_types_from_cif(path))
        starts = struc.get_residue_starts(array, add_exclusive_stop=True)
        row["input"] = dict(
            atoms=len(array),
            residues=len(starts) - 1,
            missing_atoms=int((~np.isfinite(array.coord).all(axis=1)).sum()),
            components=sorted(set(array.res_name.tolist())),
            bonds=0 if array.bonds is None else len(array.bonds.as_array()),
        )
        context = mark(
            "prepare",
            lambda: build_context_from_biotite(
                array,
                device,
                prepare_ligands=True,
                strict_ligands=True,
                ligand_seed=20260909,
                chem_comp_types=comp_types,
            ),
        )
        # Audit the constructor's documented filtering without changing input.
        co = context.canonical_ordering
        aliases = {
            alias.name3: alias.read_as
            for alias in context.parameter_database.chemical.name3_aliases
        }
        excluded = []
        for first, stop in zip(starts[:-1], starts[1:]):
            name = aliases.get(str(array.res_name[first]), str(array.res_name[first]))
            reason = None
            if name == "HOH":
                reason = "default water exclusion"
            elif name not in co.restype_io_equiv_classes:
                reason = "unrecognized component"
            else:
                required = co.restypes_required_mainchain_atoms.get(name, ())
                observed = set(
                    array.atom_name[first:stop][
                        np.isfinite(array.coord[first:stop]).all(axis=1)
                    ]
                )
                missing = sorted(set(required) - observed)
                if missing:
                    reason = "missing required mainchain atoms: " + ", ".join(missing)
            if reason:
                excluded.append(
                    dict(
                        chain=str(array.chain_id[first]),
                        residue=int(array.res_id[first]),
                        component=str(array.res_name[first]),
                        reason=reason,
                    )
                )
        row["constructor_exclusions"] = excluded
        pose = mark(
            "construct",
            lambda: pose_stack_from_biotite(
                array, device, context=context, no_optH=True
            ),
        )
        row["pose"] = dict(
            atoms=int(pose.real_atoms.sum()),
            blocks=int((pose.block_type_ind >= 0).sum()),
        )
        row["pose"]["expected_blocks_after_filtering"] = len(starts) - 1 - len(excluded)
        assert torch.isfinite(
            pose.coords[pose.real_atoms]
        ).all(), "non-finite constructed coordinates"
        score = beta2016_score_function(device, param_db=context.parameter_database)
        network = mark("score_setup", lambda: CartesianSfxnNetwork(score, pose))

        def evaluate():
            network.zero_grad()
            energy = network()
            assert torch.isfinite(energy).all(), "non-finite energy"
            energy.sum().backward()
            gradient = network.masked_coords.grad
            assert torch.isfinite(gradient).all(), "non-finite gradient"
            return dict(
                energy=float(energy.sum().detach()),
                max_abs_gradient=float(gradient.abs().max()),
            )

        row["initial"] = mark("score_gradient", evaluate)
        optimizer = LBFGS_Armijo(
            network.parameters(),
            max_iter=args.max_iter,
            segment_ids=network.segment_ids,
        )

        def minimize():
            def closure():
                optimizer.zero_grad()
                energy = network()
                energy.sum().backward()
                return energy

            optimizer.step(closure)

        mark("minimize", minimize)
        row["final"] = mark("final_score_gradient", evaluate)
        state = optimizer.state[next(iter(network.parameters()))]
        row["optimizer"] = {
            key: (
                value.detach().cpu().tolist()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in state.items()
            if key in ("n_iter", "func_evals", "converged", "stalled")
        }
        row["optimizer"]["max_iter"] = args.max_iter
        minimized = network.pose_stack_from_dofs()
        assert torch.equal(minimized.block_type_ind, pose.block_type_ind)
        assert torch.equal(
            minimized.inter_residue_connections, pose.inter_residue_connections
        )
        connections = []
        block_types = pose.packed_block_types.active_block_types
        indices = pose.block_type_ind[0].tolist()
        offsets = pose.block_coord_offset[0].tolist()
        links = pose.inter_residue_connections[0].tolist()
        for block, type_index in enumerate(indices):
            if type_index < 0:
                continue
            first_type = block_types[type_index]
            for connection, (partner, partner_connection) in enumerate(links[block]):
                if partner <= block:
                    continue
                second_type = block_types[indices[partner]]
                first_atom = int(first_type.ordered_connection_atoms[connection])
                second_atom = int(
                    second_type.ordered_connection_atoms[partner_connection]
                )
                first = offsets[block] + first_atom
                second = offsets[partner] + second_atom
                connections.append(
                    dict(
                        blocks=[block, partner],
                        types=[first_type.name, second_type.name],
                        atoms=[
                            first_type.atoms[first_atom].name,
                            second_type.atoms[second_atom].name,
                        ],
                        initial_distance=float(
                            torch.linalg.vector_norm(
                                pose.coords[0, first] - pose.coords[0, second]
                            )
                        ),
                        final_distance=float(
                            torch.linalg.vector_norm(
                                minimized.coords[0, first] - minimized.coords[0, second]
                            )
                        ),
                    )
                )
        row["connection_distances"] = connections
        row["connections_lengthened_over_half_angstrom"] = sum(
            link["final_distance"] - link["initial_distance"] > 0.5
            for link in connections
        )
        row["max_displacement"] = float(
            torch.linalg.vector_norm(
                minimized.coords[pose.real_atoms] - pose.coords[pose.real_atoms], dim=-1
            ).max()
        )
        tolerance = 1e-5 * max(1, abs(row["initial"]["energy"]))
        assert (
            row["final"]["energy"] <= row["initial"]["energy"] + tolerance
        ), "energy increased"
        row.update(status="passed", stage="complete")
        if any(item["reason"] != "default water exclusion" for item in excluded):
            row["status"] = "partial"
        if row["pose"]["blocks"] != row["pose"]["expected_blocks_after_filtering"]:
            row["status"] = "partial"
    except Exception:
        row.update(status="failed", stage=stage, error=traceback.format_exc())
    row["seconds"] = time.perf_counter() - start
    row["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if device.type == "cuda":
        row["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    write(output, row)
    return row["status"] == "passed"


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = inventory(Path(args.root))
    write(output / "inventory.json", cases)
    versions = {}
    for package in ("torch", "numpy", "biotite", "rdkit", "atomworks"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    atomworks_spec = importlib.util.find_spec("atomworks")
    atomworks_path = None if atomworks_spec is None else atomworks_spec.origin
    atomworks_commit = None
    if atomworks_path:
        result = subprocess.run(
            ["git", "-C", str(Path(atomworks_path).parent), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            atomworks_commit = result.stdout.strip()
    write(
        output / "provenance.json",
        {
            "arguments": vars(args),
            "versions": versions,
            "atomworks_source": atomworks_path,
            "atomworks_commit": atomworks_commit,
            "python": sys.version,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tmol_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "fixture_commit": subprocess.check_output(
                ["git", "-C", args.root, "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    rows = []
    for index, case in enumerate(cases):
        for reader in ("tmol", "atomworks"):
            result = output / f"{index:02d}-{reader}.json"
            command = [
                sys.executable,
                __file__,
                "--case",
                str(Path(args.root) / case["path"]),
                "--reader",
                reader,
                "--device",
                args.device,
                "--output",
                str(result),
                "--max-iter",
                str(args.max_iter),
            ]
            with result.with_suffix(".log").open("w") as log:
                try:
                    completed = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                    )
                    exit_code = completed.returncode
                except subprocess.TimeoutExpired:
                    exit_code = "timeout"
            row = (
                json.loads(result.read_text())
                if result.exists()
                else {"status": "failed", "stage": "startup"}
            )
            if exit_code != 0:
                row["exit_code"] = exit_code
                if row.get("status") not in ("failed", "partial"):
                    row["status"] = "failed"
            row.update(source=case, reader=reader, result=str(result))
            rows.append(row)
            write(output / "measurements.json", rows)
            print(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "path",
                            "reader",
                            "status",
                            "stage",
                            "seconds",
                            "exit_code",
                        )
                    }
                ),
                flush=True,
            )
    return all(row["status"] == "passed" for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/mnt/home/kdidi/projects/atomworks-dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--case")
    parser.add_argument("--reader", choices=("tmol", "atomworks"))
    args = parser.parse_args()
    raise SystemExit(0 if (trial(args) if args.case else main(args)) else 1)
