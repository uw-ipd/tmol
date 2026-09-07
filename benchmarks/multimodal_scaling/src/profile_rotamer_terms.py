"""Measure individual fixed-sequence rotamer score terms."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark_tmol import imports, replicated_pose, select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--modality",
        choices=("protein", "protein_ligand", "protein_nucleic"),
        default="protein",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch, _, *_, beta2016_score_function, _ = imports()
    import tmol.database
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import (
        FixedAAChiSampler,
        IncludeCurrentSampler,
        build_rotamers,
    )
    from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database

    device = torch.device(args.device)
    row = select(args.dataset, args.modality)
    pose, database = replicated_pose(row, device, args.batch_size)
    score_function = beta2016_score_function(device, param_db=database)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    defaults = tmol.database.ParameterDatabase.get_default()
    task.add_conformer_sampler(create_dunbrack_sampler_from_database(defaults, device))
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(IncludeCurrentSampler())
    pose, rotamers = build_rotamers(
        pose,
        SetPackerTask.from_packer_task(task),
        pose.packed_block_types.chem_db,
    )
    scorer = score_function.render_rotamer_scoring_module(pose, rotamers)
    coords = rotamers.coords

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    terms = []
    with torch.no_grad():
        for term in scorer.term_modules:
            for _ in range(args.warmup):
                scores, indices = term.forward(coords)
                del scores, indices
            synchronize()
            samples = []
            nnz = None
            checksum = None
            for _ in range(args.repeats):
                start = time.perf_counter()
                scores, indices = term.forward(coords)
                synchronize()
                samples.append(time.perf_counter() - start)
                nnz = int(indices.shape[1])
                checksum = float(scores.sum())
                del scores, indices
            terms.append(
                {
                    "term": term.classname,
                    "median_seconds": statistics.median(samples),
                    "samples_seconds": samples,
                    "nnz": nnz,
                    "checksum": checksum,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "label": args.label,
                "commit": args.commit,
                "dataset_id": args.dataset,
                "modality": args.modality,
                "polymer_residues": sum(
                    int(row[key])
                    for key in (
                        "protein_residues",
                        "dna_residues",
                        "rna_residues",
                    )
                ),
                "device": args.device,
                "batch_size": args.batch_size,
                "n_rotamers": int(rotamers.n_rots_for_pose.sum()),
                "terms": terms,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
