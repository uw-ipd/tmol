from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import machine_metadata, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--engine-commit", default="")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modality")
    parser.add_argument("--cuda-execution", default="")
    parser.add_argument("--error", required=True)
    args = parser.parse_args()
    row = next(
        row
        for row in read_manifest()
        if row["dataset_id"] == args.dataset
        and (args.modality is None or row["modality"] == args.modality)
    )
    result = {
        "engine": args.engine,
        "engine_version": args.engine_version,
        "engine_commit": args.engine_commit,
        "protocol": args.protocol,
        "device": args.device,
        "batch_size": args.batch_size,
        "modality": row["modality"],
        "dataset_id": row["dataset_id"],
        "residues": int(row["residues"]),
        "polymer_residues": sum(
            int(row[key])
            for key in ("protein_residues", "dna_residues", "rna_residues")
        ),
        "atoms": int(row["atoms"]),
        "status": "failed",
        "error": args.error,
        "requested_cuda_execution": args.cuda_execution,
        "score_function": None,
        "atom_type_set": None,
        "parameter_file_sha256": None,
        "machine": machine_metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
