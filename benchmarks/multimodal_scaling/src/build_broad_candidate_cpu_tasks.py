"""Build the frozen all-input task tables for a candidate CPU sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import read_manifest


FIELDS = ("protocol", "modality", "dataset_id", "batch_size", "cuda_execution")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datasets = [row for row in read_manifest() if row["status"] == "ok"]
    args.output.mkdir(parents=True, exist_ok=True)
    for protocol in ("score_gradient", "fastrelax"):
        rows = []
        for dataset in datasets:
            if protocol == "fastrelax" and dataset.get(
                "fastrelax", "yes"
            ).lower() not in {"yes", "true", "1"}:
                continue
            rows.append(
                {
                    "protocol": protocol,
                    "modality": dataset["modality"],
                    "dataset_id": dataset["dataset_id"],
                    "batch_size": 1,
                    "cuda_execution": "auto" if protocol == "fastrelax" else "eager",
                }
            )
        path = args.output / f"tasks-broad-candidate-{protocol}-cpu.tsv"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
        print(path, len(rows))


if __name__ == "__main__":
    main()
