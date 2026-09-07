"""Build deterministic, independently resumable benchmark task tables."""

from __future__ import annotations

import csv

from common import ROOT, read_manifest

COMMITS = {
    "0.1.46": "4ac54af52f9dc02cfc6a439f151eea63561e6cf0",
    "0.1.47": "2e73c9fb030e64e175161e9a385bb3229262a99b",
    "0.1.55": "39f757f8837f853b85d6ce367938934e832c1033",
}
FIELDS = (
    "engine",
    "engine_version",
    "engine_commit",
    "protocol",
    "device",
    "batch_size",
    "modality",
    "dataset_id",
    "cuda_execution",
)


def add_tmol(rows, dataset, protocol, version, device, batch, execution):
    rows.append(
        {
            "engine": "tmol",
            "engine_version": version,
            "engine_commit": COMMITS[version],
            "protocol": protocol,
            "device": device,
            "batch_size": batch,
            "modality": dataset["modality"],
            "dataset_id": dataset["dataset_id"],
            "cuda_execution": execution,
        }
    )


def main() -> None:
    datasets = [row for row in read_manifest() if row["status"] == "ok"]
    tables = {
        (protocol, device): []
        for protocol in ("score_gradient", "fastrelax")
        for device in ("cpu", "cuda")
    }
    for dataset in datasets:
        historical = dataset["historical_version"]
        for protocol in ("score_gradient", "fastrelax"):
            if protocol == "fastrelax" and dataset.get(
                "fastrelax", "yes"
            ).lower() not in {
                "yes",
                "true",
                "1",
            }:
                continue
            tables[(protocol, "cpu")].append(
                {
                    "engine": "pyrosetta",
                    "engine_version": "2024.39",
                    "engine_commit": "59628fbc5bc09f1221e1642f1f8d157ce49b1410",
                    "protocol": protocol,
                    "device": "cpu",
                    "batch_size": 1,
                    "modality": dataset["modality"],
                    "dataset_id": dataset["dataset_id"],
                    "cuda_execution": "",
                }
            )
            for version in ("0.1.55", historical):
                add_tmol(
                    tables[(protocol, "cpu")],
                    dataset,
                    protocol,
                    version,
                    "cpu",
                    1,
                    "auto" if protocol == "fastrelax" else "eager",
                )
                for batch in (1, 10, 100, 1000):
                    if protocol == "score_gradient" and version == "0.1.55":
                        executions = ("eager", "graph")
                    elif protocol == "fastrelax":
                        executions = ("auto",)
                    else:
                        executions = ("eager",)
                    for execution in executions:
                        add_tmol(
                            tables[(protocol, "cuda")],
                            dataset,
                            protocol,
                            version,
                            "cuda",
                            batch,
                            execution,
                        )
    for (protocol, device), rows in tables.items():
        path = ROOT / f"metadata/tasks-{protocol}-{device}.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
        print(path.name, len(rows))


if __name__ == "__main__":
    main()
