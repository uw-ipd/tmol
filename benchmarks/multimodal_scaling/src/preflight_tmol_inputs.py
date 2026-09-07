from __future__ import annotations

import argparse
import json
import sys
import traceback

from benchmark_tmol import imports, load_pose
from common import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmol-version", required=True)
    args = parser.parse_args()
    torch, *_ = imports()
    failed = False
    for row in read_manifest():
        if row["status"] != "ok" or row["modality"] != "protein_nucleic":
            continue
        result = {
            "dataset_id": row["dataset_id"],
            "expected_residues": int(row["residues"]),
            "status": "ok",
            "tmol_version": args.tmol_version,
        }
        try:
            pose, _ = load_pose(row, torch.device("cpu"))
            loaded = int((pose.block_type_ind64[0] >= 0).sum())
            result["loaded_residues"] = loaded
            if loaded != result["expected_residues"]:
                raise ValueError(
                    f"loaded {loaded} blocks; expected {result['expected_residues']}"
                )
        except Exception as error:
            failed = True
            result.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
        print(json.dumps(result, sort_keys=True), flush=True)
    sys.exit(failed)


if __name__ == "__main__":
    main()
