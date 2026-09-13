"""Compare paired scoring outputs, timings and allocator peaks across checkouts."""

import argparse
import json
from pathlib import Path
import statistics

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--before-prefix", type=Path, required=True)
parser.add_argument("--after-prefix", type=Path, required=True)
parser.add_argument("--trials", type=int, default=3)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
labels = ("term", "poses", "residues_per_pose", "block_pairs", "gradient")
profiles = {}
outputs = {}
for side in ("before", "after"):
    prefix = getattr(args, f"{side}_prefix")
    profiles[side] = [
        json.loads(Path(f"{prefix}-{i}.json").read_text()) for i in range(args.trials)
    ]
    outputs[side] = [
        torch.load(Path(f"{prefix}-{i}.pt"), weights_only=True)
        for i in range(args.trials)
    ]
records = []
for index, case in enumerate(profiles["before"][0]["results"]):
    differences = []
    exact = True
    repeat_differences = {side: 0.0 for side in outputs}
    for trial in range(args.trials):
        a, b = [profiles[side][trial] for side in ("before", "after")]
        for field in ("input_pdb_sha256", "device", "torch", "gpu"):
            assert a[field] == b[field]
        for side in profiles:
            assert {k: profiles[side][trial]["results"][index][k] for k in labels} == {
                k: case[k] for k in labels
            }
        before, after = outputs["before"][trial][index], outputs["after"][trial][index]
        assert len(before) == len(after)
        trial_diffs = []
        for x, y in zip(before, after):
            torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-4)
            trial_diffs.append(float((x - y).abs().max()))
            exact = exact and torch.equal(x, y)
        differences.append(trial_diffs)
        for side in outputs:
            for x, y in zip(outputs[side][0][index], outputs[side][trial][index]):
                repeat_differences[side] = max(
                    repeat_differences[side], float((x - y).abs().max())
                )
    times = {
        side: [p["results"][index]["median_milliseconds"] for p in profiles[side]]
        for side in profiles
    }
    medians = {side: statistics.median(v) for side, v in times.items()}
    peaks = {
        side: [
            p["results"][index]["peak_allocated_bytes_above_inputs"]
            for p in profiles[side]
        ]
        for side in profiles
    }
    records.append(
        {
            **{k: case[k] for k in labels},
            "trial_median_milliseconds": times,
            "median_milliseconds": medians,
            "speedup": medians["before"] / medians["after"],
            "peak_allocated_bytes_above_inputs": peaks,
            "max_absolute_differences_per_trial": differences,
            "all_tensors_exact": exact,
            "max_absolute_difference_across_repeated_processes": repeat_differences,
        }
    )
result = {
    "profiles": profiles,
    "comparisons": records,
    "rtol": 2e-5,
    "atol": 2e-4,
    "limits": "Median of three alternating-checkout process trials by default, each holding seven warm timing rounds. Numerical comparisons include repeated-process controls for reduction/atomic accumulation differences. Ratios apply to these individual terms and pose shapes, not a full score function, packing or preparation.",
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "cases": len(records),
            "all_tensors_exact": all(r["all_tensors_exact"] for r in records),
        },
        indent=2,
    )
)
