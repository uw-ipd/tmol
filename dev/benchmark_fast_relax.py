#!/usr/bin/env python3
"""Time end-to-end Cartesian FastRelax on a chosen benchmark system."""

from __future__ import annotations

import argparse
import json
import time

import torch

from benchmark_score_matrix import SYSTEMS, load_system, synchronize
from tmol.kinematics import CartesianMoveMap, FoldForest
from tmol.pack import PackerPalette
from tmol.pose import PoseStackBuilder
from tmol.relax import fast_relax
from tmol.score import beta2016_score_function


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="protein-76")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260902)
    graph_group = parser.add_mutually_exclusive_group()
    graph_group.add_argument(
        "--cuda-graph",
        dest="cuda_graph",
        action="store_true",
        default=None,
        help="force CUDA graph replay (default: automatic by chemistry)",
    )
    graph_group.add_argument(
        "--no-cuda-graph",
        dest="cuda_graph",
        action="store_false",
        help="force eager scoring",
    )
    parser.add_argument("--no-opt-h", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch, args.trials, args.repeats, args.cpu_threads) < 1:
        raise SystemExit("batch, trials, repeats, and cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    base, database = load_system(SYSTEMS[args.system], device, not args.no_opt_h)
    initial = PoseStackBuilder.from_poses([base] * args.batch, device)
    fold_forest = FoldForest.reasonable_fold_forest(initial)
    move_map = CartesianMoveMap()
    sfxn = beta2016_score_function(device, param_db=database)
    initial_scorer = sfxn.render_whole_pose_scoring_module(initial)
    initial_score = initial_scorer(initial.coords).detach()

    if args.cuda_graph is True and device.type != "cuda":
        raise SystemExit("--cuda-graph requires --device cuda")

    for trial in range(args.trials):
        torch.manual_seed(args.seed + trial)
        pose = initial.clone()
        synchronize(device)
        start = time.perf_counter()
        relaxed = fast_relax(
            pose,
            sfxn,
            PackerPalette(),
            move_map,
            fold_forest,
            num_repeats=args.repeats,
            cuda_graph=args.cuda_graph,
            verbose=args.verbose,
        )
        synchronize(device)
        seconds = time.perf_counter() - start
        scorer = sfxn.render_whole_pose_scoring_module(relaxed)
        final_score = scorer(relaxed.coords).detach()
        print(
            json.dumps(
                {
                    "batch": args.batch,
                    "device": str(device),
                    "final_score": final_score.cpu().tolist(),
                    "initial_score": initial_score.cpu().tolist(),
                    "repeats": args.repeats,
                    "seconds": seconds,
                    "system": args.system,
                    "trial": trial,
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
