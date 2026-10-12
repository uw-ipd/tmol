#!/usr/bin/env python3
"""Score and differentiate with an installed tmol-jax distribution."""

import argparse
from importlib import metadata, util
from pathlib import Path

import jax
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--expect-no-torch", action="store_true")
    parser.add_argument("--require-native-cuda", action="store_true")
    parser.add_argument("--also-torch", action="store_true")
    args = parser.parse_args()

    if args.expect_no_torch:
        assert util.find_spec("torch") is None, "Torch leaked into JAX-only install"
    assert metadata.version("tmol-jax") == args.version
    expected_platform = "cpu" if args.device == "cpu" else "gpu"
    assert jax.default_backend() == expected_platform

    from tmol_jax import beta2016_score_function, pose_stack_from_pdb

    if args.require_native_cuda:
        from tmol_jax.score.elec import cuda as elec_cuda
        from tmol_jax.score.hbond import cuda as hbond_cuda
        from tmol_jax.score.ljlk import cuda as ljlk_cuda
        from tmol_jax.score.lk_ball import cuda as lkball_cuda

        assert ljlk_cuda.available()
        assert ljlk_cuda.fused_available()
        assert elec_cuda.available()
        assert hbond_cuda.available()
        assert lkball_cuda.available()

    pose = pose_stack_from_pdb(args.pdb)
    score_function = beta2016_score_function()
    energy, gradient = score_function.jitted_value_and_grad(pose)(pose.coords)
    energy.block_until_ready()
    gradient.block_until_ready()
    assert np.isfinite(energy).all()
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(np.asarray(gradient)) > 0

    if args.also_torch:
        import torch
        from tmol.io import pose_stack_from_pdb as torch_pose_from_pdb
        from tmol.score import beta2016_score_function as torch_score_function

        torch_pose = torch_pose_from_pdb(str(args.pdb), torch.device("cpu"))
        torch_coords = torch_pose.coords.detach().clone().requires_grad_(True)
        torch_scorer = torch_score_function(
            torch.device("cpu")
        ).render_whole_pose_scoring_module(
            torch_pose,
        )
        torch_energy = torch_scorer(torch_coords)
        torch_gradient = torch.autograd.grad(torch_energy.sum(), torch_coords)[0]
        assert torch.isfinite(torch_energy).all()
        assert torch.isfinite(torch_gradient).all()
        np.testing.assert_allclose(
            np.asarray(energy), torch_energy.detach().cpu().numpy(), atol=5e-3
        )

    print(
        f"tmol-jax {args.version} {args.device}: "
        f"energy={float(np.asarray(energy).sum()):.6f} "
        f"gradient_norm={float(np.linalg.norm(np.asarray(gradient))):.6f}"
    )


if __name__ == "__main__":
    main()
