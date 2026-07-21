#!/usr/bin/env python3
"""One-time script to generate the ubq_atomworks.pt test fixture.

Run from the tmol repo root:
    TORCH_CUDA_ARCH_LIST="7.0 8.0 9.0+PTX" python3 tmol/tests/data/atomworks/generate_fixture.py

Or inside the CI container where extensions are already built.

This loads 1UBQ via tmol, converts to atomworks format, and saves tensors
that the test suite loads as ground-truth fixtures.
"""

import os
import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0 8.0 9.0+PTX")

import tmol  # noqa: E402
from tmol.io.pose_stack_from_atomworks import atomworks_from_pose_stack  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDB_DIR = os.path.join(SCRIPT_DIR, "..", "pdb")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "ubq_atomworks.pt")


def main():
    pdb_path = os.path.join(PDB_DIR, "1ubq.pdb")
    with open(pdb_path) as f:
        pdb_string = f.read()

    print("Loading 1UBQ...")
    pose_stack = tmol.pose_stack_from_pdb(pdb_string, torch.device("cpu"))
    print(f"  PoseStack: {pose_stack.n_poses} poses, {pose_stack.max_n_blocks} blocks")

    print("Converting to atomworks format...")
    aw_coords, aw_residue_type, aw_chain_iid = atomworks_from_pose_stack(pose_stack)

    ps_coords = pose_stack.coords.detach().clone()

    fixture = {
        "coords": aw_coords.cpu(),
        "residue_type": aw_residue_type.cpu(),
        "chain_iid": aw_chain_iid.cpu(),
        "pose_stack_coords": ps_coords.cpu(),
    }

    torch.save(fixture, OUTPUT_PATH)
    print(f"Saved fixture to {OUTPUT_PATH}")
    print(f"  coords:            {fixture['coords'].shape}")
    print(f"  residue_type:      {fixture['residue_type'].shape}")
    print(f"  chain_iid:         {fixture['chain_iid'].shape}")
    print(f"  pose_stack_coords: {fixture['pose_stack_coords'].shape}")


if __name__ == "__main__":
    main()
