#!/usr/bin/env python3
"""Fail when rendered API pages omit their public package exports."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_ANCHORS = {
    "io": (
        "tmol.io.pose_stack_from_biotite",
        "tmol.io.selection_gallery",
        "tmol.io.write_pose_stack_pdb",
    ),
    "ligand": (
        "tmol.ligand.inject_params_file",
        "tmol.ligand.prepare_ligand_from_smiles",
        "tmol.ligand.prepare_ligands",
    ),
    "pack": (
        "tmol.pack.PackerTask",
        "tmol.pack.pack_rotamers",
        "tmol.pack.rotamer.RotamerSet",
    ),
    "relax": (
        "tmol.relax.fast_relax",
        "tmol.relax.relax_pack_min_step",
    ),
    "score": (
        "tmol.score.ScoreFunction",
        "tmol.score.ScoreType",
        "tmol.score.beta2016_score_function",
    ),
}

NONEMPTY_API_PAGES = (
    "analysis",
    "chemical",
    "database",
    "io",
    "kinematics",
    "ligand",
    "numeric",
    "optimization",
    "pack",
    "pose",
    "relax",
    "score",
    "score_terms",
    "top_level",
    "types",
    "utility",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=Path("docs/_build/html"),
        help="Sphinx HTML output directory",
    )
    args = parser.parse_args()

    failures: list[str] = []
    api_dir = args.html_dir / "api"
    for page in NONEMPTY_API_PAGES:
        page_path = api_dir / f"{page}.html"
        if not page_path.is_file():
            failures.append(f"missing rendered API page: {page_path}")
            continue
        html = page_path.read_text(encoding="utf-8")
        if 'class="sig sig-object py"' not in html:
            failures.append(f"{page_path} contains no Python API signatures")

    for page, anchors in EXPECTED_ANCHORS.items():
        page_path = api_dir / f"{page}.html"
        if not page_path.is_file():
            continue
        html = page_path.read_text(encoding="utf-8")
        for anchor in anchors:
            if f'id="{anchor}"' not in html:
                failures.append(f"{page_path} is missing public symbol {anchor}")

    if failures:
        print("\n".join(failures))
        return 1

    print("Rendered API pages contain the expected public symbols.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
