#!/usr/bin/env python3
"""Verify that rendered documentation exposes the task-first navigation."""

from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_LINKS = {
    "index.html": (
        'href="learning_paths.html"',
        'href="workflows/index.html"',
        'href="tutorial/recipe_index.html"',
        'href="tutorial/rosetta_crosswalk.html"',
    ),
    "learning_paths.html": (
        'href="tutorial/01_working_with_tmol.html"',
        'href="tutorial/02_gpu_batching.html"',
        'href="tutorial/03_scoring_and_analysis.html"',
        'href="tutorial/04_packing_and_mutation_scan.html"',
        'href="tutorial/05_minimization_constraints_kinematics.html"',
        'href="tutorial/06_fast_relax.html"',
        'href="tutorial/07_ligand_and_params.html"',
        'href="tutorial/08_nucleic_acids.html"',
    ),
    "workflows/index.html": (
        'href="structure_io.html"',
        'href="../user_guide/scoring.html"',
        'href="packing.html"',
        'href="../user_guide/optimization.html"',
        'href="../user_guide/ligands.html"',
        'href="nucleic_acids.html"',
    ),
    "workflows/structure_io.html": (
        'href="../tutorial/01_working_with_tmol.html"',
        'href="../api/io.html"',
        'href="../api/pose.html"',
    ),
    "examples_index.html": (
        'href="tutorial/01_working_with_tmol.html"',
        'href="tutorial/08_nucleic_acids.html"',
        'href="tutorial/recipe_index.html"',
    ),
}


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
    for relative_path, expected_links in EXPECTED_LINKS.items():
        page_path = args.html_dir / relative_path
        if not page_path.is_file():
            failures.append(f"missing rendered navigation page: {page_path}")
            continue

        html = page_path.read_text(encoding="utf-8")
        for link in expected_links:
            if link not in html:
                failures.append(f"{page_path} is missing {link}")

    if failures:
        print("\n".join(failures))
        return 1

    print("Rendered documentation contains the expected workflow navigation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
