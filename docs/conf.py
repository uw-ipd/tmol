"""Sphinx configuration for the tmol documentation."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# Keep autodoc lightweight and deterministic. The docs should not require a
# locally built C++/CUDA extension package just to import module docstrings.
os.environ.setdefault("TMOL_DOCS_BUILD", "1")

try:
    import pypandoc

    pandoc_dir = str(Path(pypandoc.get_pandoc_path()).parent)
    os.environ["PATH"] = f"{pandoc_dir}{os.pathsep}{os.environ['PATH']}"
except ImportError:
    pass


def _project_version() -> str:
    pyproject = REPO_ROOT / "pyproject.toml"
    with pyproject.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _module_name_from_path(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _compiled_module_mocks() -> list[str]:
    mocks: set[str] = set()
    for path in (REPO_ROOT / "tmol").rglob("*.py"):
        rel_parts = path.relative_to(REPO_ROOT).parts
        if "compiled" in rel_parts or path.stem.startswith("compiled"):
            mocks.add(_module_name_from_path(path))
            if path.name == "__init__.py":
                mocks.add(".".join(rel_parts[:-1]))
    return sorted(mocks)


project = "tmol"
author = "Institute for Protein Design"
copyright = "2018-2026, Institute for Protein Design"
version = _project_version()
release = version
docs_base_url = os.environ.get(
    "TMOL_DOCS_BASE_URL", "https://uw-ipd.github.io/tmol"
).rstrip("/")
switcher_version = os.environ.get("TMOL_DOCS_VERSION_MATCH", version)
switcher_json_url = os.environ.get(
    "TMOL_DOCS_SWITCHER_JSON_URL",
    f"{docs_base_url}/latest/_static/switcher.json",
)

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx_togglebutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "examples/GALLERY_HEADER.rst",
    "sg_execution_times.rst",
    "auto_examples",
    "auto_examples/*.codeobj.json",
    "auto_examples/*.ipynb",
    "auto_examples/*.py",
    "auto_examples/*.zip",
    "auto_examples/**/*.codeobj.json",
    "auto_examples/**/*.ipynb",
    "auto_examples/**/*.py",
    "auto_examples/**/*.zip",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 4

nbsphinx_execute = "never"
# nbsphinx loads RequireJS on every page. 3Dmol.js is a UMD bundle, so when an
# AMD loader is present it registers as an anonymous module and never exposes
# the ``$3Dmol`` global that the tmol viewers use, leaving an empty grey box.
# ITables, the only other interactive output, loads itself as an ES module and
# does not use RequireJS, so we disable it and load 3Dmol directly with a plain
# script tag.
nbsphinx_requirejs_path = ""
nbsphinx_thumbnails = {
    "tutorial/01_working_with_tmol": "_static/tutorials/01_working_with_tmol.svg",
    "tutorial/02_gpu_batching": "_static/tutorials/02_gpu_batching.svg",
    "tutorial/03_scoring_and_analysis": "_static/tutorials/03_scoring_and_analysis.svg",
    "tutorial/04_packing_and_mutation_scan": "_static/tutorials/04_packing_and_mutation_scan.svg",
    "tutorial/05_minimization_constraints_kinematics": "_static/tutorials/05_minimization_constraints_kinematics.svg",
    "tutorial/06_fast_relax": "_static/tutorials/06_fast_relax.svg",
    "tutorial/07_ligand_and_params": "_static/tutorials/07_ligand_and_params.svg",
    "tutorial/08_nucleic_acids": "_static/tutorials/08_nucleic_acids.svg",
}
nbsphinx_epilog = r"""
----

:download:`Download this notebook <{{ env.doc2path(env.docname, base=None) }}>`
"""

# Keep every executable input available without forcing readers to scroll
# through implementation details. Outputs remain visible and interactive.
togglebutton_selector = ".nbinput"
togglebutton_hint = "Show code"
togglebutton_hint_hide = "Hide code"

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_warningiserror = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
suppress_warnings = ["docutils", "ref.doc", "ref.python"]

autodoc_mock_imports = ["openbabel"] + _compiled_module_mocks()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
# Documentation builds must not consume an entire GPU allocation when an
# external inventory host is slow or unavailable.
intersphinx_timeout = 10

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path: list[str] = []
html_title = "tmol documentation"
html_theme_options = {
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_depth": -1,
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
    "globaltoc_maxdepth": -1,
    "header_links_before_dropdown": 8,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": switcher_json_url,
        "version_match": switcher_version,
    },
    "check_switcher": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/uw-ipd/tmol",
            "icon": "fa-brands fa-github",
        },
    ],
}

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "auto_examples",
    "image_scrapers": ("matplotlib",),
    "plot_gallery": False,
    "capture_repr": ("_repr_html_", "__repr__"),
    "download_all_examples": False,
    "filename_pattern": r".*",
    "thumbnail_size": (350, 350),
    "ignore_pattern": r"GALLERY_HEADER\.rst",
}
