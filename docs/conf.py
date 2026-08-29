"""Sphinx configuration for the tmol documentation."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import nbformat
from docutils import nodes
from sphinx.application import Sphinx

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


project = "TMol"
author = "Institute for Protein Design"
copyright = "2018-2026, Institute for Protein Design"
version = _project_version()
release = version

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx_togglebutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
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
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 6

nbsphinx_execute = "never"
# nbsphinx loads RequireJS on every page. 3Dmol.js is a UMD bundle, so when an
# AMD loader is present it registers as an anonymous module and never exposes
# the ``$3Dmol`` global that the tmol viewers use, leaving an empty grey box.
# ITables, the only other interactive output, loads itself as an ES module and
# does not use RequireJS, so we disable it and load 3Dmol directly with a plain
# script tag.
nbsphinx_requirejs_path = ""
# Explicit screenshots keep the gallery representative even though the live
# 3Dmol canvases cannot be used as static nbsphinx thumbnails.
nbsphinx_thumbnails = {
    "tutorial/01_working_with_tmol": "_static/tutorials/01_working_with_tmol.png",
    "tutorial/02_gpu_batching": "_static/tutorials/02_gpu_batching.png",
    "tutorial/03_scoring_and_analysis": "_static/tutorials/03_scoring_and_analysis.png",
    "tutorial/04_packing_and_mutation_scan": "_static/tutorials/04_packing_and_mutation_scan.png",
    "tutorial/05_minimization_constraints_kinematics": "_static/tutorials/05_minimization_constraints_kinematics.png",
    "tutorial/06_fast_relax": "_static/tutorials/06_fast_relax.png",
    "tutorial/07_ligand_and_params": "_static/tutorials/07_ligand_and_params.png",
    "tutorial/08_nucleic_acids": "_static/tutorials/08_nucleic_acids.png",
    "tutorial/09_protein_interface_hotspot_scan": "_static/tutorials/09_protein_interface_hotspot_scan.png",
    "tutorial/10_ligand_pose_sensitivity": "_static/tutorials/10_ligand_pose_sensitivity.png",
}
nbsphinx_epilog = r"""
----

:download:`Download this notebook <{{ env.doc2path(env.docname, base=None).name }}>`
"""

# Match the RF4 tutorial treatment: collapse setup and explicitly tagged
# presentation cells while leaving every modeling operation visible. The
# selector targets only notebook input, so tables, plots, and viewers remain
# open. ``_mark_tagged_cells`` bridges Jupyter metadata to rendered HTML.
togglebutton_selector = "#Setup .nbinput, .nbinput.collapse-code"
togglebutton_hint = "show code"
togglebutton_hint_hide = "hide code"

autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_warningiserror = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = ["openbabel"] + _compiled_module_mocks()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    # Pin the inventory to the current supported release. The ``stable`` URL
    # serves redirect stubs whose anchors cannot be validated by linkcheck.
    "torch": ("https://docs.pytorch.org/docs/2.13", None),
}
# Documentation builds must not consume an entire GPU allocation when an
# external inventory host is slow or unavailable.
intersphinx_timeout = 10

html_theme = "pydata_sphinx_theme"
html_baseurl = os.environ.get(
    "TMOL_DOCS_BASE_URL", "https://uw-ipd.github.io/tmol/latest/"
)
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "TMol"
html_theme_options = {
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_depth": -1,
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
    "globaltoc_maxdepth": 2,
    "header_links_before_dropdown": 5,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/uw-ipd/tmol",
            "icon": "fa-brands fa-github",
        },
    ],
}


def _mark_tagged_cells(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    """Copy the ``collapse-code`` notebook tag onto rendered input cells."""

    source = Path(str(app.env.doc2path(docname)))
    if source.suffix != ".ipynb":
        return
    tagged = {
        cell.source.strip()
        for cell in nbformat.read(source, as_version=4).cells
        if cell.cell_type == "code" and "collapse-code" in cell.metadata.get("tags", [])
    }
    for node in doctree.findall(nodes.container):
        if "nbinput" not in node["classes"]:
            continue
        *_, code = node.findall(nodes.literal_block)
        if code.astext().strip() in tagged:
            node["classes"].append("collapse-code")


def setup(app: Sphinx) -> None:
    app.connect("doctree-resolved", _mark_tagged_cells)
