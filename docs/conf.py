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
    "autoapi.extension",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.aafig",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
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
    "api/modules.rst",
    "sg_execution_times.rst",
    "auto_examples/*.codeobj.json",
    "auto_examples/*.ipynb",
    "auto_examples/*.py",
    "auto_examples/*.zip",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 4

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

autoapi_type = "python"
autoapi_dirs = [str(REPO_ROOT / "tmol")]
autoapi_root = "api"
autoapi_template_dir = "_templates/autoapi"
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_ignore = [
    "*/tmol/tests/*",
    "*/tmol/extern/*",
    "*/compiled/*",
    "*/compiled.py",
    "*/tmol/io/details/*",
    "*/tmol/pack/compiled/*",
    "*/tmol/relax/*",
    "*/tmol/score/constraint/*",
    "*/_C*",
]
suppress_warnings = [
    "autoapi.python_import_resolution",
    "ref.doc",
]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "astor",
    "attrs",
    "attrs_strict",
    "biotite",
    "cattrs",
    "frozendict",
    "hypothesis",
    "llvmlite",
    "networkx",
    "numba",
    "numpy",
    "openbabel",
    "pandas",
    "pint",
    "psutil",
    "py3Dmol",
    "pyarrow",
    "rdkit",
    "scipy",
    "sparse",
    "toolz",
    "torch",
    "typing_inspect",
    "yaml",
] + _compiled_module_mocks()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

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
    "plot_gallery": True,
    "abort_on_example_error": True,
    "capture_repr": ("_repr_html_", "__repr__"),
    "download_all_examples": False,
    "thumbnail_size": (350, 350),
    "ignore_pattern": r"GALLERY_HEADER\.rst",
}
