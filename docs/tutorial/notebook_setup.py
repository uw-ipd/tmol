"""Shared setup for the interactive TMol tutorial notebooks.

Every tutorial opens with the same cell::

    from notebook_setup import *  # noqa: F401,F403

    device, param_db = setup()

Importing this module on Google Colab installs the released TMol wheel (matched
to the runtime's PyTorch) and downloads the example data. Everywhere else it is a
cheap import: the names the tutorials use are resolved lazily, so a tutorial
never needs its own import block for the common TMol API.

``setup()`` picks the device, seeds PyTorch and returns the default parameter
database. ``data_path()`` locates the checked-in example structures, both in a
repository checkout and in Colab.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from urllib.request import urlretrieve

TUTORIAL_REF = "master"
RAW_BASE = f"https://raw.githubusercontent.com/uw-ipd/tmol/{TUTORIAL_REF}"
# Advance only after the release wheels have been published and exercised.
# The source project's next version may not have downloadable wheels yet.
TMOL_RELEASE = "0.1.64"

SEED = 0

# Example structures the tutorials read through ``data_path``, relative to
# ``tmol/tests/data``. Colab downloads exactly these; a test keeps the list in
# step with the notebooks. (Tutorials with their own files pass repository paths
# to ``setup_colab``.)
FIXTURES = (
    "metal_fixtures/zn_tetrahedral_3ks3.cif.zst",
    "metal_fixtures/fes_ferredoxin_6e6r.cif.zst",
    "ncaa_fixtures/uic1_aib_9bzt.cif",
    "ncaa_fixtures/phosphopeptide_5ema.cif",
    "protein_ligand_test/ace.tmol.nomin.cif",
    "protein_ligand_test/ace.xtal-lig.mmff94.tmol",
)
_REPO_DATA_DIR = "tmol/tests/data"
_colab_ready = False


def _pip_install(requirements: list[str], torch_version: str) -> None:
    """Install packages without allowing pip to replace the active PyTorch."""
    with tempfile.TemporaryDirectory(prefix="tmol-colab-") as temp_dir:
        constraint = Path(temp_dir) / "constraints.txt"
        constraint.write_text(f"torch=={torch_version}\n", encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--only-binary=:all:",
                "--constraint",
                str(constraint),
                *requirements,
            ]
        )


def _download(relative_path: str, destination: Path) -> None:
    """Fetch one repository file unless a non-empty copy is already present."""
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RAW_BASE}/{relative_path}"
    partial = destination.with_name(f"{destination.name}.part")
    try:
        urlretrieve(url, partial)
        if partial.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty fixture from {url}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def setup_colab(fixtures: list[str] | None = None) -> None:
    """Install the matching TMol wheel and download tutorial fixtures.

    Safe to call again: the wheel is installed once per runtime.

    Args:
        fixtures: Repository-relative paths to download into the working
            directory. Defaults to the ``FIXTURES`` the tutorials read.
    """
    global _colab_ready
    import torch
    from packaging.version import Version

    if fixtures is None:
        fixtures = [f"{_REPO_DATA_DIR}/{fixture}" for fixture in FIXTURES]

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This tutorial requires CUDA-enabled PyTorch and a Colab GPU runtime. "
            "Choose a GPU under 'Runtime > Change runtime type', reconnect, "
            "and run again."
        )
    cuda_major = (torch.version.cuda or "").split(".")[0]
    if Version(torch.__version__) < Version("2.11") or cuda_major not in {"12", "13"}:
        raise RuntimeError(
            f"TMol {TMOL_RELEASE} needs PyTorch 2.11+ with CUDA 12 or 13; "
            f"this runtime has PyTorch {torch.__version__}, CUDA {torch.version.cuda}."
        )

    if not _colab_ready:
        runtime_packages = [
            "itables>=2.0",
            "py3Dmol>=2.4,<3",
        ]
        _pip_install(
            [f"tmol[cu{cuda_major}]=={TMOL_RELEASE}", *runtime_packages],
            torch.__version__,
        )
        _colab_ready = True

    for relative_path in fixtures:
        _download(relative_path, Path(relative_path))

    print(
        f"Colab GPU environment ready: {torch.cuda.get_device_name(0)}; "
        f"PyTorch {torch.__version__}."
    )


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


# Colab must have TMol installed before any name below is resolved.
if _in_colab():
    setup_colab()


def data_path(*parts: str) -> Path:
    """Path to a checked-in example file, e.g. ``data_path("ncaa_fixtures", "x.cif")``.

    This is under ``tmol/tests/data``. The installed wheel does not ship that
    directory, so on Colab it is the copy ``setup_colab`` downloaded.
    """
    if _in_colab():
        root = Path(_REPO_DATA_DIR)
    else:
        for base in (Path.cwd(), Path(__file__).resolve().parent):
            root = next(
                (
                    parent / _REPO_DATA_DIR
                    for parent in (base, *base.parents)
                    if (parent / _REPO_DATA_DIR).is_dir()
                ),
                None,
            )
            if root is not None:
                break
        else:
            raise FileNotFoundError(
                f"Could not find {_REPO_DATA_DIR}; run from a TMol checkout."
            )
    path = root.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Tutorial data file not found: {path}")
    return path


def setup(seed: int = SEED):
    """Choose the device, seed PyTorch and load the default parameter database.

    Returns:
        ``(device, param_db)``: CUDA when available, otherwise CPU, and
        ``ParameterDatabase.get_default()``.
    """
    import torch

    from tmol.database import ParameterDatabase

    # The tutorials read local example files, not PDB or CCD mirrors.
    logging.getLogger("atomworks.constants").addFilter(
        lambda record: (
            not record.getMessage().startswith(
                (
                    "Environment variable CCD_MIRROR_PATH not set.",
                    "Environment variable PDB_MIRROR_PATH not set.",
                )
            )
        )
    )
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, ParameterDatabase.get_default()


# Names ``from notebook_setup import *`` provides, by the module they come from
# (``_ALIASES`` covers the few that are modules or need renaming). They resolve on
# first use, so importing this module stays cheap.
_EXPORTS = {
    "tmol.database": ("ParameterDatabase",),
    "tmol.io": ("fetch_cif", "file_from_pose_stack", "pose_stack_from_file"),
    "tmol.kinematics": ("BondDOFTypes", "PoseStackKinematicsModule", "torsion_labels"),
    "tmol.ligand": ("smiles_from_pose_stack",),
    "tmol.ops": (
        "block_energies",
        "build_sidechain_coord_mask",
        "calculate_block_pair_ddg",
        "get_chain_mask",
        "per_residue_block_pair_energies",
        "res_mask_to_coord_mask",
    ),
    "tmol.optimization": ("KinForestSfxnNetwork",),
    "tmol.pack": ("PackerPalette", "PackerTask", "pack_rotamers"),
    "tmol.pack.rotamer": ("FixedAAChiSampler", "IncludeCurrentSampler", "OptHSampler"),
    "tmol.pack.rotamer.dunbrack": ("create_dunbrack_sampler_from_database",),
    "tmol.pose": (
        "ConstraintSet",
        "PoseStackBuilder",
        "atom_label",
        "metal_blocks",
        "residue_label",
    ),
    "tmol.score": ("ScoreFunction", "ScoreType", "beta2016_score_function"),
    "tmol.score.constraint": (
        "ConstraintEnergyTerm",
        "constrain_all_ca",
        "create_mainchain_coordinate_constraints",
    ),
}
_ALIASES = {
    "io": "io",
    "np": "numpy",
    "tmol": "tmol",
    "torch": "torch",
    "display": "IPython.display:display",
}
_SOURCES = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = ["SEED", "data_path", "setup", *_ALIASES, *_SOURCES]  # noqa: PLE0604


def __getattr__(name: str):
    if name in _ALIASES:
        module, _, attr = _ALIASES[name].partition(":")
        value = import_module(module)
        if attr:
            value = getattr(value, attr)
    elif name in _SOURCES:
        value = getattr(import_module(_SOURCES[name]), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
