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

# atomworks.constants warns the moment it's imported if these are unset, which
# happens as soon as anything below resolves a lazy tmol name -- including the
# very first `from notebook_setup import *`, well before setup() could install
# a filter. Installed here, at module import time, so it's in place first.
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

TUTORIAL_REF = "master"
RAW_BASE = f"https://raw.githubusercontent.com/uw-ipd/tmol/{TUTORIAL_REF}"
# Advance only after the release wheels have been published and exercised.
# The source project's next version may not have downloadable wheels yet.
TMOL_RELEASE = "0.2.0"

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

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, ParameterDatabase.get_default()


def connect_atoms(
    viewer, pose_stack, atom_a, atom_b, *, radius: float = 0.12, color: str = "yellow"
):
    """Draw a cylinder between two atoms of a PoseStack in a ``tmol.view`` viewer.

    ``atom_a``/``atom_b`` are each a ``(pose, block, atom)`` index triple
    (e.g. from ``tmol.atom_for_residue_label``, or one row of a
    ``ConstraintSet``'s ``constraint_atoms``). Call before ``viewer.show()``.
    Useful for making a specific atom pair -- e.g. the two atoms of a
    distance constraint -- visually obvious when highlighting alone is easy
    to miss.
    """

    def coord_for(atom):
        pose_ind, block_ind, atom_ind = (int(x) for x in atom)
        offset = int(pose_stack.block_coord_offset[pose_ind, block_ind])
        return pose_stack.coords[pose_ind, offset + atom_ind]

    coord_a = coord_for(atom_a)
    coord_b = coord_for(atom_b)
    viewer.addCylinder(
        {
            "start": {
                "x": float(coord_a[0]),
                "y": float(coord_a[1]),
                "z": float(coord_a[2]),
            },
            "end": {
                "x": float(coord_b[0]),
                "y": float(coord_b[1]),
                "z": float(coord_b[2]),
            },
            "radius": radius,
            "color": color,
        }
    )


def view_constraints(
    pose_stack, pose_ind: int = 0, *, radius: float = 0.12, color: str = "yellow"
):
    """Render one pose with every attached constraint highlighted and connected.

    Highlights every atom that appears in one of ``pose_stack``'s attached
    constraints (``pose_stack.constraint_set``) for the given pose, and
    connects each constraint's atoms in sequence with :func:`connect_atoms`
    -- a coordinate constraint's single atom is just highlighted, a distance
    constraint's two atoms get one connecting cylinder, and a torsion-angle
    constraint's four atoms get three, tracing the dihedral. Calls
    ``viewer.show()``; returns nothing, so calling this as a cell's last
    statement doesn't also trigger the notebook's own auto-display of a
    returned viewer (which would otherwise render it a second time).
    """
    import torch
    import tmol

    constraint_set = pose_stack.constraint_set
    highlight_mask = torch.zeros_like(pose_stack.real_atoms)

    constraints = []
    if constraint_set is not None:
        for row in constraint_set.constraint_atoms.tolist():
            if row[0][0] != pose_ind:
                continue
            atoms = []
            for triple in row:
                if triple[0] == -1:
                    continue
                if atoms and tuple(triple) == tuple(atoms[-1]):
                    continue
                atoms.append(triple)
                offset = int(pose_stack.block_coord_offset[triple[0], triple[1]])
                highlight_mask[triple[0], offset + triple[2]] = True
            if atoms:
                constraints.append(atoms)

    viewer = tmol.view(
        pose_stack,
        highlighted=highlight_mask[pose_ind][pose_stack.real_atoms[pose_ind]],
    )
    for atoms in constraints:
        for atom_a, atom_b in zip(atoms, atoms[1:]):
            connect_atoms(
                viewer, pose_stack, atom_a, atom_b, radius=radius, color=color
            )
    viewer.show()


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
        "select_blocks_from_pose_stack",
        "select_from_pose_stack",
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
        "constrain_closest_interchain_ca_pair",
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

__all__ = [  # noqa: PLE0604
    "SEED",
    "connect_atoms",
    "data_path",
    "setup",
    "view_constraints",
    *_ALIASES,
    *_SOURCES,
]


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
