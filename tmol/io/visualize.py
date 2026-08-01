"""Visualization helpers for TMol structures."""

from os import PathLike
from pathlib import Path
from typing import Literal

from tmol.pose.pose_stack import PoseStack


def pose_stack_to_pdb_string(pose_stack: PoseStack) -> str:
    """Convert a ``PoseStack`` into PDB text suitable for molecular viewers."""
    from tmol.io.pdb_parsing import to_pdb
    from tmol.io.write_pose_stack_pdb import atom_records_from_pose_stack

    return to_pdb(atom_records_from_pose_stack(pose_stack))


def _pdb_text(model: PoseStack | str | PathLike[str]) -> str:
    if isinstance(model, PoseStack):
        return pose_stack_to_pdb_string(model)

    if isinstance(model, PathLike):
        return Path(model).read_text()

    if isinstance(model, str):
        candidate = Path(model)
        if "\n" not in model and candidate.exists():
            return candidate.read_text()
        return model

    raise TypeError(
        "view() expects a PoseStack, PDB text, or a path to a PDB file; "
        f"received {type(model)!r}"
    )


def _add_hover_labels(viewer) -> None:
    hover_in = """function(atom,viewer) {
        if(!atom.label) {
            atom.label = viewer.addLabel(
                atom.chain + ':' + atom.resn + '(' + atom.resi + '):' +
                atom.atom + '(idx' + atom.serial + ')',
                {position: atom, backgroundColor:"white", fontColor:"black"}
            );
        }
    }"""
    hover_out = """function(atom,viewer) {
        if(atom.label) {
            viewer.removeLabel(atom.label);
            delete atom.label;
        }
    }"""
    viewer.setHoverable({}, True, hover_in, hover_out)


def view(
    model: PoseStack | str | PathLike[str],
    *,
    width: int = 720,
    height: int = 420,
    style: Literal["cartoon", "stick"] = "cartoon",
    background_color: str = "white",
    cartoon_color: str = "spectrum",
    show_sidechains: bool = True,
    show_heteroatoms: bool = True,
    show_hover: bool = True,
    zoom_to: dict | None = None,
):
    """Create a draggable py3Dmol viewer for a TMol structure.

    The return value is a real ``py3Dmol.view`` object, matching the display
    contract used by AtomWorks and RFD4 notebooks. Sphinx/nbsphinx can preserve
    its HTML and ``application/3dmoljs_load.v0`` MIME output in built docs.
    """
    try:
        import py3Dmol
    except ImportError as exc:
        raise ImportError(
            "tmol.view() requires py3Dmol. Install it with "
            "`python -m pip install py3Dmol` or `python -m pip install -e '.[docs]'`."
        ) from exc

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(_pdb_text(model), "pdb")
    viewer.setBackgroundColor(background_color)

    if style == "stick":
        viewer.setStyle({}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.18}})
    elif style == "cartoon":
        viewer.setStyle({}, {"cartoon": {"color": cartoon_color}})
        if show_sidechains:
            viewer.addStyle(
                {"not": {"hetflag": True}},
                {"stick": {"radius": 0.08, "opacity": 0.55}},
            )
    else:
        raise ValueError("style must be either 'cartoon' or 'stick'")

    if show_heteroatoms:
        viewer.setStyle(
            {"hetflag": True},
            {"stick": {"colorscheme": "orangeCarbon", "radius": 0.22}},
        )

    if show_hover:
        _add_hover_labels(viewer)

    if zoom_to is None:
        viewer.zoomTo()
    else:
        viewer.zoomTo(zoom_to)

    return viewer
