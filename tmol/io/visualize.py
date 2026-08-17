"""Small, notebook-friendly visualization helpers for molecular structures.

Optional display dependencies are imported only when a helper is called, so
importing :mod:`tmol.io.visualize` does not require IPython or py3Dmol.
"""

from __future__ import annotations

import json
import warnings
from html import escape
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping
from uuid import uuid4

if TYPE_CHECKING:
    from biotite.structure import AtomArray, AtomArrayStack
    from tmol.pose import PoseStack

    StructureInput = PoseStack | AtomArray | AtomArrayStack | str | PathLike[str]
else:
    AtomArray = Any
    AtomArrayStack = Any
    PoseStack = Any
    StructureInput = Any


def pose_stack_to_pdb_string(pose_stack: PoseStack) -> str:
    """Convert a ``PoseStack`` into PDB text suitable for molecular viewers."""
    from tmol.io import to_pdb
    from tmol.io import atom_records_from_pose_stack

    return to_pdb(atom_records_from_pose_stack(pose_stack))


def _is_biotite_structure(model: object) -> bool:
    try:
        from biotite.structure import AtomArray, AtomArrayStack
    except ImportError:
        return False
    return isinstance(model, (AtomArray, AtomArrayStack))


def _is_atom_array(model: object) -> bool:
    try:
        from biotite.structure import AtomArray
    except ImportError:
        return False
    return isinstance(model, AtomArray)


def _biotite_to_pdb_string(model: AtomArray | AtomArrayStack) -> str:
    """Serialize a Biotite structure with Biotite's PDB writer."""
    try:
        from biotite.structure.io.pdb import PDBFile
    except ImportError as exc:
        raise ImportError("Viewing a Biotite structure requires biotite.") from exc

    pdb_file = PDBFile()
    # Biotite warns when PDB's fixed-width compatibility format cannot carry
    # every annotation. The viewer only consumes coordinates and connectivity,
    # so these warnings are expected and would otherwise become alarming red
    # stderr boxes in rendered notebooks.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pdb_file.set_structure(model)
    output = StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pdb_file.write(output)
    return output.getvalue()


def _pdb_text(model: StructureInput) -> str:
    if isinstance(model, PathLike):
        return Path(model).read_text()

    if isinstance(model, str):
        candidate = Path(model)
        if "\n" not in model and candidate.exists():
            return candidate.read_text()
        return model

    if _is_biotite_structure(model):
        return _biotite_to_pdb_string(model)

    # Keep the PoseStack import lazy: importing this module should not load
    # compiled TMol code merely to render PDB text or a Biotite AtomArray.
    from tmol.pose import PoseStack

    if isinstance(model, PoseStack):
        return pose_stack_to_pdb_string(model)

    raise TypeError(
        "Expected a PoseStack, Biotite AtomArray/AtomArrayStack, PDB text, "
        "or a path to a PDB file; "
        f"received {type(model)!r}"
    )


def _first_model_atom_serials(pdb_text: str) -> list[int]:
    """Return atom serials from the first PDB model."""
    serials = []
    in_model = False
    for line in pdb_text.splitlines():
        if line.startswith("MODEL"):
            if in_model and serials:
                break
            in_model = True
            continue
        if line.startswith("ENDMDL") and serials:
            break
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                serials.append(int(line[6:11]))
            except ValueError:
                # Biotite emits valid serials; this fallback keeps hand-written
                # PDB snippets usable with py3Dmol.
                serials.append(len(serials) + 1)
    return serials


def _boolean_mask(mask: object, expected_length: int, *, name: str) -> list[bool]:
    """Normalize a one-dimensional boolean atom mask."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    import numpy

    values = numpy.asarray(mask)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype.kind != "b":
        raise TypeError(f"{name} must contain boolean values")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} has length {len(values)}; expected {expected_length} atoms"
        )
    return values.tolist()


def _safe_json(payload: object) -> str:
    """Encode JSON without allowing data to terminate an HTML script element."""
    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )


def _display_html(html: str):
    try:
        from IPython.display import HTML
    except ImportError as exc:
        raise ImportError(
            "switchable_view() and selection_gallery() require IPython."
        ) from exc
    return HTML(html)


def _viewer_html(viewer: object) -> str:
    """Render a py3Dmol viewer through py3Dmol's supported notebook path."""
    make_html = getattr(viewer, "_make_html", None)
    if not callable(make_html):
        raise RuntimeError("The installed py3Dmol version cannot render HTML")
    return str(make_html())


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
    model: StructureInput,
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
    highlighted: object | None = None,
    highlight_color: str = "#e83e8c",
):
    """Create a draggable py3Dmol viewer for a molecular structure.

    ``model`` may be a :class:`PoseStack`, a Biotite ``AtomArray`` or
    ``AtomArrayStack``, PDB text, or a PDB path. ``highlighted`` is an optional
    boolean mask over the atoms in the first model; highlighted atoms are shown
    as thicker sticks and spheres. The return value remains a real
    ``py3Dmol.view`` object for compatibility with existing notebooks.
    """
    try:
        import py3Dmol
    except ImportError as exc:
        raise ImportError(
            "tmol.view() requires py3Dmol. Install it with "
            "`python -m pip install py3Dmol` or `python -m pip install -e '.[docs]'`."
        ) from exc

    pdb_text = _pdb_text(model)
    serials = _first_model_atom_serials(pdb_text)
    highlighted_serials: list[int] = []
    if highlighted is not None:
        mask = _boolean_mask(highlighted, len(serials), name="highlighted")
        highlighted_serials = [
            serial for serial, selected in zip(serials, mask) if selected
        ]

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(pdb_text, "pdb")
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

    if highlighted_serials:
        viewer.addStyle(
            {"serial": highlighted_serials},
            {
                "stick": {"color": highlight_color, "radius": 0.28},
                "sphere": {"color": highlight_color, "radius": 0.45},
            },
        )

    if show_hover:
        _add_hover_labels(viewer)

    if zoom_to is None:
        viewer.zoomTo()
    else:
        viewer.zoomTo(zoom_to)

    return viewer


def switchable_view(
    structures: Mapping[str, StructureInput],
    *,
    notes: Mapping[str, str] | None = None,
    width: int = 720,
    height: int = 420,
):
    """Return HTML that switches one 3Dmol viewer among labeled structures.

    Args:
        structures: Ordered mapping of display labels to structures accepted by
            :func:`view`.
        notes: Optional mapping of structure labels to short explanatory text.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
    """
    if not structures:
        raise ValueError("structures must contain at least one labeled structure")

    notes = notes or {}
    root_id = f"tmol-switch-{uuid4().hex}"
    select_id = f"{root_id}-select"
    panels = []
    options = []
    for index, (label, model) in enumerate(structures.items()):
        viewer = view(model, width=width, height=height)
        panel_id = f"{root_id}-panel-{index}"
        note = str(notes.get(label, ""))
        options.append(f'<option value="{index}">{escape(str(label))}</option>')
        panels.append(f"""
<section id="{panel_id}" class="tmol-switch-panel"
         style="position:absolute;inset:0;visibility:{'visible' if index == 0 else 'hidden'}">
  <div class="tmol-viewer-note">{escape(note)}</div>
  {_viewer_html(viewer)}
</section>""")
    html = f"""
<div class="tmol-switchable-view" id="{root_id}" style="max-width:{width}px">
  <label>Structure: <select id="{select_id}">{''.join(options)}</select></label>
  <div style="position:relative;width:100%;height:{height + 32}px">
    {''.join(panels)}
  </div>
</div>
<script>
(() => {{
  const root = document.getElementById({json.dumps(root_id)});
  const select = document.getElementById({json.dumps(select_id)});
  const panels = root.querySelectorAll('.tmol-switch-panel');
  select.addEventListener('change', () => {{
    panels.forEach((panel, index) => {{
      panel.style.visibility = index === Number(select.value) ? 'visible' : 'hidden';
    }});
    window.dispatchEvent(new Event('resize'));
  }});
}})();
</script>
"""
    return _display_html(html)


def _selection_mask(atom_array: AtomArray, selection: object, label: str) -> list[bool]:
    if isinstance(selection, str):
        query = getattr(atom_array, "mask", None)
        if not callable(query):
            raise TypeError(
                f"Selection {label!r} is a query string, but this AtomArray "
                "does not provide a callable mask() method"
            )
        selection = query(selection)
    return _boolean_mask(selection, len(atom_array), name=f"selection {label!r}")


def selection_gallery(
    atom_array: AtomArray,
    selections: Mapping[str, object],
    *,
    width: int = 720,
    height: int = 420,
    highlight_color: str = "#e83e8c",
):
    """Return one interactive viewer for several labeled AtomArray selections.

    Selection values may be boolean atom masks. Query strings are also accepted
    when the supplied AtomArray provides a callable ``aa.mask(query)`` method.
    Selection results are resolved in Python and exact PDB atom serials are
    baked into the HTML. This avoids viewer-side query-language differences
    and remains exact when atom names or residue identifiers are duplicated.
    Clicking a label restyles the same model and animates the camera to the
    selected atoms, following the AtomWorks selection-gallery interaction.
    """
    if not _is_atom_array(atom_array):
        raise TypeError("selection_gallery() expects a Biotite AtomArray")
    if not selections:
        raise ValueError("selections must contain at least one labeled selection")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    pdb_text = _biotite_to_pdb_string(atom_array)
    serials = _first_model_atom_serials(pdb_text)
    if len(serials) != len(atom_array):
        raise ValueError(
            "The serialized AtomArray atom count does not match the input array"
        )

    gallery = []
    for label, selection in selections.items():
        mask = _selection_mask(atom_array, selection, str(label))
        selected_serials = [
            serial for serial, selected in zip(serials, mask) if selected
        ]
        gallery.append({"label": str(label), "serials": selected_serials})

    root_id = f"tmol-selection-{uuid4().hex}"
    controls_id = f"{root_id}-controls"
    viewer_id = f"{root_id}-viewer"
    html = f"""
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
<style>
  #{root_id} {{
    border: 1px solid var(--pst-color-border, #ccc);
    border-radius: 8px;
    max-width: {width}px;
    overflow: hidden;
  }}
  #{controls_id} {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    padding: 0.7rem;
    background: var(--pst-color-surface, #f4f4f4);
  }}
  #{controls_id} button {{
    border: 1px solid var(--pst-color-border, #bbb);
    border-radius: 999px;
    background: var(--pst-color-on-surface, #fff);
    color: inherit;
    cursor: pointer;
    font-size: 0.85em;
    padding: 0.25rem 0.8rem;
  }}
  #{controls_id} button.active {{
    color: #fff;
  }}
  #{viewer_id} {{ position: relative; width: 100%; height: {height}px; }}
  #{root_id} .tmol-selection-hint {{
    font-size: 0.78em;
    opacity: 0.65;
    padding: 0.35rem 0.9rem;
  }}
</style>
<div id="{root_id}">
  <div id="{controls_id}"></div>
  <div id="{viewer_id}"></div>
  <div class="tmol-selection-hint">
    pick a selection · drag to rotate · scroll to zoom · click a highlighted atom to label it
  </div>
</div>
<script>
(() => {{
  const root = document.getElementById({json.dumps(root_id)});
  const controls = document.getElementById({json.dumps(controls_id)});
  const viewerElement = document.getElementById({json.dumps(viewer_id)});
  const pdb = {_safe_json(pdb_text)};
  const presets = {_safe_json(gallery)};
  const highlightColor = {_safe_json(highlight_color)};

  function initialize() {{
    if (!window.$3Dmol) {{
      window.setTimeout(initialize, 50);
      return;
    }}
    const viewer = window.$3Dmol.createViewer(
      viewerElement, {{backgroundColor: "white"}}
    );
    viewer.addModel(pdb, "pdb");
    const buttons = presets.map((preset, index) => {{
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = preset.label;
      button.addEventListener("click", () => apply(index));
      controls.appendChild(button);
      return button;
    }});

    function apply(index) {{
      const preset = presets[index];
      const selection = preset.serials.length
        ? {{serial: preset.serials}}
        : {{}};
      viewer.removeAllLabels();
      viewer.setStyle(
        {{}}, {{cartoon: {{color: "lightgrey", opacity: 0.55}}}}
      );
      if (preset.serials.length) {{
        viewer.addStyle(selection, {{
          stick: {{color: highlightColor, radius: 0.18}},
          sphere: {{color: highlightColor, radius: 0.35}},
          cartoon: {{color: highlightColor}}
        }});
        viewer.setClickable(selection, true, (atom) => {{
          viewer.addLabel(
            atom.chain + ":" + atom.resn + atom.resi + ":" + atom.atom,
            {{
              position: atom,
              backgroundColor: "0x1b1b1b",
              backgroundOpacity: 0.85,
              fontColor: "white",
              fontSize: 11
            }}
          );
          viewer.render();
        }});
      }}
      buttons.forEach((button, buttonIndex) => {{
        const active = buttonIndex === index;
        button.className = active ? "active" : "";
        button.style.backgroundColor = active ? highlightColor : "";
        button.style.borderColor = active ? highlightColor : "";
      }});
      viewer.zoomTo(selection, 400);
      viewer.render();
    }}

    apply(0);
  }}
  initialize();
}})();
</script>
"""
    return _display_html(html)
