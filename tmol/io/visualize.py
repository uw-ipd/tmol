"""Small, notebook-friendly visualization helpers for molecular structures.

Optional display dependencies are imported only when a helper is called, so
importing :mod:`tmol.io.visualize` does not require IPython or py3Dmol.
"""

from __future__ import annotations

import json
import re
import warnings
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping
from uuid import uuid4

if TYPE_CHECKING:
    from biotite.structure import AtomArray, AtomArrayStack
    from tmol.pose.pose_stack import PoseStack

    StructureInput = PoseStack | AtomArray | AtomArrayStack | str | PathLike[str]
else:
    AtomArray = Any
    AtomArrayStack = Any
    PoseStack = Any
    StructureInput = Any


def pose_stack_to_pdb_string(pose_stack: PoseStack) -> str:
    """Convert a ``PoseStack`` into PDB text suitable for molecular viewers."""
    from tmol.io.pdb_parsing import to_pdb
    from tmol.io.write_pose_stack_pdb import atom_records_from_pose_stack

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
    from tmol.pose.pose_stack import PoseStack

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


def _hex_color(color: str, *, name: str) -> tuple[str, int]:
    """Validate a ``#rrggbb`` color and return it with its integer form.

    Colors reach both a stylesheet and a 3Dmol element-color map, so they are
    restricted to a shape that is safe to interpolate and cheap to convert.
    """
    text = str(color).strip()
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        raise ValueError(f"{name} must be a hex color of the form '#rrggbb'")
    return text, int(text[1:], 16)


def _display_html(html: str):
    try:
        from IPython.display import HTML
    except ImportError as exc:
        raise ImportError(
            "switchable_view() and selection_gallery() require IPython."
        ) from exc
    return HTML(html)


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


VIEWER_SOURCE = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"

#: Highlight color for selected atoms; the carbon color of 3Dmol's
#: ``orangeCarbon`` scheme, as used by the AtomWorks selection widget.
HIGHLIGHT_COLOR = "#ffa500"

#: Fill color of the active pill button, as used by the AtomWorks widget.
_ACTIVE_COLOR = "#e8850c"

# Chrome shared by every switchable viewer, matching the AtomWorks selection
# widget: a row of pill buttons, the active selection written out underneath,
# then the viewer and a one-line interaction hint.
_WIDGET_CSS = """
  .sel-widget { border: 1px solid var(--pst-color-border, #ccc); border-radius: 8px;
                overflow: hidden; margin: 1rem 0; }
  .sel-controls { display: flex; flex-wrap: wrap; gap: 0.4rem; padding: 0.7rem;
                  background: var(--pst-color-surface, #f4f4f4); }
  .sel-controls button { border: 1px solid var(--pst-color-border, #bbb);
    border-radius: 999px; background: var(--pst-color-on-surface, #fff);
    color: inherit; padding: 0.25rem 0.8rem; font-size: 0.85em; cursor: pointer; }
  .sel-desc { padding: 0.7rem 0.9rem;
              border-top: 1px solid var(--pst-color-border, #eee); }
  .sel-desc code { font-weight: 600; }
  .sel-desc-text { font-size: 0.9em; opacity: 0.85; margin-top: 0.25rem; }
  .sel-viewer { position: relative; width: 100%; height: 420px; }
  .sel-hint { font-size: 0.78em; opacity: 0.6; padding: 0.3rem 0.9rem; }
  .sel-error { background: #ffcccc; color: #1b1b1b; font-size: 0.85em;
               margin: 0; padding: 0.7rem 0.9rem; }
"""


def _widget_shell(root_id: str, *, width: int, height: int, hint: str) -> str:
    """Return the shared widget markup for one switchable viewer."""
    return f"""
<style>{_WIDGET_CSS}
  #{root_id} {{ max-width: {width}px; }}
  #{root_id} .sel-viewer {{ height: {height}px; }}
  #{root_id} .sel-controls button.active {{ background: {_ACTIVE_COLOR};
    border-color: {_ACTIVE_COLOR}; color: #fff; }}
</style>
<div class="sel-widget" id="{root_id}">
  <div class="sel-controls"></div>
  <div class="sel-desc"><code></code><div class="sel-desc-text"></div></div>
  <div class="sel-viewer"></div>
  <div class="sel-hint">{hint}</div>
  <p class="sel-error" hidden></p>
</div>"""


def _viewer_script_tag() -> str:
    """Return the plain 3Dmol.js include, loaded like the AtomWorks docs.

    3Dmol.js is a UMD bundle. On a page without an AMD loader a normal
    ``<script src>`` runs synchronously before the widget's own script and sets
    the ``$3Dmol`` global directly. tmol disables nbsphinx's RequireJS injection
    (see ``docs/conf.py``) so this AtomWorks-style direct load works unchanged.
    """
    return f'<script src="{VIEWER_SOURCE}"></script>'


def _widget_prelude_js(root_id: str) -> str:
    """Return the JS that resolves widget elements and guards on 3Dmol.js."""
    return f"""
  var root = document.getElementById({_safe_json(root_id)});
  var controls = root.querySelector(".sel-controls");
  var label = root.querySelector(".sel-desc code");
  var description = root.querySelector(".sel-desc-text");
  var viewerElement = root.querySelector(".sel-viewer");
  var errorElement = root.querySelector(".sel-error");
  if (typeof $3Dmol === "undefined") {{
    errorElement.textContent =
      "3Dmol.js failed to load, so this viewer is not interactive. " +
      "Please check your browser console for error messages.";
    errorElement.hidden = false;
    return;
  }}
"""


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
    highlight_color: str = HIGHLIGHT_COLOR,
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

    The widget uses the same chrome and pill-button switching as
    :func:`selection_gallery` and the AtomWorks selection widget. One viewer is
    reused for every structure and the camera is carried across switches, so
    conformational differences are visible in place instead of being hidden by
    an independent starting orientation per structure.

    Args:
        structures: Ordered mapping of display labels to structures accepted by
            :func:`view`.
        notes: Optional mapping of structure labels to short explanatory text.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
    """
    if not structures:
        raise ValueError("structures must contain at least one labeled structure")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    notes = notes or {}
    gallery = [
        {
            "label": str(label),
            "note": str(notes.get(label, "")),
            "pdb": _pdb_text(model),
        }
        for label, model in structures.items()
    ]

    root_id = f"tmol-switch-{uuid4().hex}"
    html = f"""{_viewer_script_tag()}{_widget_shell(
        root_id,
        width=width,
        height=height,
        hint=(
            "pick a structure above · drag to rotate · scroll to zoom · "
            "click an atom to label it"
        ),
    )}
<script>
(function() {{{_widget_prelude_js(root_id)}
  var presets = {_safe_json(gallery)};
  var viewer = $3Dmol.createViewer(viewerElement, {{backgroundColor: "white"}});
  var camera = null;

  var buttons = presets.map(function(preset, index) {{
    var button = document.createElement("button");
    button.type = "button";
    button.textContent = preset.label;
    button.onclick = function() {{ apply(index); }};
    controls.appendChild(button);
    return button;
  }});

  function apply(index) {{
    var preset = presets[index];
    if (camera !== null) {{
      camera = viewer.getView();
    }}
    viewer.clear();
    viewer.addModel(preset.pdb, "pdb");
    viewer.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
    viewer.addStyle(
      {{not: {{hetflag: true}}}}, {{stick: {{radius: 0.08, opacity: 0.55}}}}
    );
    viewer.setStyle(
      {{hetflag: true}}, {{stick: {{colorscheme: "orangeCarbon", radius: 0.22}}}}
    );
    viewer.setClickable({{}}, true, function(atom) {{
      viewer.addLabel(
        atom.chain + " . " + atom.resn + atom.resi + " . " + atom.atom,
        {{position: atom, backgroundColor: "0x1b1b1b", backgroundOpacity: 0.85,
          fontColor: "white", fontSize: 11}}
      );
      viewer.render();
    }});
    label.textContent = preset.label;
    description.textContent = preset.note;
    buttons.forEach(function(button, buttonIndex) {{
      button.className = buttonIndex === index ? "active" : "";
    }});
    if (camera === null) {{
      viewer.zoomTo();
      camera = viewer.getView();
    }} else {{
      viewer.setView(camera);
    }}
    viewer.render();
  }}

  apply(0);
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
    notes: Mapping[str, str] | None = None,
    width: int = 720,
    height: int = 420,
    highlight_color: str = HIGHLIGHT_COLOR,
):
    """Return one interactive viewer for several labeled AtomArray selections.

    Selection values may be boolean atom masks. Query strings are also accepted
    when the supplied AtomArray provides a callable ``aa.mask(query)`` method.
    Selection results are resolved in Python and exact PDB atom serials are
    baked into the HTML. This avoids viewer-side query-language differences,
    remains exact when atom names or residue identifiers are duplicated, and
    stays aligned when the viewer drops atoms it will not draw, such as the
    alternate locations of a crystal structure.

    Choosing a selection restyles one shared model and animates the camera onto
    the selected atoms, matching the AtomWorks selection widget.
    """
    if not _is_atom_array(atom_array):
        raise TypeError("selection_gallery() expects a Biotite AtomArray")
    if not selections:
        raise ValueError("selections must contain at least one labeled selection")
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    highlight_color, highlight_carbon = _hex_color(
        highlight_color, name="highlight_color"
    )

    notes = notes or {}
    pdb_text = _biotite_to_pdb_string(atom_array)
    serials = _first_model_atom_serials(pdb_text)
    if len(serials) != len(atom_array):
        raise ValueError(
            "The serialized AtomArray atom count does not match the input array"
        )

    gallery = []
    for label, selection in selections.items():
        mask = _selection_mask(atom_array, selection, str(label))
        selected = [serial for serial, keep in zip(serials, mask) if keep]
        note = str(notes.get(label, ""))
        if not note:
            note = f"{len(selected)} of {len(atom_array)} atoms"
        gallery.append(
            {
                "label": str(label),
                "expression": selection if isinstance(selection, str) else str(label),
                "note": note,
                "serials": selected,
            }
        )

    root_id = f"tmol-selection-{uuid4().hex}"
    html = f"""{_viewer_script_tag()}{_widget_shell(
        root_id,
        width=width,
        height=height,
        hint=(
            "pick a selection above · drag to rotate · scroll to zoom · "
            "click a highlighted atom to label it"
        ),
    )}
<script>
(function() {{{_widget_prelude_js(root_id)}
  var pdb = {_safe_json(pdb_text)};
  var presets = {_safe_json(gallery)};
  var highlightColor = {_safe_json(highlight_color)};
  var viewer = $3Dmol.createViewer(viewerElement, {{backgroundColor: "white"}});
  viewer.addModel(pdb, "pdb");
  // Element colors with carbon recolored, which is how 3Dmol builds its own
  // "<color>Carbon" schemes (e.g. orangeCarbon) internally.
  var scheme = {{
    prop: "elem",
    map: Object.assign(
      {{}}, $3Dmol.elementColors.defaultColors, {{C: {highlight_carbon}}}
    )
  }};

  var buttons = presets.map(function(preset, index) {{
    var button = document.createElement("button");
    button.type = "button";
    button.textContent = preset.label;
    button.onclick = function() {{ apply(index); }};
    controls.appendChild(button);
    return button;
  }});

  function apply(index) {{
    var preset = presets[index];
    var selection = preset.serials.length ? {{serial: preset.serials}} : {{}};
    viewer.removeAllLabels();
    viewer.setStyle({{}}, {{cartoon: {{color: "lightgrey", opacity: 0.55}}}});
    if (preset.serials.length) {{
      viewer.addStyle(selection, {{
        stick: {{radius: 0.18, colorscheme: scheme}},
        sphere: {{scale: 0.18, colorscheme: scheme}}
      }});
      viewer.addStyle(selection, {{cartoon: {{color: highlightColor}}}});
      viewer.setClickable(selection, true, function(atom) {{
        viewer.addLabel(
          atom.chain + " . " + atom.resn + atom.resi + " . " + atom.atom,
          {{position: atom, backgroundColor: "0x1b1b1b", backgroundOpacity: 0.85,
            fontColor: "white", fontSize: 11}}
        );
        viewer.render();
      }});
    }}
    label.textContent = preset.expression;
    description.textContent = preset.note;
    buttons.forEach(function(button, buttonIndex) {{
      button.className = buttonIndex === index ? "active" : "";
    }});
    viewer.zoomTo(selection, 400);
    viewer.render();
  }}

  apply(0);
}})();
</script>
"""
    return _display_html(html)
