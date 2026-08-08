"""Small, notebook-friendly visualization helpers for molecular structures.

Optional display dependencies are imported only when a helper is called, so
importing :mod:`tmol.io.visualize` does not require IPython or py3Dmol.
"""

from __future__ import annotations

import json
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
    pdb_file.set_structure(model)
    output = StringIO()
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


def _display_html(html: str):
    try:
        from IPython.display import HTML
    except ImportError as exc:
        raise ImportError(
            "switchable_view() and selection_gallery() require IPython."
        ) from exc
    return HTML(html)


def _loader_javascript(on_ready: str) -> str:
    """Return a tiny 3Dmol.js loader followed by ``on_ready``."""
    return f"""
const start = () => {{ {on_ready} }};
if (window.$3Dmol) {{
  start();
}} else {{
  const prior = document.querySelector('script[data-tmol-3dmol]');
  if (prior) {{
    prior.addEventListener('load', start, {{once: true}});
  }} else {{
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/3dmol@2.5.4/build/3Dmol-min.js';
    script.setAttribute('data-tmol-3dmol', 'true');
    script.addEventListener('load', start, {{once: true}});
    script.addEventListener('error', () => {{
      document.getElementById(rootId).textContent =
        'Interactive viewer unavailable; download the notebook to view locally.';
    }}, {{once: true}});
    document.head.appendChild(script);
  }}
}}"""


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
    payload = [
        {
            "label": str(label),
            "pdb": _pdb_text(model),
            "note": str(notes.get(label, "")),
        }
        for label, model in structures.items()
    ]
    root_id = f"tmol-switch-{uuid4().hex}"
    data_id = f"{root_id}-data"
    viewer_id = f"{root_id}-viewer"
    select_id = f"{root_id}-select"
    note_id = f"{root_id}-note"
    encoded = _safe_json(payload)
    on_ready = f"""
const data = JSON.parse(
  document.getElementById({json.dumps(data_id)}).textContent
);
const select = document.getElementById({json.dumps(select_id)});
const note = document.getElementById({json.dumps(note_id)});
const viewer = window.$3Dmol.createViewer(
  document.getElementById({json.dumps(viewer_id)}),
  {{backgroundColor: 'white'}}
);
data.forEach((item, index) => {{
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = item.label;
  select.appendChild(option);
}});
const render = (index) => {{
  const item = data[index];
  viewer.clear();
  viewer.addModel(item.pdb, 'pdb');
  viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
  viewer.addStyle({{hetflag: true}}, {{stick: {{colorscheme: 'orangeCarbon'}}}});
  viewer.zoomTo();
  viewer.render();
  note.textContent = item.note;
}};
select.addEventListener('change', () => render(Number(select.value)));
render(0);
"""
    html = f"""
<div class="tmol-switchable-view" id="{root_id}" style="max-width:{width}px">
  <label>Structure: <select id="{select_id}"></select></label>
  <div class="tmol-viewer-note" id="{note_id}"></div>
  <div id="{viewer_id}" style="width:{width}px;height:{height}px"></div>
</div>
<script type="application/json" id="{data_id}">{encoded}</script>
<script>
(() => {{
  const rootId = {json.dumps(root_id)};
  {_loader_javascript(on_ready)}
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
    width: int = 340,
    height: int = 300,
    highlight_color: str = "#e83e8c",
):
    """Return an HTML gallery highlighting labeled AtomArray selections.

    Selection values may be boolean atom masks. Query strings are also accepted
    when the supplied AtomArray provides a callable ``aa.mask(query)`` method.
    Selection results are resolved in Python and exact PDB atom serials are
    baked into the HTML. This avoids viewer-side query-language differences
    and remains exact when atom names or residue identifiers are duplicated.
    """
    if not _is_atom_array(atom_array):
        raise TypeError("selection_gallery() expects a Biotite AtomArray")
    if not selections:
        raise ValueError("selections must contain at least one labeled selection")

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

    root_id = f"tmol-gallery-{uuid4().hex}"
    data_id = f"{root_id}-data"
    payload = {"pdb": pdb_text, "selections": gallery}
    encoded = _safe_json(payload)
    encoded_highlight_color = _safe_json(highlight_color)
    on_ready = f"""
const data = JSON.parse(
  document.getElementById({json.dumps(data_id)}).textContent
);
const root = document.getElementById(rootId);
data.selections.forEach((selection) => {{
  const card = document.createElement('section');
  card.className = 'tmol-selection-card';
  const title = document.createElement('strong');
  title.textContent = selection.label;
  const viewport = document.createElement('div');
  viewport.style.width = {json.dumps(f"{width}px")};
  viewport.style.height = {json.dumps(f"{height}px")};
  card.append(title, viewport);
  root.appendChild(card);

  const viewer = window.$3Dmol.createViewer(viewport, {{backgroundColor: 'white'}});
  viewer.addModel(data.pdb, 'pdb');
  viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
  const exactAtoms = {{serial: selection.serials}};
  if (selection.serials.length) {{
    viewer.addStyle(
      exactAtoms,
      {{
        stick: {{color: {encoded_highlight_color}, radius: 0.28}},
        sphere: {{color: {encoded_highlight_color}, radius: 0.45}}
      }}
    );
  }}
  viewer.zoomTo(selection.serials.length ? exactAtoms : {{}});
  viewer.render();
}});
"""
    html = f"""
<div class="tmol-selection-gallery" id="{root_id}"></div>
<script type="application/json" id="{data_id}">{encoded}</script>
<script>
(() => {{
  const rootId = {json.dumps(root_id)};
  {_loader_javascript(on_ready)}
}})();
</script>
"""
    return _display_html(html)
