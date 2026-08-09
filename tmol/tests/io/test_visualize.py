"""Pure-Python tests for notebook visualization helpers."""

from __future__ import annotations

import builtins
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import biotite.structure as struc
import numpy
import pytest

from tmol.io import visualize


def _atom_array() -> struc.AtomArray:
    atoms = struc.AtomArray(3)
    atoms.coord = numpy.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 1.2, 0.0]])
    atoms.chain_id = numpy.array(["A", "A", "A"])
    atoms.res_id = numpy.array([1, 1, 1])
    atoms.res_name = numpy.array(["GLY", "GLY", "GLY"])
    atoms.atom_name = numpy.array(["N", "CA", "C"])
    atoms.element = numpy.array(["N", "C", "C"])
    atoms.hetero = numpy.array([False, False, False])
    return atoms


def test_module_import_does_not_require_display_dependencies(monkeypatch):
    module_path = Path(visualize.__file__)
    spec = importlib.util.spec_from_file_location(
        "_tmol_visualize_lazy_test", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "py3Dmol" or name.startswith("IPython"):
            raise AssertionError(f"optional dependency imported eagerly: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def test_biotite_atom_array_and_stack_are_written_as_pdb():
    atoms = _atom_array()
    pdb_text = visualize._pdb_text(atoms)
    assert " GLY A   1" in pdb_text
    assert len(visualize._first_model_atom_serials(pdb_text)) == len(atoms)

    stack = struc.stack([atoms, atoms.copy()])
    stack_text = visualize._pdb_text(stack)
    assert "MODEL        1" in stack_text
    assert "MODEL        2" in stack_text


class _Viewer:
    def __init__(self):
        self.models = []
        self.styles = []
        self.added_styles = []

    def addModel(self, text, file_format):
        self.models.append((text, file_format))

    def setBackgroundColor(self, color):
        self.background = color

    def setStyle(self, selection, style):
        self.styles.append((selection, style))

    def addStyle(self, selection, style):
        self.added_styles.append((selection, style))

    def setHoverable(self, *args):
        self.hover = args

    def zoomTo(self, *args):
        self.zoom = args

    def _make_html(self):
        return '<div class="mock-py3dmol-viewer"></div>'


def test_view_highlights_validated_biotite_mask(monkeypatch):
    viewer = _Viewer()
    monkeypatch.setitem(
        sys.modules,
        "py3Dmol",
        SimpleNamespace(view=lambda **kwargs: viewer),
    )

    result = visualize.view(
        _atom_array(),
        highlighted=numpy.array([False, True, False]),
        show_hover=False,
    )

    assert result is viewer
    assert viewer.models[0][1] == "pdb"
    selection, style = viewer.added_styles[-1]
    assert selection == {"serial": [2]}
    assert set(style) == {"stick", "sphere"}


@pytest.mark.parametrize(
    ("mask", "error"),
    [
        ([True, False], ValueError),
        ([1, 0, 0], TypeError),
        ([[True, False, False]], ValueError),
    ],
)
def test_view_rejects_invalid_highlight_masks(monkeypatch, mask, error):
    monkeypatch.setitem(
        sys.modules,
        "py3Dmol",
        SimpleNamespace(view=lambda **kwargs: _Viewer()),
    )
    with pytest.raises(error):
        visualize.view(_atom_array(), highlighted=mask, show_hover=False)


def test_switchable_view_escapes_payload_and_uses_unique_ids(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "py3Dmol",
        SimpleNamespace(view=lambda **kwargs: _Viewer()),
    )
    unsafe = "</script><img src=x onerror=alert(1)>"
    first = visualize.switchable_view({"before": "ATOM\n"}, notes={"before": unsafe})
    second = visualize.switchable_view({"before": "ATOM\n"})

    assert unsafe not in first.data
    assert "&lt;/script&gt;" in first.data
    first_match = re.search(r'id="(tmol-switch-[a-f0-9]+)"', first.data)
    second_match = re.search(r'id="(tmol-switch-[a-f0-9]+)"', second.data)
    assert first_match is not None
    assert second_match is not None
    first_id = first_match.group(1)
    second_id = second_match.group(1)
    assert first_id != second_id
    assert "mock-py3dmol-viewer" in first.data
    assert "visibility:visible" in first.data
    assert "window.dispatchEvent(new Event('resize'))" in first.data


def test_selection_gallery_uses_one_switchable_viewer_and_escapes_payload():
    atoms = _atom_array()
    unsafe_label = "</script><selected>"
    first = visualize.selection_gallery(
        atoms,
        {
            unsafe_label: numpy.array([False, True, False]),
            "all": numpy.array([True, True, True]),
        },
    )
    second = visualize.selection_gallery(
        atoms, {"other": numpy.array([True, False, False])}
    )

    assert unsafe_label not in first.data
    assert r"\u003c/script\u003e\u003cselected\u003e" in first.data
    assert '"serials": [2]' in first.data
    assert '"serials": [1, 2, 3]' in first.data
    assert first.data.count("$3Dmol.createViewer") == 1
    assert "viewer.zoomTo(selection, 400)" in first.data
    assert 'button.addEventListener("click"' in first.data
    first_match = re.search(r'id="(tmol-selection-[a-f0-9]+)"', first.data)
    second_match = re.search(r'id="(tmol-selection-[a-f0-9]+)"', second.data)
    assert first_match is not None
    assert second_match is not None
    assert first_match.group(1) != second_match.group(1)


def test_query_selection_uses_mask_method_when_available():
    class QueryArray:
        def __len__(self):
            return 3

        def mask(self, query):
            assert query == "chain A"
            return numpy.array([True, False, True])

    assert visualize._selection_mask(QueryArray(), "chain A", "protein") == [
        True,
        False,
        True,
    ]


def test_query_selection_requires_mask_method():
    with pytest.raises(TypeError, match=r"callable mask\(\)"):
        visualize._selection_mask(_atom_array(), "chain A", "protein")
