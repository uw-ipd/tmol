import ast
import inspect
from pathlib import Path

import tmol.database
import tmol.io.pose_stack_from_biotite
import tmol.ligand
import tmol.relax.fast_relax
import tmol.score


def _assert_google_docstring(obj) -> None:
    doc = inspect.getdoc(obj)
    assert doc is not None
    assert "Args:" in doc
    assert "Returns:" in doc


def test_public_api_docstrings_have_google_sections():
    _assert_google_docstring(tmol.ligand.prepare_ligands)
    _assert_google_docstring(tmol.io.pose_stack_from_biotite.pose_stack_from_biotite)
    _assert_google_docstring(tmol.io.pose_stack_from_biotite.build_context_from_biotite)
    _assert_google_docstring(tmol.database.ParameterDatabase.add_residue_type)
    _assert_google_docstring(tmol.score.beta2016_score_function)
    _assert_google_docstring(tmol.relax.fast_relax.fast_relax)


def test_library_modules_do_not_use_print_statements():
    files = [
        Path(tmol.ligand.__file__),
        Path(tmol.io.pose_stack_from_biotite.__file__),
        Path(tmol.database.__file__),
        Path(tmol.score.__file__),
        Path(tmol.relax.fast_relax.__file__),
    ]
    for file_path in files:
        module_ast = ast.parse(file_path.read_text())
        print_calls = [
            node
            for node in ast.walk(module_ast)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ]
        assert not print_calls, f"print() found in {file_path}"
