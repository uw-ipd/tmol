import ast
from pathlib import Path

import tmol


_ALLOWED_PRIVATE_MODULES = {"_load_ext", "_cpp_lib"}


def _imports_private_tmol_extension(module_ast: ast.AST) -> bool:
    for node in ast.walk(module_ast):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.startswith("tmol."):
            continue
        base = node.module.rsplit(".", 1)[-1]
        if not base.startswith("_"):
            continue
        if base in _ALLOWED_PRIVATE_MODULES:
            continue
        return True
    return False


def _has_jit_selector_call(module_ast: ast.AST) -> bool:
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "ensure_compiled_or_jit":
                return True
    return False


def test_private_extension_imports_are_gated_for_jit():
    tmol_root = Path(tmol.__file__).resolve().parent
    offenders = []

    for file_path in tmol_root.rglob("*.py"):
        if "tests" in file_path.parts:
            continue
        module_ast = ast.parse(file_path.read_text())
        if not _imports_private_tmol_extension(module_ast):
            continue
        if _has_jit_selector_call(module_ast):
            continue
        offenders.append(str(file_path.relative_to(tmol_root.parent)))

    assert not offenders, (
        "Private extension imports must be gated by ensure_compiled_or_jit(): "
        + ", ".join(sorted(offenders))
    )
