"""The scripted teacher (pathfinding, trigonometry, if/else control) is training-time
only: no module under agent/, env/ or evaluate/ (the inference path) may import
`teachers`, directly or by `from teachers... import`, or via importlib with a literal."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_PACKAGES = ("agent", "env", "evaluate")
FORBIDDEN = "teachers"


def _imports(path: str) -> list[str]:
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
        elif isinstance(node, ast.Call):   # importlib.import_module("teachers...") / __import__
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in ("import_module", "__import__") and node.args and \
                    isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                out.append(node.args[0].value)
    return out


def _py_files(pkg: str) -> list[str]:
    files = []
    for dp, _, fns in os.walk(os.path.join(ROOT, pkg)):
        files.extend(os.path.join(dp, fn) for fn in fns if fn.endswith(".py"))
    return files


def test_inference_path_never_imports_teachers() -> None:
    bad = []
    for pkg in INFERENCE_PACKAGES:
        files = _py_files(pkg)
        assert files, f"{pkg}/ has no Python files: the scan would prove nothing"
        for p in files:
            for name in _imports(p):
                if name == FORBIDDEN or name.startswith(FORBIDDEN + "."):
                    bad.append((os.path.relpath(p, ROOT), name))
    assert not bad, f"inference-path modules import the scripted teacher: {bad}"


def test_scanner_sees_real_teacher_imports() -> None:
    """The scanner works: teachers/collect.py imports teachers.controller."""
    names = _imports(os.path.join(ROOT, "teachers", "collect.py"))
    assert any(n.startswith("teachers.") for n in names), names
