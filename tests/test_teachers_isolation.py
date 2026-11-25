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


# ---------------------------------------------------------------- transitive
def _first_party_top() -> set[str]:
    return {d for d in os.listdir(ROOT)
            if os.path.isfile(os.path.join(ROOT, d, "__init__.py"))}


def _module_file(mod: str) -> str | None:
    base = os.path.join(ROOT, *mod.split("."))
    for p in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(p):
            return p
    return None


def _module_name(path: str) -> str:
    rel = os.path.relpath(path, ROOT)[:-3].replace(os.sep, ".")
    return rel[: -len(".__init__")] if rel.endswith(".__init__") else rel


def _resolved_imports(path: str) -> set[str]:
    """Absolute module names imported by `path`, relative imports resolved, each
    `from X import y` also tried as module X.y (a submodule import)."""
    me = _module_name(path)
    pkg = me if path.endswith("__init__.py") else me.rpartition(".")[0]
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out: set[str] = set(_imports(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".")
                parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join(p for p in (*parts, base) if p)
            if base:
                out.add(base)
                out.update(f"{base}.{a.name}" for a in node.names)
    return out


def test_inference_path_never_reaches_teachers_transitively() -> None:
    """Follow every first-party import (any depth, through train/, configs/, maps/, ...)
    from every module under agent/, env/, evaluate/: `teachers` is never reached."""
    top = _first_party_top()
    assert FORBIDDEN in top
    start = [p for pkg in INFERENCE_PACKAGES for p in _py_files(pkg)]
    seen: dict[str, str | None] = {}
    stack = [(p, None) for p in start]
    bad = []
    while stack:
        path, via = stack.pop()
        name = _module_name(path)
        if name in seen:
            continue
        seen[name] = via
        for mod in _resolved_imports(path):
            if mod.split(".")[0] not in top:
                continue
            if mod == FORBIDDEN or mod.startswith(FORBIDDEN + "."):
                chain, cur = [mod], name
                while cur is not None:
                    chain.append(cur)
                    cur = seen.get(cur)
                bad.append(" <- ".join(chain))
                continue
            f = _module_file(mod)
            if f is not None and _module_name(f) not in seen:
                stack.append((f, name))
    assert len(seen) > len(start), "the walk followed no import: it would prove nothing"
    assert not bad, f"inference path reaches the scripted teacher: {bad}"


def test_transitive_walk_detects_a_planted_chain(tmp_path) -> None:
    """The walker is not vacuous: a module that imports a module that imports teachers
    is reported."""
    global ROOT
    for pkg, body in (("agent", "from mid import x\n"), ("env", "import os\n"), ("evaluate", "import os\n"),
                      ("mid", "from teachers.controller import y\n"), ("teachers", "")):
        d = tmp_path / pkg
        d.mkdir()
        (d / "__init__.py").write_text(body if pkg != "mid" else "")
        if pkg in ("agent", "mid"):
            (d / "m.py").write_text(body if pkg == "mid" else "from mid.m import z\n")
    import pytest
    old = ROOT
    try:
        ROOT = str(tmp_path)
        with pytest.raises(AssertionError, match="reaches the scripted teacher"):
            test_inference_path_never_reaches_teachers_transitively()
    finally:
        ROOT = old
