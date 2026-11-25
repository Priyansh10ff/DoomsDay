"""Map QA (graph search) and the curriculum scheduler must never be reachable from the
agent or the inference path: no module under agent/ or env/ imports maps.qa,
maps.generator or curriculum. maps.qa is also not imported by train/."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = {"agent": ("maps", "curriculum"), "env": ("maps", "curriculum"),
             "train": ("maps.qa",)}


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            for a in node.names:
                yield f"{node.module}.{a.name}"


def test_no_forbidden_imports():
    bad = []
    for d, prefixes in FORBIDDEN.items():
        for dp, _, files in os.walk(os.path.join(ROOT, d)):
            for fn in files:
                if fn.endswith(".py"):
                    p = os.path.join(dp, fn)
                    for name in _imports(p):
                        if any(name == x or name.startswith(x + ".") for x in prefixes):
                            bad.append((os.path.relpath(p, ROOT), name))
    assert not bad, bad


def test_scan_sees_real_imports():
    """The scanner itself works: maps/build.py does import maps.qa."""
    assert any(n.startswith("maps.qa") for n in _imports(os.path.join(ROOT, "maps", "build.py")))
