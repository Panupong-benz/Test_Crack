#!/usr/bin/env python3
"""check_module_scope - names used at IMPORT TIME that were never bound there.

Amendment A1.25. The A6 trainer died on its first real execution since A1.22
with

    line 55: if _BENCH_DIR.exists() and str(_BENCH_DIR) not in sys.path:
    NameError: name 'sys' is not defined

because the file's only `import sys` sits INSIDE the multi-GPU launcher
function, so module scope never bound the name. No gate caught it: the dev box
cannot import that module at all (no sam3 / triton / bitsandbytes), and A1.22's
offline resume proof exercised benchmark/resume_state.py through stubs, never
the trainer itself.

The rule this closes: a file that cannot be imported on the dev box must have a
STATIC gate, or its import-time code is only ever tested by paying for a GPU.

What it checks - deliberately narrow, because module-level code is short and a
narrow check has no false positives to argue with:

  * collect every name BOUND at module level - imports, assignments, def/class
    names, `except ... as`, for targets, with-as, and any parameter of a lambda
    appearing in a module-level expression - plus builtins;
  * flag every name READ at module level that is not in that set.

Function and class BODIES are never descended into: a name imported inside a
function is that function's business, and pruning those bodies is exactly what
separates this check from a naive walk. An earlier version without the pruning
got it wrong in both directions at once - it MISSED the real `sys` (it counted
the function-local `import sys` as a module binding) and it invented two
findings in infer_sam.py (`_e` from an `except ... as`, `iterable` from a def
nested in a try).

Usage:
  python check_module_scope.py                 # default target list
  python check_module_scope.py FILE [FILE ...]
  python check_module_scope.py --selftest
"""
from __future__ import annotations

import argparse
import ast
import builtins
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

SCOPED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

# Everything that runs on the rented box. Missing entries are skipped, so the
# same list works from the deploy layout (model files at the repo root) and
# from the canonical tree (benchmark/ only).
DEFAULT_GLOBS = [
    "benchmark/*.py",
    "train_sam3_lora_native_claude.py",
    "tile_dataset.py",
    "cldice_loss.py",
    "lora_layers.py",
    "infer_sam.py",
]

_DUNDER = {"__file__", "__name__", "__doc__", "__spec__", "__package__",
           "__builtins__", "__loader__", "__debug__"}


def _module_level_nodes(stmts):
    """Every node reachable from module-level statements, NOT descending into
    function or class bodies (the def/class node itself is still yielded so its
    name gets bound)."""
    for st in stmts:
        if isinstance(st, SCOPED):
            yield st
            continue
        yield st
        for _field, val in ast.iter_fields(st):
            items = val if isinstance(val, list) else [val]
            for it in items:
                if isinstance(it, ast.stmt):
                    yield from _module_level_nodes([it])
                elif isinstance(it, ast.excepthandler):
                    yield it
                    yield from _module_level_nodes(it.body)
                elif isinstance(it, ast.AST):
                    for sub in ast.walk(it):
                        yield sub


def analyse(path: Path):
    """-> sorted [(lineno, name)] read at module level but never bound there."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound = set(dir(builtins)) | _DUNDER
    nodes = list(_module_level_nodes(tree.body))

    for n in nodes:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, SCOPED) and not isinstance(n, ast.Lambda):
            bound.add(n.name)
        elif isinstance(n, ast.excepthandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, ast.Global):
            bound.update(n.names)

    bad = {(n.lineno, n.id) for n in nodes
           if isinstance(n, ast.Name)
           and isinstance(n.ctx, ast.Load)
           and n.id not in bound}
    return sorted(bad)


def targets(root: Path):
    out = []
    for g in DEFAULT_GLOBS:
        if "*" in g:
            out.extend(sorted(root.glob(g)))
        elif (root / g).exists():
            out.append(root / g)
    return out


def check(paths) -> int:
    hits = 0
    for p in paths:
        try:
            bad = analyse(p)
        except SyntaxError as e:
            print(f"SYNTAX ERROR {p}: {e}")
            hits += 1
            continue
        if bad:
            hits += 1
            for line, name in bad:
                print(f"{p}:{line}: name '{name}' is read at import time but "
                      f"never bound at module scope")
    if hits:
        print(f"check_module_scope FAIL: {hits} file(s) of {len(paths)}. "
              f"This code raises NameError the moment it is imported - no "
              f"runtime test the dev box can run would ever reach it.")
        return 1
    print(f"check_module_scope PASS: {len(paths)} file(s), no name read at "
          f"import time is unbound")
    return 0


_BAD = """
from pathlib import Path
D = Path(".")
if str(D) not in sys.path:
    sys.path.insert(0, str(D))
"""

_GOOD = """
import os
from pathlib import Path

try:
    import cv2
except ImportError as _e:
    raise ImportError("need cv2") from _e

try:
    from tqdm import tqdm as _t
except ImportError:
    def _t(iterable, **_kw):
        return iterable

KEY = lambda row, idx: row[idx]
ROOT = Path(os.getcwd())
for _w in ("a", "b"):
    LAST = _w
with open(os.devnull) as _fh:
    NAME = _fh.name


class C:
    import json

    def m(self):
        import shutil
        return shutil.which("git")


def f():
    import glob
    return glob.glob("*")
"""


def selftest() -> int:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "bad.py").write_text(_BAD, encoding="utf-8")
        (d / "good.py").write_text(_GOOD, encoding="utf-8")

        bad = analyse(d / "bad.py")
        assert [n for _l, n in bad] == ["sys", "sys"], bad

        # The negative case is the half that matters: a gate that only ever
        # says PASS proves nothing, and a gate that cries wolf gets disabled.
        good = analyse(d / "good.py")
        assert good == [], f"false positives: {good}"

    print("check_module_scope selftest PASS: catches a module-level name that "
          "was only imported inside a function; does not flag `except ... as`, "
          "a def nested in a try, lambda parameters, class-body imports, "
          "for targets or with-as bindings")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", type=Path, default=_ROOT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    paths = a.files or targets(a.root)
    if not paths:
        print(f"FATAL: nothing to scan under {a.root} - a gate that scans "
              f"zero files is not a gate")
        return 1
    return check(paths)


if __name__ == "__main__":
    sys.exit(main())
