# -*- coding: utf-8 -*-
"""Static gate for `_main_core_loss`, the A1.34 comparable training loss.

WHY THIS FILE EXISTS. `train_sam3_lora_native_claude.py` imports `sam3` at
module scope, so it cannot be imported on the dev box - and by the rule
recorded in 8bz, a file that cannot be imported here MUST have a static
gate, or its code is tested for the first time with GPU money (8bz itself
was a missing `import sys` that died at 0.002 h of a paid rental).

WHAT IT PROTECTS. Until A1.34 the trainer logged `train_loss` and
`val_loss`, which are NOT the same function: Sam3LossWrapper sums core_loss
over every output present (sam3_loss.py:88-100) while the auxiliary-decoder
and one-to-many heads are gated on self.training (sam3_image.py:343-386), so
a training epoch summed 26 terms against validation's 3 - a ~12.6x offset
that also drifts as the aux terms converge. `_main_core_loss` re-reduces
only the MAIN output's terms, which the wrapper stores with an EMPTY suffix
(aux get `_aux_<i>`, one-to-many get a trailing `_o2m`, sam3_loss.py:136),
using each loss fn's own weight_dict - the same reduction as
LossWithWeights.reduce_loss (loss_fns.py:256-264). Get the key convention
wrong and the column still looks plausible while meaning something else.

The training run has a second, stronger check: under model.eval() the same
expression must reproduce core_loss exactly, and the validation loop raises
if it does not. That one needs a GPU. This one needs nothing, so it can run
before the money starts.

HOW IT TESTS THE REAL CODE WITHOUT IMPORTING IT. The method is located with
`ast` (by name, inside the trainer class), its source extracted verbatim,
dedented and exec'd against a stub `torch` and a planted `loss_dict`. So
this gate runs the SHIPPED bytes; it cannot drift from a copy.

Cases: the weighted main-only sum, exclusion of `_aux_*` and `_o2m` keys, a
zero weight, a key the wrapper did not emit, detach() actually being called,
an empty dict returning None - and a NEGATIVE case, a wrong implementation
that sums every key, which must produce a different number. Without that
last one the gate could pass by accident and prove nothing.

  python benchmark/check_main_core_loss.py            # both trees if present
  python benchmark/check_main_core_loss.py --trainer <path>
"""
import argparse
import ast
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CANDIDATES = [
    ROOT / "train_sam3_lora_native_claude.py",                    # Test_Crack
    ROOT.parent / "Result_Coding" / "22.4.2025"                   # canonical
    / "train_sam3_lora_native_claude.py",
    ROOT.parent / "Test_Crack"                                    # deploy
    / "train_sam3_lora_native_claude.py",
]

WRONG_SRC = textwrap.dedent('''
    def _main_core_loss(self, loss_dict):
        """The failure this gate exists to catch: reduce over EVERY key
        rather than the un-suffixed ones, folding aux and o2m back in."""
        total = None
        for fn in self.loss_wrapper.loss_fns_find:
            for k, w in getattr(fn, "weight_dict", {}).items():
                for key, v in loss_dict.items():
                    if not key.startswith(k) or w == 0:
                        continue
                    v = v.detach() if torch.is_tensor(v) else torch.as_tensor(
                        float(v), device=self.device)
                    total = v * w if total is None else total + v * w
        return total
''')


# --------------------------------------------------------------- stub torch
class _T:
    """Minimal stand-in for a scalar tensor. Records detach() so the gate can
    assert the graph is actually dropped - keeping 3000 attached scalars per
    epoch would hold the whole epoch's autograd graph in VRAM."""

    def __init__(self, v, detached=False):
        self.v = float(v)
        self.detached = detached

    def detach(self):
        return _T(self.v, True)

    def float(self):
        return self

    def __mul__(self, o):
        return _T(self.v * float(o), self.detached)

    __rmul__ = __mul__

    def __add__(self, o):
        return _T(self.v + (o.v if isinstance(o, _T) else float(o)),
                  self.detached and getattr(o, "detached", True))


class _Torch:
    @staticmethod
    def is_tensor(x):
        return isinstance(x, _T)

    @staticmethod
    def as_tensor(v, device=None):
        return _T(v)


class _Fn:
    def __init__(self, wd):
        self.weight_dict = wd


class _Wrapper:
    def __init__(self, fns):
        self.loss_fns_find = fns


class _Self:
    def __init__(self, fns):
        self.loss_wrapper = _Wrapper([_Fn(w) for w in fns])
        self.device = "cpu"


def extract_method(path: Path) -> str:
    """Pull `_main_core_loss` out of the trainer by structure, not by line."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_main_core_loss":
            src = ast.get_source_segment(path.read_text(encoding="utf-8"), node)
            if src:
                return textwrap.dedent(src)
    raise SystemExit(f"FATAL: _main_core_loss not found in {path} - the "
                     "A1.34 logging is missing from this trainer copy.")


def compile_fn(src: str):
    ns = {"torch": _Torch}
    exec(compile(src, "<extracted>", "exec"), ns)     # noqa: S102
    return ns["_main_core_loss"]


# The three loss fns the trainer builds, with the weights declared there once
WEIGHTS = [{"loss_bbox": 5.0, "loss_giou": 2.0},
           {"loss_ce": 20.0, "presence_loss": 20.0},
           {"loss_mask": 200.0, "loss_dice": 50.0, "loss_cldice": 30.0}]


def selftest(path: Path) -> int:
    fn = compile_fn(extract_method(path))
    me = _Self(WEIGHTS)

    # main terms, plus exactly the decoys the wrapper really emits
    main = {"loss_bbox": 0.10, "loss_giou": 0.20, "loss_ce": 0.30,
            "presence_loss": 0.40, "loss_mask": 0.50, "loss_dice": 0.60,
            "loss_cldice": 0.70}
    ld = {k: _T(v) for k, v in main.items()}
    for i in range(5):                                  # 5 auxiliary layers
        for k, v in main.items():
            ld[f"{k}_aux_{i}"] = _T(v * 3 + i)
    for k, v in main.items():                           # one-to-many branch
        ld[f"{k}_o2m"] = _T(v * 7)
        for i in range(5):
            ld[f"{k}_aux_{i}_o2m"] = _T(v * 11)
    ld["core_loss"] = _T(999.0)

    want = sum(main[k] * w for wd in WEIGHTS for k, w in wd.items())
    got = fn(me, ld)
    assert abs(got.v - want) < 1e-9, f"main-only sum {got.v} != {want}"
    assert got.detached, "result is still attached to the autograd graph"
    # the decoys are not small: folding them in would be obvious here
    assert abs(got.v - 999.0) > 1, "suspiciously equal to core_loss"

    # a zero weight contributes nothing
    me0 = _Self([{"loss_bbox": 5.0, "loss_giou": 0.0}])
    assert abs(fn(me0, ld).v - 0.10 * 5.0) < 1e-9, "zero weight was summed"

    # a key the wrapper did not emit is skipped, not a KeyError
    me1 = _Self([{"loss_bbox": 5.0, "loss_nonexistent": 3.0}])
    assert abs(fn(me1, ld).v - 0.10 * 5.0) < 1e-9, "missing key not skipped"

    # a plain float still reduces (LossWithWeights can return 0.0)
    ld2 = dict(ld, loss_bbox=0.10)
    assert abs(fn(me1, ld2).v - 0.10 * 5.0) < 1e-9, "float term not handled"

    # nothing to reduce -> None, so the caller can log null rather than 0.0
    assert fn(me, {}) is None, "empty loss_dict did not return None"

    # NEGATIVE: a wrong implementation must NOT reproduce the main-only sum,
    # otherwise this gate would pass whatever it was given
    bad = compile_fn(WRONG_SRC)(me, ld)
    assert abs(bad.v - want) > 1.0, ("the negative case matched the correct "
                                     "answer - this gate proves nothing")

    print(f"check_main_core_loss: PASS  {path.name} "
          f"(main-only {want:.2f} vs all-keys {bad.v:.2f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", type=Path, default=None)
    a = ap.parse_args()

    paths = [a.trainer] if a.trainer else [p for p in CANDIDATES if p.exists()]
    if not paths:
        print("FATAL: no trainer found. This is an ENVIRONMENT/layout fault, "
              "not a code fault; looked in:", file=sys.stderr)
        for c in CANDIDATES:
            print(f"  {c}", file=sys.stderr)
        return 1
    for p in paths:
        selftest(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
