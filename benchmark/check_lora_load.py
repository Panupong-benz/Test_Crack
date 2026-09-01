# -*- coding: utf-8 -*-
"""check_lora_load - prove load_lora_weights cannot load NOTHING in silence
(Amendment A1.24 item 135).

The old body was:

    lora_state_dict = torch.load(load_path)
    model.load_state_dict(lora_state_dict, strict=False)   # result discarded
    print(f"Loaded LoRA weights from {load_path}")

so a checkpoint whose key names did not match the model loaded ZERO tensors,
printed "Loaded LoRA weights", and inference then ran the plain base network
while its output was labelled as the fine-tuned row. On this benchmark that
would have made row A6 identical to row A5 with no error anywhere - the
a5_vs_a6 ablation, which is the whole point of the interim rental, silently
reporting no effect.

Four planted cases, torch only (no sam3, no GPU):
  1. every key matches      -> no raise, n_matched == len(ckpt), and the
                               parameters really do take the saved values
  2. every key renamed      -> RuntimeError (this is the case that used to be
                               silent)
  3. one of two matches     -> no raise, n_matched == 1, warning printed
  4. case 2 with strict_lora=False -> no raise (the documented escape hatch)

Run it after touching lora_layers.py; it is also wired into setup step F.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from lora_layers import load_lora_weights  # noqa: E402


class Tiny(nn.Module):
    """Two LoRA-shaped parameters under a submodule, as apply_lora_to_model
    produces them (<module>.lora_A / <module>.lora_B)."""

    def __init__(self):
        super().__init__()
        self.block = nn.Linear(4, 4, bias=False)
        self.block.lora_A = nn.Parameter(torch.zeros(2, 4))
        self.block.lora_B = nn.Parameter(torch.zeros(4, 2))


def _save(sd, d, name):
    p = Path(d) / name
    torch.save(sd, p)
    return str(p)


def main() -> int:
    good = {"block.lora_A": torch.randn(2, 4),
            "block.lora_B": torch.randn(4, 2)}
    renamed = {"wrapped." + k: v for k, v in good.items()}
    half = {"block.lora_A": good["block.lora_A"],
            "wrapped.block.lora_B": good["block.lora_B"]}

    with tempfile.TemporaryDirectory() as d:
        # 1. all keys match
        m = Tiny()
        n, unexpected = load_lora_weights(m, _save(good, d, "good.pt"))
        assert (n, unexpected) == (2, []), (n, unexpected)
        assert torch.equal(m.block.lora_A.data, good["block.lora_A"]), \
            "reported a match but the parameter did not change"
        assert torch.equal(m.block.lora_B.data, good["block.lora_B"])

        # 2. no key matches - the silent case
        m = Tiny()
        try:
            load_lora_weights(m, _save(renamed, d, "renamed.pt"))
        except RuntimeError as e:
            assert "matched NOTHING" in str(e), str(e)
        else:
            raise AssertionError(
                "load_lora_weights accepted a checkpoint that matched nothing "
                "- row A6 would silently be row A5")
        assert torch.equal(m.block.lora_A.data, torch.zeros(2, 4)), \
            "nothing should have been loaded"

        # 3. partial match: allowed, counted, warned
        m = Tiny()
        n, unexpected = load_lora_weights(m, _save(half, d, "half.pt"))
        assert n == 1 and len(unexpected) == 1, (n, unexpected)

        # 4. the escape hatch still works
        m = Tiny()
        n, _ = load_lora_weights(m, _save(renamed, d, "renamed2.pt"),
                                 strict_lora=False)
        assert n == 0, n

    print("check_lora_load PASS: a non-matching LoRA checkpoint now RAISES "
          "(it used to load nothing and print success); matching load is "
          "verified at the parameter level; partial load and strict_lora=False "
          "behave as documented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
