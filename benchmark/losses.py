# -*- coding: utf-8 -*-
"""Per-pixel composite loss paralleling the SAM3-LoRA mask objective.

SAM3 rows train with 200*focal(a=0.85, g=3) + 50*dice + 30*clDice inside the
DETR set loss (train config, cldice_loss.MasksWithCLDice). Semantic-seg rows
(A2-A4) get the same three terms at the same weights, per-pixel — declared in
benchmark_protocol.md SS3 as the chosen parity. soft_skeletonize is imported
from the production cldice_loss so both families share one skeleton
definition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# cldice_loss.py: same dual layout as seg_dataset.py (repo root on a
# Test_Crack clone; Result_Coding/22.4.2025 in the local thesis tree)
_CANDIDATES = [
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2] / "Result_Coding" / "22.4.2025",
]
for _c in _CANDIDATES:
    if (_c / "cldice_loss.py").exists():
        if str(_c) not in sys.path:
            sys.path.insert(0, str(_c))
        break
else:
    raise ImportError(f"cldice_loss.py not found in any of: {_CANDIDATES}")

from cldice_loss import soft_skeletonize, _maybe_downsample  # noqa: E402

W_FOCAL, W_DICE, W_CLDICE = 200.0, 50.0, 30.0
FOCAL_ALPHA, FOCAL_GAMMA = 0.85, 3.0
CLDICE_ITERS, CLDICE_MAX_SIZE, SMOOTH = 3, 384, 1.0


def focal_loss(logits, target):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = FOCAL_ALPHA * target + (1 - FOCAL_ALPHA) * (1 - target)
    return (a_t * (1 - p_t) ** FOCAL_GAMMA * ce).mean()


def dice_loss(logits, target):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + SMOOTH) / (p.sum(1) + t.sum(1) + SMOOTH)).mean()


def cldice_loss(logits, target):
    p = torch.sigmoid(logits)
    p = _maybe_downsample(p, CLDICE_MAX_SIZE)
    t = _maybe_downsample(target, CLDICE_MAX_SIZE)
    with torch.no_grad():
        skel_t = soft_skeletonize(t, iters=CLDICE_ITERS)
    skel_p = soft_skeletonize(p, iters=CLDICE_ITERS)
    tprec = ((skel_p * t).sum((1, 2, 3)) + 1e-8) / (skel_p.sum((1, 2, 3)) + 1e-8)
    tsens = ((skel_t * p).sum((1, 2, 3)) + 1e-8) / (skel_t.sum((1, 2, 3)) + 1e-8)
    return (1 - 2 * tprec * tsens / (tprec + tsens + 1e-8)).mean()


def composite_loss(logits, target):
    lf = focal_loss(logits, target)
    ld = dice_loss(logits, target)
    lc = cldice_loss(logits, target)
    total = W_FOCAL * lf + W_DICE * ld + W_CLDICE * lc
    return total, {"focal": lf.item(), "dice": ld.item(), "cldice": lc.item(),
                   "total": total.item()}
