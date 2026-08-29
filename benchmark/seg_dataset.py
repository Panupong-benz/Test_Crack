# -*- coding: utf-8 -*-
"""Adapter: TiledCOCODataset (SAM3 Datapoint) -> (image, binary mask) pairs.

Fairness core of the benchmark (docs/benchmark_protocol.md SS3): every row
A1-A6 consumes the SAME tiling / augmentation / normalization / sampling as
SAM3-LoRA, by importing the production TiledCOCODataset + the
WeightedDistributedSampler unchanged and only re-shaping the output.

Requires the sam3 package importable (tile_dataset imports it at module
level) - the benchmark runs in ONE environment for all rows by design.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# tile_dataset.py lives in one of two layouts (byte-identical modulo CRLF,
# verified 2026-08-28):
#   local  : <thesis>/04_models/Result_Coding/22.4.2025/  (canonical, CLAUDE.md SS4)
#   vast.ai: Test_Crack repo root (this benchmark/ dir sits inside the clone)
_CANDIDATES = [
    Path(__file__).resolve().parents[1],                                  # repo root (Test_Crack clone)
    Path(__file__).resolve().parents[2] / "Result_Coding" / "22.4.2025",  # local thesis tree
]
for _c in _CANDIDATES:
    if (_c / "tile_dataset.py").exists():
        if str(_c) not in sys.path:
            sys.path.insert(0, str(_c))
        break
else:
    raise ImportError(f"tile_dataset.py not found in any of: {_CANDIDATES}")

from tile_dataset import (  # noqa: E402
    TiledCOCODataset,
    WeightedDistributedSampler,
)


class SegTileAdapter(Dataset):
    """Wraps TiledCOCODataset; unions per-object segments into one binary mask."""

    def __init__(self, data_dir: str, split: str, tile_size: int = 1008,
                 overlap: float = 0.25, augment: bool = True,
                 random_offset: bool = True, compute_tile_stats: bool = False):
        self.base = TiledCOCODataset(
            data_dir=data_dir, split=split, tile_size=tile_size,
            overlap=overlap, min_crack_pixels=0,
            random_offset=random_offset, augment=augment,
            compute_tile_stats=compute_tile_stats,
        )
        self.tile_size = tile_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        dp = self.base[idx]
        img = dp.images[0]
        mask = torch.zeros((self.tile_size, self.tile_size), dtype=torch.float32)
        for obj in img.objects:
            mask = torch.maximum(mask, obj.segment.to(torch.float32))
        return {"image": img.data, "mask": mask.unsqueeze(0),
                "tile_idx": idx}


def _worker_init(seed: int):
    """Per-worker RNG seeding. Without this every forked worker inherits ONE
    numpy state, and tile_dataset's augmentation draws (flip / rotate /
    jitter) come from global np.random - so all workers emit correlated
    transforms and the "augmented" batch is far less diverse than it looks.
    Also makes a seeded run actually reproducible (Amendment A1.4)."""
    def _init(worker_id: int):
        import random as _r
        s = (seed * 100003 + worker_id) % (2 ** 31 - 1)
        np.random.seed(s)
        _r.seed(s)
        torch.manual_seed(s)
    return _init


def build_loaders(data_dir: str, batch_size: int, tile_size: int = 1008,
                  overlap: float = 0.25, num_workers: int = 4, seed: int = 0):
    """Train loader with the production weighted sampler; eval loaders plain."""
    train_ds = SegTileAdapter(data_dir, "train", tile_size, overlap,
                              augment=True, random_offset=True,
                              compute_tile_stats=True)
    weights = train_ds.base.compute_tile_weights()
    sampler = WeightedDistributedSampler(
        weights=weights, num_samples=len(train_ds),
        num_replicas=1, rank=0, seed=seed,
    )
    train = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                       num_workers=num_workers, pin_memory=True,
                       drop_last=True, persistent_workers=num_workers > 0,
                       prefetch_factor=4 if num_workers > 0 else None,
                       worker_init_fn=_worker_init(seed))

    def eval_loader(split):
        ds = SegTileAdapter(data_dir, split, tile_size, overlap,
                            augment=False, random_offset=False)
        return DataLoader(ds, batch_size=max(1, batch_size // 2),
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True,
                          persistent_workers=num_workers > 0)

    return train, eval_loader("valid"), eval_loader("test"), sampler
