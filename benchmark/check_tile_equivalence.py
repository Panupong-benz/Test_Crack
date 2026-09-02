# -*- coding: utf-8 -*-
"""Prove the tile_dataset bbox pre-filter changes nothing, and measure what
it buys.

Why a gate and not just a code review
------------------------------------
`tile_dataset.py` is PRODUCTION code: it feeds row A6 and the SAM3-LoRA
training the whole thesis rests on. The pre-filter (skip annotations whose
COCO bbox misses the tile before decoding a full-frame RLE mask) is
*argued* to be exact — the crop happens in original-image coordinates before
any flip/rotate, and a non-intersecting annotation can only yield an empty
tile mask, which the existing `ys.size == 0` guard already drops. Arguments
are not evidence, so this script builds the SAME tiles both ways and compares
them byte for byte.

Old behaviour is reachable via the `TILE_DATASET_NO_BBOX_FILTER` env var, so
both paths run from one checkout — no branch juggling.

--mode window (A1.30 item 168) gates the second data-path change the same
way: windowed polygon decode (default) vs full-frame decode
(`TILE_DATASET_FULLFRAME_DECODE=1`). In this mode the sample always includes
the densest tiles by hitting-bbox count — the tiles the optimization exists
for. --selftest-window proves the tricky rasterization math (tile-border
crossing, out-of-frame clamping, RLE passthrough) on a tiny synthetic fold,
laptop-safe.

What is compared, per sampled tile: the image tensor, the number of objects,
and every object's segment mask and bbox tensor. Augmentation and random
tile offsets are DISABLED (`augment=False, random_offset=False`) so the two
runs are deterministic and any difference is attributable to the filter.

Usage:
  python benchmark/check_tile_equivalence.py [--data <dir with train/>] [-n 50]

Exit 1 on any mismatch. Prints the measured per-tile speedup.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for c in (HERE.parent.parent / "Result_Coding" / "22.4.2025", HERE.parent):
    if (c / "tile_dataset.py").exists():
        sys.path.insert(0, str(c))
        break


def _stub_sam3_if_needed():
    """tile_dataset pulls five dataclasses out of the sam3 package, whose
    import chain reaches triton (Linux-only). On the instance the real
    package is there and nothing happens here; off-GPU this substitutes
    equivalent containers so the DATA path can be gated on any machine.
    The stub touches only the container types - every line of tiling,
    decoding and cropping under test is the production code."""
    try:
        import sam3.train.data.sam3_image_dataset  # noqa: F401
        return "real sam3"
    except Exception:                                        # noqa: BLE001
        pass
    import types
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class Object:
        bbox: Any = None
        area: float = 0.0
        object_id: int = 0
        segment: Any = None

    @dataclass
    class Image:
        data: Any = None
        objects: list = field(default_factory=list)
        size: tuple = (0, 0)

    class _Loose:
        """Accepts whatever kwargs the production code passes - the real
        dataclasses carry more fields than this gate needs to compare."""
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Datapoint(_Loose):
        pass

    class FindQueryLoaded(_Loose):
        pass

    class InferenceMetadata(_Loose):
        pass

    mod = types.ModuleType("sam3.train.data.sam3_image_dataset")
    for k, v in dict(Object=Object, Image=Image, Datapoint=Datapoint,
                     FindQueryLoaded=FindQueryLoaded,
                     InferenceMetadata=InferenceMetadata).items():
        setattr(mod, k, v)
    for name in ("sam3", "sam3.train", "sam3.train.data"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sam3.train.data.sam3_image_dataset"] = mod
    return "stubbed sam3 containers (data path is production code)"


def build(data_dir: Path, split: str, legacy: bool, mode: str = "bbox"):
    # the flags are read inside __getitem__, so they must be set before
    # iterating. Exactly one axis is toggled per mode; the other is left at
    # its default so each gate isolates one change.
    os.environ.pop("TILE_DATASET_NO_BBOX_FILTER", None)
    os.environ.pop("TILE_DATASET_FULLFRAME_DECODE", None)
    if legacy:
        var = ("TILE_DATASET_NO_BBOX_FILTER" if mode == "bbox"
               else "TILE_DATASET_FULLFRAME_DECODE")
        os.environ[var] = "1"
    import importlib
    import tile_dataset
    importlib.reload(tile_dataset)
    return tile_dataset.TiledCOCODataset(
        data_dir=str(data_dir), split=split, tile_size=1008, overlap=0.25,
        augment=False, random_offset=False, min_crack_pixels=0)


def fetch(ds, idxs):
    out, t0 = [], time.time()
    for i in idxs:
        out.append(ds[i])
    return out, time.time() - t0


def _image_of(dp):
    if hasattr(dp, "images") and dp.images:
        return dp.images[0]
    return dp.image if hasattr(dp, "image") else dp


def same(a, b) -> str | None:
    """Return a description of the first difference, or None."""
    ia, ib = _image_of(a), _image_of(b)
    ta = ia.data if hasattr(ia, "data") else ia
    tb = ib.data if hasattr(ib, "data") else ib
    if isinstance(ta, torch.Tensor) and not torch.equal(ta, tb):
        return "image tensor differs"
    oa = ia.objects if hasattr(ia, "objects") else []
    ob = ib.objects if hasattr(ib, "objects") else []
    if len(oa) != len(ob):
        return f"object count {len(oa)} vs {len(ob)}"
    for k, (x, y) in enumerate(zip(oa, ob)):
        if not torch.equal(x.segment, y.segment):
            return f"object {k} segment differs"
        if not torch.equal(x.bbox, y.bbox):
            return f"object {k} bbox {x.bbox.tolist()} vs {y.bbox.tolist()}"
    return None


def selftest_window() -> int:
    """Synthetic fold exercising exactly the cases where windowed decode
    could diverge from full-frame-then-crop: a polygon crossing a tile
    border, one with coordinates outside the frame (rleFrPoly clamps), a
    sub-pixel sliver, a multi-part polygon, and an RLE dict (passthrough).
    Every tile is compared on both paths."""
    import json
    import tempfile

    import cv2
    from pycocotools import mask as mask_utils

    print(f"[env] {_stub_sam3_if_needed()}")
    with tempfile.TemporaryDirectory() as td:
        fold = Path(td)
        tr = fold / "train"
        tr.mkdir()
        w, h = 1300, 1100                       # 2x2 tiles at ts=1008
        img = (np.random.default_rng(0)
               .integers(0, 255, (h, w, 3), dtype=np.uint8))
        cv2.imwrite(str(tr / "syn.jpg"), img)

        rle = mask_utils.encode(np.asfortranarray(
            (np.arange(h)[:, None] % 7 == 0).astype(np.uint8)
            * (np.arange(w)[None, :] % 5 == 0).astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        anns = [
            # crosses the vertical tile border at x=292 (2nd tile origin)
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [200, 50, 300, 40], "area": 1,
             "segmentation": [[200, 50, 500, 55, 498, 90, 205, 88]],
             "iscrowd": 0},
            # vertices OUTSIDE the frame -> rleFrPoly clamping on both paths
            {"id": 2, "image_id": 1, "category_id": 1,
             "bbox": [0, 900, 400, 199], "area": 1,
             "segmentation": [[-50, 950, 380, 940, 390, 1150, -60, 1160]],
             "iscrowd": 0},
            # sub-pixel sliver with fractional coords near a tile corner
            {"id": 3, "image_id": 1, "category_id": 1,
             "bbox": [290, 90, 6, 6], "area": 1,
             "segmentation": [[291.3, 90.7, 295.9, 91.2, 293.1, 95.8]],
             "iscrowd": 0},
            # multi-part polygon spanning both tile rows
            {"id": 4, "image_id": 1, "category_id": 1,
             "bbox": [100, 60, 900, 400], "area": 1,
             "segmentation": [[100, 60, 990, 70, 985, 120, 105, 110],
                              [120, 380, 940, 390, 935, 450, 125, 440]],
             "iscrowd": 0},
            # RLE dict: must take the unchanged full-frame path
            {"id": 5, "image_id": 1, "category_id": 1,
             "bbox": [0, 0, w, h], "area": 1,
             "segmentation": rle, "iscrowd": 0},
        ]
        coco = {"images": [{"id": 1, "file_name": "syn.jpg",
                            "width": w, "height": h}],
                "annotations": anns,
                "categories": [{"id": 1, "name": "crack"}]}
        (tr / "_annotations.coco.json").write_text(json.dumps(coco),
                                                   encoding="utf-8")

        ds_new = build(fold, "train", legacy=False, mode="window")
        idxs = list(range(len(ds_new)))
        new, _ = fetch(ds_new, idxs)
        ds_old = build(fold, "train", legacy=True, mode="window")
        old, _ = fetch(ds_old, idxs)
        for i, (x, y) in zip(idxs, zip(new, old)):
            d = same(x, y)
            assert d is None, f"tile {i}: {d}"
    print(f"selftest-window PASS: {len(idxs)} synthetic tiles identical on "
          f"both decode paths (border-crossing, out-of-frame clamp, "
          f"sliver, multi-part, RLE passthrough)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=None,
                    help="dir holding <split>/_annotations.coco.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("-n", type=int, default=50, help="tiles to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("bbox", "window"), default="bbox",
                    help="bbox = pre-filter gate (A1.4); window = windowed "
                         "polygon decode gate (A1.30 item 168)")
    ap.add_argument("--selftest-window", action="store_true",
                    help="synthetic fixture for the window math; no real "
                         "data, laptop-safe")
    a = ap.parse_args()

    if a.selftest_window:
        return selftest_window()

    print(f"[env] {_stub_sam3_if_needed()}")
    data = a.data
    if data is None:                      # local default: the frozen pool
        cand = (HERE.parent.parent.parent / "03_annotation" / "_pool" /
                "POOL_BM")
        data = cand
        a.split = "train"
    if not (Path(data) / a.split / "_annotations.coco.json").exists():
        print(f"no COCO at {data}/{a.split} — pass --data")
        return 2

    ds_new = build(Path(data), a.split, legacy=False, mode=a.mode)
    n = len(ds_new)
    rng = np.random.default_rng(a.seed)
    idxs = set(rng.choice(n, size=min(a.n, n), replace=False).tolist())
    if a.mode == "window":
        # always include the densest tiles - the ones the windowed decode
        # exists for; a random sample of a heavy-tailed distribution would
        # usually miss them entirely
        hits = []
        for img_id, x0, y0 in ds_new.tile_specs:
            anns = ds_new.img_to_anns.get(img_id, [])
            hits.append(sum(1 for t in anns
                            if ds_new._bbox_hits_tile(t, x0, y0)))
        dense = sorted(range(n), key=lambda i: hits[i], reverse=True)[:10]
        idxs |= set(dense)
        print(f"[window] densest tiles added: "
              f"{[(i, hits[i]) for i in dense[:5]]} ...")
    idxs = sorted(idxs)
    print(f"{n} tiles in {data}/{a.split}; comparing {len(idxs)} "
          f"(mode {a.mode})")

    new, t_new = fetch(ds_new, idxs)
    ds_old = build(Path(data), a.split, legacy=True, mode=a.mode)
    old, t_old = fetch(ds_old, idxs)

    bad = []
    for i, (x, y) in zip(idxs, zip(new, old)):
        d = same(x, y)
        if d:
            bad.append((i, d))

    per_new, per_old = t_new / len(idxs), t_old / len(idxs)
    print(f"old path {per_old * 1000:7.1f} ms/tile")
    print(f"new path {per_new * 1000:7.1f} ms/tile   "
          f"speedup {per_old / max(per_new, 1e-9):.2f}x")
    if bad:
        print(f"\nFAIL: {len(bad)} tile(s) differ")
        for i, d in bad[:10]:
            print(f"  tile {i}: {d}")
        return 1
    print(f"\nPASS: {len(idxs)} tiles identical (image, object count, "
          f"every segment and bbox)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
