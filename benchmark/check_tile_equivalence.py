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


def build(data_dir: Path, split: str, no_filter: bool):
    # the flag is read inside __getitem__, so it must be set before iterating
    if no_filter:
        os.environ["TILE_DATASET_NO_BBOX_FILTER"] = "1"
    else:
        os.environ.pop("TILE_DATASET_NO_BBOX_FILTER", None)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=None,
                    help="dir holding <split>/_annotations.coco.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("-n", type=int, default=50, help="tiles to sample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

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

    ds_new = build(Path(data), a.split, no_filter=False)
    n = len(ds_new)
    rng = np.random.default_rng(a.seed)
    idxs = sorted(rng.choice(n, size=min(a.n, n), replace=False).tolist())
    print(f"{n} tiles in {data}/{a.split}; comparing {len(idxs)}")

    new, t_new = fetch(ds_new, idxs)
    ds_old = build(Path(data), a.split, no_filter=True)
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
