# -*- coding: utf-8 -*-
"""Exact per-epoch step counts + dense-tile profile for a fold (A1.30).

Why this exists (Amendment A1.30 items 163/165/167): the smoke hour used to
extrapolate tiles-per-image from 8 randomly sampled images, while POOL_BM
mixes Roboflow-resized frames (1256x2404 -> 2x3 = 6 tiles) with full-res
frames (5184x3456 -> 7x5 = 35 tiles). Whichever 8 images the smoke drew
decided the whole rental's hour estimate - it under-read fold_RW20 by 2x
(1,491 predicted vs 3,001 real steps/epoch), which combined with the old
it/s misread (A1.27) is how a normal 55-min epoch was experienced as a hang.

This script COUNTS instead of extrapolating, using the REAL TiledCOCODataset
(never a reimplementation - the tiling grid, the min_crack_pixels filter and
the bbox-hit test are the production methods, imported via a sam3 stub the
same way resume_state.py is tested off-box, A1.22). The constructor never
opens an image file, so counting needs only the fold's COCO json.

Modes
  (default)        exact tiles -> steps/epoch/rank -> hours at a given s/it
  --hist-bbox      per-tile hitting-annotation histogram = the H1 structural
                   test (dense-tile head-of-line block, item 163)
  --time-worst N   time real __getitem__ on the N densest tiles vs N median
                   tiles (needs the image files -> run on the rented box)
  --selftest       synthetic fixture, no real data touched

Machine policy (item 164): the laptop runs only --selftest; every measurement
mode runs on the rented box. The script is CPU-only by construction - no
model is built, nothing calls .cuda() - and it ASSERTS at exit that
torch.cuda was never initialised, so an accidental GPU touch is a failure,
not a surprise.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path

# ---------------------------------------------------------------- imports --
# tile_dataset.py: repo root on a Test_Crack clone, Result_Coding/22.4.2025
# in the local thesis tree (byte-identical, the run_a5_zeroshot idiom).
_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATES = (
    _ROOT / "tile_dataset.py",
    _ROOT.parent / "Result_Coding" / "22.4.2025" / "tile_dataset.py",
)
for _TD in _CANDIDATES:
    if _TD.exists():
        break
else:
    raise FileNotFoundError(f"tile_dataset.py not found near {_ROOT}")


def _stub_sam3():
    """Register just enough of sam3 for `import tile_dataset` to succeed.

    tile_dataset imports 6 names from sam3.train.data.sam3_image_dataset and
    uses them only to BUILD return values in __getitem__. Counting and the
    bbox histogram never construct them; --time-worst constructs Datapoint/
    Image/Object as plain containers, which is all the timing needs. The
    real sam3 (on the rented box) wins if it is importable - the stub is
    registered only when the import fails, so on-box numbers go through the
    genuine classes.
    """
    try:
        import sam3.train.data.sam3_image_dataset  # noqa: F401
        return
    except Exception:
        pass

    def _container(name):
        def __init__(self, **kw):
            self.__dict__.update(kw)
        return type(name, (), {"__init__": __init__})

    mod = types.ModuleType("sam3.train.data.sam3_image_dataset")
    for name in ("Datapoint", "FindQueryLoaded", "Image",
                 "InferenceMetadata", "Object"):
        setattr(mod, name, _container(name))
    pkg_names = ("sam3", "sam3.train", "sam3.train.data")
    for p in pkg_names:
        sys.modules.setdefault(p, types.ModuleType(p))
    sys.modules["sam3.train.data.sam3_image_dataset"] = mod


def load_dataset_class():
    _stub_sam3()
    sys.path.insert(0, str(_TD.parent))
    import tile_dataset  # noqa: PLC0415
    return tile_dataset.TiledCOCODataset


# ----------------------------------------------------------------- config --
def tiling_cfg(config_path: Path) -> dict:
    """training.tiling of the run config - the SAME keys the trainer reads
    (train_sam3_lora_native_claude.py _make_dataset), so the count is the
    training count by construction."""
    import yaml  # noqa: PLC0415
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return (cfg.get("training", {}) or {}).get("tiling", {}) or {}


def cfg_batch_epochs(config_path: Path) -> tuple:
    """(batch_size, num_epochs) from the run config - one source of truth
    with the trainer, so the smoke's projection cannot drift from what the
    queue actually trains."""
    import yaml  # noqa: PLC0415
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tr = cfg.get("training", {}) or {}
    return int(tr.get("batch_size", 2)), int(tr.get("num_epochs", 30))


def build(fold: Path, tcfg: dict, split: str = "train", stats: bool = False,
          for_timing: bool = False):
    TiledCOCODataset = load_dataset_class()
    mcp = int(tcfg.get("min_crack_pixels", 0))
    if mcp > 0 and not stats:
        print(f"WARN: min_crack_pixels={mcp} > 0 - the exact training count "
              f"needs the mask-decode path; forcing compute_tile_stats=True "
              f"(minutes of CPU, run this on the box).")
        stats = True
    return TiledCOCODataset(
        data_dir=str(fold), split=split,
        tile_size=int(tcfg.get("tile_size", 1008)),
        overlap=float(tcfg.get("overlap", 0.25)),
        min_crack_pixels=mcp,
        # timing mode mirrors training-time augmentation cost; counting
        # neither needs nor wants randomness
        random_offset=for_timing, augment=for_timing,
        image_cache_size=int(tcfg.get("image_cache_size", 8)),
        compute_tile_stats=stats,
    )


# ------------------------------------------------------------------ maths --
def steps_per_rank(tiles: int, batch: int, world: int) -> int:
    """Mirrors the run: WeightedDistributedSampler shards
    (tiles // world) per rank, DataLoader default drop_last=False ->
    ceil(per_rank / batch). world=1 -> 3001 and world=2 -> 1501 on
    fold_RW20's 6002 tiles, matching the observed tqdm denominators."""
    return math.ceil((tiles // world) / batch)


def report_counts(ds, batch: int, world: int, epochs: int, s_it: float):
    tiles = len(ds.tile_specs)
    per_img = {}
    for img_id, _, _ in ds.tile_specs:
        per_img[img_id] = per_img.get(img_id, 0) + 1
    counts = sorted(per_img.values())
    spr = steps_per_rank(tiles, batch, world)
    hours = spr * epochs * s_it / 3600.0
    print(f"\n[count] tiles total          : {tiles}")
    print(f"[count] images               : {len(ds.images)} "
          f"({len(per_img)} contribute tiles)")
    if counts:
        print(f"[count] tiles/image          : min {counts[0]} / median "
              f"{statistics.median(counts):g} / max {counts[-1]}")
    print(f"[count] steps/epoch/rank     : {spr}   "
          f"(batch {batch} x world {world}, drop_last=False)")
    print(f"[count] {epochs} epochs @ {s_it} s/it -> {hours:.1f} h/rank "
          f"(training only, + eval/infer overhead)")
    return tiles, spr, hours


def hist_bbox(ds, top: int = 10):
    """Hitting-annotation count per tile via the REAL _bbox_hits_tile -
    the H1 structural test."""
    hits = []
    for img_id, x0, y0 in ds.tile_specs:
        anns = ds.img_to_anns.get(img_id, [])
        hits.append(sum(1 for a in anns if ds._bbox_hits_tile(a, x0, y0)))
    hs = sorted(hits)
    n = len(hs)

    def q(p):
        return hs[min(n - 1, int(p * n))] if n else 0

    print(f"\n[hist-bbox] tiles            : {n}")
    print(f"[hist-bbox] hitting anns/tile: min {hs[0] if n else 0} / median "
          f"{q(0.5)} / p90 {q(0.90)} / p99 {q(0.99)} / max {hs[-1] if n else 0}")
    order = sorted(range(n), key=lambda i: hits[i], reverse=True)
    print(f"[hist-bbox] top {top} densest tiles (img_id, x0, y0, n_anns):")
    for i in order[:top]:
        img_id, x0, y0 = ds.tile_specs[i]
        fn = ds.images[img_id].get("file_name", "?")
        print(f"    {fn[:48]:48s} ({x0:5d},{y0:5d})  {hits[i]}")
    if n and hs[-1] >= 10 * max(1, q(0.5)):
        print("[hist-bbox] VERDICT: heavy tail present (max >= 10x median) "
              "- H1 structurally confirmed")
    else:
        print("[hist-bbox] VERDICT: no heavy tail - H1 not supported "
              "structurally")
    return hits, order


def time_worst(ds, hits, order, n: int):
    """Time real __getitem__ on the densest vs median tiles. Needs images."""
    med_start = max(0, len(order) // 2 - n // 2)
    groups = [("densest", order[:n]),
              ("median ", order[med_start:med_start + n])]
    out = {}
    for label, idxs in groups:
        ts = []
        for i in idxs:
            t0 = time.perf_counter()
            ds[i]
            ts.append(time.perf_counter() - t0)
            print(f"[time-worst] {label} tile anns={hits[i]:4d}  "
                  f"{ts[-1]:7.2f} s")
        out[label.strip()] = ts
        print(f"[time-worst] {label}: mean {statistics.mean(ts):.2f} s / "
              f"max {max(ts):.2f} s over {len(ts)} tiles")
    return out


# --------------------------------------------------------------- selftest --
def selftest() -> int:
    import tempfile

    import numpy as np  # noqa: F401  (tile_dataset needs it anyway)

    with tempfile.TemporaryDirectory() as td:
        fold = Path(td)
        (fold / "train").mkdir()
        # two frames pinning the bimodal-size fact of item 163:
        # 1256x2404 -> 2x3 = 6 tiles, 5184x3456 -> 7x5 = 35 tiles
        imgs = [{"id": 1, "file_name": "small.jpg",
                 "width": 1256, "height": 2404},
                {"id": 2, "file_name": "big.jpg",
                 "width": 5184, "height": 3456}]
        anns = [
            # a polygon inside the big frame's top-left tile only
            {"id": 1, "image_id": 2, "category_id": 1,
             "bbox": [10, 10, 100, 100],
             "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
             "area": 10000, "iscrowd": 0},
            # a wide polygon crossing the whole top tile row (y 100-130 sits
            # inside the y0=0 band only), overlapping ann 1's tile at (0,0)
            {"id": 2, "image_id": 2, "category_id": 1,
             "bbox": [0, 100, 5000, 30],
             "segmentation": [[0, 100, 5000, 100, 5000, 130, 0, 130]],
             "area": 150000, "iscrowd": 0},
        ]
        coco = {"images": imgs, "annotations": anns,
                "categories": [{"id": 1, "name": "crack"}]}
        (fold / "train" / "_annotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8")

        tcfg = {"tile_size": 1008, "overlap": 0.25, "min_crack_pixels": 0,
                "image_cache_size": 2}
        ds = build(fold, tcfg)
        tiles = len(ds.tile_specs)
        assert tiles == 6 + 35, f"grid count {tiles} != 41"
        per = {}
        for img_id, _, _ in ds.tile_specs:
            per[img_id] = per.get(img_id, 0) + 1
        assert per == {1: 6, 2: 35}, per

        # steps formula pins the observed denominators (scaled fixture)
        assert steps_per_rank(6002, 2, 1) == 3001
        assert steps_per_rank(6002, 2, 2) == 1501
        assert steps_per_rank(41, 2, 1) == 21

        # bbox histogram through the real _bbox_hits_tile: ann 1 hits only
        # the (0,0) tile of image 2; ann 2 hits every tile of the row band
        hits, order = hist_bbox(ds, top=3)
        big00 = next(i for i, (im, x, y) in enumerate(ds.tile_specs)
                     if im == 2 and x == 0 and y == 0)
        assert hits[big00] == 2, hits[big00]
        assert all(hits[i] == 0 for i, (im, _, _) in
                   enumerate(ds.tile_specs) if im == 1)
        assert hits[order[0]] == max(hits)

    import torch  # noqa: PLC0415
    assert not torch.cuda.is_initialized(), \
        "count_tiles touched CUDA - it must be CPU-only (A1.30 item 165)"
    print("count_tiles selftest PASS (grid 6+35, steps 3001/1501, "
          "bbox-hist via real _bbox_hits_tile, CUDA untouched)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fold", type=Path,
                    help="fold dir holding <split>/_annotations.coco.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--config", type=Path,
                    default=_ROOT / "configs" / "full_lora_config.yaml",
                    help="run config; tiling params are read from it so the "
                         "count is the training count by construction")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--s-it", type=float, default=1.11,
                    help="seconds per micro-step (from the smoke)")
    ap.add_argument("--hist-bbox", action="store_true")
    ap.add_argument("--time-worst", type=int, default=0, metavar="N",
                    help="time __getitem__ on the N densest + N median tiles "
                         "(needs image files - run on the rented box)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.fold:
        ap.error("--fold is required (or --selftest)")

    tcfg = tiling_cfg(a.config) if a.config.exists() else {}
    if a.config.exists():
        _b, _e = cfg_batch_epochs(a.config)
        # config-derived defaults; explicit flags win
        if "--batch" not in sys.argv:
            a.batch = _b
        if "--epochs" not in sys.argv:
            a.epochs = _e
    ds = build(a.fold, tcfg, split=a.split, for_timing=bool(a.time_worst))
    report_counts(ds, a.batch, a.world, a.epochs, a.s_it)

    if a.hist_bbox or a.time_worst:
        hits, order = hist_bbox(ds)
        if a.time_worst:
            time_worst(ds, hits, order, a.time_worst)

    import torch  # noqa: PLC0415
    assert not torch.cuda.is_initialized(), \
        "count_tiles touched CUDA - it must be CPU-only (A1.30 item 165)"
    return 0


if __name__ == "__main__":
    sys.exit(main())
