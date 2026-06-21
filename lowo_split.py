#!/usr/bin/env python3
"""
lowo_split.py — Leave-One-Wall-Out (LOWO) cross-validation splitter for the
SAM3-LoRA crack pipeline.

Produces, for each held-out wall W, a self-contained COCO fold:

    <out>/fold_<W>/
        train/ <images...> _annotations.coco.json
        valid/ <images...> _annotations.coco.json
        test/  <images...> _annotations.coco.json

which plugs directly into train_sam3_lora_native_claude.py
(point its config `data_dir` at <out>/fold_<W>).

LOWO rules implemented
----------------------
- Each fold holds out ALL images of exactly one wall as `test`.
- The held-out wall NEVER appears in that fold's train/valid -> no wall-level
  leakage (the thing that inflates mAP if you random-split).
- Walls in --exclude-test (default: RW40) are never held out; they stay in the
  training pool of EVERY fold. This is the "RW40 excluded from crack-grounding
  test folds" requirement.
- `valid` is carved from the training pool (wall-stratified, deterministic) for
  early-stopping / model selection only. It may share walls with train by
  design; it is always disjoint from the held-out test wall.

Wall identity is parsed from the filename. The wall list is matched
longest-token-first with non-alnum boundaries so RW20 does not swallow
RW20C / RW20L / RW20T.

Source can be either:
  (a) a flat COCO dir:  <src>/_annotations.coco.json  + images, or
  (b) a split dir:      <src>/{train,valid,test}/_annotations.coco.json + images
In case (b) all splits are merged into one pool, then re-partitioned by wall.

Image ids and annotation ids are re-indexed per output split so every
_annotations.coco.json is internally consistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

# ---- optional PIL, only used to backfill missing height/width -------------
try:
    from PIL import Image as _PILImage
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


DEFAULT_WALLS = ["RW40", "RW20", "RW20C", "RW20L", "RW20T"]
DEFAULT_EXCLUDE_TEST = ["RW40"]
ANN_FILE = "_annotations.coco.json"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# Wall identification
# --------------------------------------------------------------------------- #
def build_wall_regex(walls):
    """Longest token first + non-alnum boundaries so RW20 != RW20C/L/T."""
    ordered = sorted(set(walls), key=len, reverse=True)
    alt = "|".join(re.escape(w) for w in ordered)
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alt})(?![A-Za-z0-9])", re.IGNORECASE)


def wall_of(filename, wall_rx, canon):
    """Return canonical wall id for a filename, or None if no match."""
    m = wall_rx.search(filename)
    if not m:
        return None
    return canon[m.group(0).upper()]


# --------------------------------------------------------------------------- #
# Source loading
# --------------------------------------------------------------------------- #
def _read_coco(ann_path):
    with open(ann_path, "r") as f:
        return json.load(f)


def _resolve_image_path(img_dir, file_name):
    """file_name in COCO is usually a basename living next to the json.
    Fall back to a basename search under img_dir."""
    cand = img_dir / file_name
    if cand.exists():
        return cand
    cand2 = img_dir / Path(file_name).name
    if cand2.exists():
        return cand2
    # last resort: scan (handles nested image dirs)
    base = Path(file_name).name
    for p in img_dir.rglob(base):
        return p
    return None


def load_pool(src: Path):
    """Merge all available COCO data under src into one list of records.

    Each record: {
        "wall": None,                 # filled later
        "src_split": "train"/"flat",
        "img_dir": Path,              # where the image file lives
        "file_name": str,             # original basename
        "height": int, "width": int,
        "anns": [ {category_id,bbox,area,segmentation,iscrowd}, ... ],
    }
    Returns (records, categories_dict id->name).
    """
    records = []
    categories = OrderedDict()

    def ingest(ann_path: Path, split_tag: str):
        coco = _read_coco(ann_path)
        img_dir = ann_path.parent
        # categories union (warn on id->name conflicts)
        for c in coco.get("categories", []):
            cid, cname = c["id"], c.get("name", str(c["id"]))
            if cid in categories and categories[cid] != cname:
                print(f"  [warn] category id {cid} maps to both "
                      f"'{categories[cid]}' and '{cname}'; keeping first.")
            categories.setdefault(cid, cname)
        # image_id -> annotations
        by_img = defaultdict(list)
        for a in coco.get("annotations", []):
            by_img[a["image_id"]].append(a)
        for im in coco.get("images", []):
            rec = {
                "wall": None,
                "src_split": split_tag,
                "img_dir": img_dir,
                "file_name": im["file_name"],
                "height": int(im.get("height", 0) or 0),
                "width": int(im.get("width", 0) or 0),
                "anns": [
                    {
                        "category_id": a.get("category_id", 1),
                        "bbox": a.get("bbox", [0, 0, 0, 0]),
                        "area": a.get("area", 0),
                        "segmentation": a.get("segmentation", []),
                        "iscrowd": a.get("iscrowd", 0),
                    }
                    for a in by_img.get(im["id"], [])
                ],
            }
            records.append(rec)

    flat = src / ANN_FILE
    if flat.exists():
        print(f"Loading flat COCO from {flat}")
        ingest(flat, "flat")
    else:
        found = False
        for split in ("train", "valid", "val", "test"):
            ap = src / split / ANN_FILE
            if ap.exists():
                print(f"Loading {split} COCO from {ap}")
                ingest(ap, split)
                found = True
        if not found:
            raise FileNotFoundError(
                f"No {ANN_FILE} found at {flat} nor under "
                f"{src}/(train|valid|test)/"
            )

    if not categories:
        categories[1] = "object"
    return records, categories


# --------------------------------------------------------------------------- #
# Writing one COCO split
# --------------------------------------------------------------------------- #
def materialize_image(src_path: Path, dst_path: Path, link_mode: str):
    if link_mode == "none":
        return
    if dst_path.exists():
        return
    if link_mode == "symlink":
        try:
            os.symlink(os.path.abspath(src_path), dst_path)
            return
        except OSError:
            pass  # fall through to copy (e.g. Windows / unsupported FS)
    if link_mode == "hardlink":
        try:
            os.link(src_path, dst_path)
            return
        except OSError:
            pass
    shutil.copy2(src_path, dst_path)


def write_split(out_dir: Path, records, categories, link_mode,
                keep_cats=None, fill_dims=False):
    """Write images + _annotations.coco.json for one split.

    keep_cats: optional set of category ids to keep (drops other annotations,
               and drops images that become empty if drop_empty handled by caller).
    Returns stats dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    used_names = {}
    img_id = 0
    ann_id = 1
    n_missing = 0
    n_anns_kept = 0

    cats_out = [
        {"id": cid, "name": cname, "supercategory": cname}
        for cid, cname in categories.items()
        if (keep_cats is None or cid in keep_cats)
    ]

    for rec in records:
        src_path = _resolve_image_path(rec["img_dir"], rec["file_name"])
        if src_path is None:
            print(f"  [warn] image not found, skipping: "
                  f"{rec['img_dir']}/{rec['file_name']}")
            n_missing += 1
            continue

        # de-collide basenames across merged source splits
        base = Path(rec["file_name"]).name
        if base in used_names and used_names[base] != str(src_path):
            stem, ext = os.path.splitext(base)
            base = f"{rec['wall'] or 'X'}_{img_id}_{stem}{ext}"
        used_names[base] = str(src_path)

        dst_path = out_dir / base
        materialize_image(src_path, dst_path, link_mode)

        h, w = rec["height"], rec["width"]
        if fill_dims and (h == 0 or w == 0) and _HAVE_PIL:
            try:
                with _PILImage.open(src_path) as im:
                    w, h = im.size
            except Exception:
                pass

        images.append({"id": img_id, "file_name": base,
                       "height": h, "width": w})

        for a in rec["anns"]:
            if keep_cats is not None and a["category_id"] not in keep_cats:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": a["category_id"],
                "bbox": a["bbox"],
                "area": a["area"],
                "segmentation": a["segmentation"],
                "iscrowd": a["iscrowd"],
            })
            ann_id += 1
            n_anns_kept += 1

        img_id += 1

    coco = {"images": images, "annotations": annotations, "categories": cats_out}
    with open(out_dir / ANN_FILE, "w") as f:
        json.dump(coco, f, indent=2)

    return {"images": len(images), "annotations": n_anns_kept,
            "missing": n_missing}


# --------------------------------------------------------------------------- #
# Val carving (wall-stratified, deterministic)
# --------------------------------------------------------------------------- #
def carve_val(pool, val_frac, rng):
    """Split pool into (train, val), stratified per wall so every wall in the
    training pool is represented in val. Deterministic given rng."""
    if val_frac <= 0:
        return pool, []
    by_wall = defaultdict(list)
    for r in pool:
        by_wall[r["wall"] or "UNKNOWN"].append(r)

    train, val = [], []
    for wall, recs in by_wall.items():
        recs = list(recs)
        rng.shuffle(recs)
        n_val = int(round(len(recs) * val_frac))
        # keep at least 1 train image per wall if possible
        n_val = min(n_val, max(0, len(recs) - 1))
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])
    return train, val


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Leave-One-Wall-Out COCO splitter for SAM3-LoRA crack work.")
    ap.add_argument("--src", required=True, type=Path,
                    help="Source dir: flat COCO, or dir with train/valid/test.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output root; creates fold_<wall>/ subdirs.")
    ap.add_argument("--walls", nargs="+", default=DEFAULT_WALLS,
                    help=f"Wall ids to recognise (default: {DEFAULT_WALLS}).")
    ap.add_argument("--exclude-test", nargs="*", default=DEFAULT_EXCLUDE_TEST,
                    help="Walls that never become a test fold "
                         f"(default: {DEFAULT_EXCLUDE_TEST}).")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of the training pool carved to valid (0=no valid).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link-mode", choices=["symlink", "copy", "hardlink", "none"],
                    default="symlink",
                    help="How to place images (none = write JSON only).")
    ap.add_argument("--keep-categories", nargs="*", default=None,
                    help="Keep only these category NAMES (e.g. CRACKS). "
                         "Default keeps all.")
    ap.add_argument("--on-unmatched", choices=["error", "drop", "train"],
                    default="error",
                    help="Images whose filename matches no wall: error out "
                         "(default), drop them, or route to train pool only.")
    ap.add_argument("--fill-dims", action="store_true",
                    help="Backfill missing image height/width via PIL.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wall_rx = build_wall_regex(args.walls)
    canon = {w.upper(): w for w in args.walls}

    records, categories = load_pool(args.src)
    print(f"\nMerged pool: {len(records)} images, "
          f"{sum(len(r['anns']) for r in records)} annotations")
    print(f"Categories: {dict(categories)}")

    # category-name filter -> set of ids
    keep_cats = None
    if args.keep_categories:
        want = {n.lower() for n in args.keep_categories}
        keep_cats = {cid for cid, cname in categories.items()
                     if cname.lower() in want}
        if not keep_cats:
            print(f"[error] --keep-categories {args.keep_categories} matched no "
                  f"category in {dict(categories)}")
            sys.exit(2)
        print(f"Keeping only category ids {sorted(keep_cats)} "
              f"({args.keep_categories})")

    # assign walls
    unmatched = []
    wall_counts = defaultdict(int)
    for r in records:
        r["wall"] = wall_of(r["file_name"], wall_rx, canon)
        if r["wall"] is None:
            unmatched.append(r["file_name"])
        else:
            wall_counts[r["wall"]] += 1

    print("\nImages per wall:")
    for w in args.walls:
        print(f"  {w:7s}: {wall_counts.get(w, 0)}")
    if unmatched:
        print(f"  UNMATCHED: {len(unmatched)} "
              f"(e.g. {unmatched[:3]})")

    if unmatched and args.on_unmatched == "error":
        print(f"\n[error] {len(unmatched)} images matched no wall id. "
              f"Fix naming, extend --walls, or pass "
              f"--on-unmatched {{drop,train}}.")
        sys.exit(2)
    if unmatched and args.on_unmatched == "drop":
        records = [r for r in records if r["wall"] is not None]
        print(f"  dropped {len(unmatched)} unmatched images.")
    # on_unmatched == 'train': keep them, wall stays None -> never a test wall,
    # always lands in the pool (train/valid) of every fold.

    present_walls = [w for w in args.walls if wall_counts.get(w, 0) > 0]
    exclude = set(args.exclude_test)
    test_walls = [w for w in present_walls if w not in exclude]
    if not test_walls:
        print("[error] No eligible test walls after exclusion.")
        sys.exit(2)

    print(f"\nFolds (held-out test wall): {test_walls}")
    print(f"Always-in-train walls: "
          f"{[w for w in present_walls if w in exclude]}"
          f"{' + UNMATCHED' if (unmatched and args.on_unmatched=='train') else ''}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for W in test_walls:
        fold_dir = args.out / f"fold_{W}"
        test_recs = [r for r in records if r["wall"] == W]
        pool_recs = [r for r in records if r["wall"] != W]  # incl. excluded + unmatched

        train_recs, val_recs = carve_val(pool_recs, args.val_frac, rng)

        # leakage guard: held-out wall must be absent from train+valid
        leak = {r["wall"] for r in (train_recs + val_recs)} & {W}
        assert not leak, f"LEAKAGE: wall {W} present in train/valid of its own fold"

        print(f"\n=== fold_{W} ===")
        s_train = write_split(fold_dir / "train", train_recs, categories,
                              args.link_mode, keep_cats, args.fill_dims)
        s_valid = write_split(fold_dir / "valid", val_recs, categories,
                              args.link_mode, keep_cats, args.fill_dims) \
            if val_recs else {"images": 0, "annotations": 0, "missing": 0}
        s_test = write_split(fold_dir / "test", test_recs, categories,
                             args.link_mode, keep_cats, args.fill_dims)

        for split_name, st, recs in (("train", s_train, train_recs),
                                     ("valid", s_valid, val_recs),
                                     ("test", s_test, test_recs)):
            walls_in = sorted({r["wall"] or "UNMATCHED" for r in recs})
            print(f"  {split_name:5s}: {st['images']:4d} imgs  "
                  f"{st['annotations']:4d} anns  walls={walls_in}")
            summary_rows.append({
                "fold": W, "split": split_name,
                "images": st["images"], "annotations": st["annotations"],
                "missing": st["missing"], "walls": "|".join(walls_in),
            })

    # write summaries
    csv_path = args.out / "lowo_summary.csv"
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["fold", "split", "images",
                                             "annotations", "missing", "walls"])
        wcsv.writeheader()
        wcsv.writerows(summary_rows)
    with open(args.out / "lowo_summary.json", "w") as f:
        json.dump({
            "src": str(args.src), "seed": args.seed,
            "walls": args.walls, "exclude_test": list(exclude),
            "val_frac": args.val_frac, "link_mode": args.link_mode,
            "keep_categories": args.keep_categories,
            "folds": summary_rows,
        }, f, indent=2)

    print(f"\nWrote {len(test_walls)} folds to {args.out}")
    print(f"Summary: {csv_path}")
    print("\nTrain a fold, e.g.:")
    print(f"  python3 train_sam3_lora_native_claude.py "
          f"--config configs/full_lora_config.yaml  "
          f"# set data_dir: {args.out}/fold_{test_walls[0]}")


if __name__ == "__main__":
    main()
