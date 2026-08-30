# -*- coding: utf-8 -*-
"""Pre-flight: every image must be in the SAME frame as its COCO record.

The bug this exists to prevent
------------------------------
PIL ignores the EXIF orientation flag; Roboflow (which wrote the COCO
width/height and every polygon) and cv2.imread apply it. On 2026-08-30 an
audit found 15 of POOL_BM's 381 photos carrying flag 6/8 - all from the
2026-08 full-res exports (RW20C 12, RW20L 1, N20B 2). For those, raw PIL
returned a LANDSCAPE array against a PORTRAIT COCO record, so:

  * tile_dataset cropped tiles at coordinates from the other axis and paired
    them with masks decoded in the COCO frame - silently corrupted training
    samples, and an occasional crash;
  * infer_sam predicted in the raw frame, so eval_masks squashed a landscape
    mask into the GT's portrait shape - garbage scores on 12 of RW20C's 53
    test images.

Both loaders now call exif_transpose. This script is the standing guard: it
compares the ORIENTED PIL size against the COCO record for every image of
every split, so a future Roboflow export cannot reintroduce the mismatch
without failing before a single GPU-hour is spent. PIL reads only the header,
so 381 images take about a second.

Usage:
  python benchmark/check_images.py --data /workspace/folds   # all folds
  python benchmark/check_images.py --data <dir with train/>  # one split set
  python benchmark/check_images.py --selftest

Exit 1 on any size mismatch or unreadable image.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageOps as PILImageOps

SPLITS = ("train", "valid", "test")
EXIF_ORIENTATION = 274


def check_split(split_dir: Path):
    """-> (n_images, n_exif_flagged, [mismatch rows])"""
    coco_fp = split_dir / "_annotations.coco.json"
    if not coco_fp.exists():
        return None
    coco = json.loads(coco_fp.read_text(encoding="utf-8"))
    flagged, bad = 0, []
    for im in coco["images"]:
        fp = split_dir / im["file_name"]
        if not fp.exists():
            bad.append((im["file_name"], "MISSING FILE", None, None))
            continue
        try:
            pil = PILImage.open(fp)
            flag = pil.getexif().get(EXIF_ORIENTATION)
            if flag not in (None, 1):
                flagged += 1
            w, h = PILImageOps.exif_transpose(pil).size
        except Exception as e:                                # noqa: BLE001
            bad.append((im["file_name"], f"UNREADABLE ({e})", None, None))
            continue
        if (w, h) != (im["width"], im["height"]):
            bad.append((im["file_name"], "SIZE MISMATCH",
                        f"{im['width']}x{im['height']}", f"{w}x{h}"))
    return len(coco["images"]), flagged, bad


def selftest():
    """A flagged image must FAIL when read raw and PASS when oriented - so a
    green run proves the check is live, not that it skipped the file."""
    import io
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        img = PILImage.new("RGB", (40, 20), "white")          # landscape
        exif = img.getexif()
        exif[EXIF_ORIENTATION] = 6                            # rotate 90 CW
        img.save(d / "IMG.jpg", exif=exif)
        # COCO records the DISPLAY size, i.e. rotated -> 20x40
        (d / "_annotations.coco.json").write_text(json.dumps(
            {"images": [{"id": 0, "file_name": "IMG.jpg",
                         "width": 20, "height": 40}],
             "annotations": [], "categories": []}), encoding="utf-8")
        raw = PILImage.open(d / "IMG.jpg").size
        oriented = PILImageOps.exif_transpose(
            PILImage.open(d / "IMG.jpg")).size
        assert raw == (40, 20), raw
        assert oriented == (20, 40), oriented
        n, flagged, bad = check_split(d)
        assert (n, flagged, bad) == (1, 1, []), (n, flagged, bad)
    print("selftest PASS: EXIF-flagged image is detected, and it matches COCO "
          "only after exif_transpose (raw 40x20 vs oriented 20x40)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=None,
                    help="a folds root (fold_*/{train,valid,test}) or a "
                         "single dir holding <split>/_annotations.coco.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.data is None:
        print("pass --data")
        return 2

    targets = []
    folds = sorted(a.data.glob("fold_*"))
    if folds:
        for f in folds:
            for s in SPLITS:
                if (f / s).is_dir():
                    targets.append((f"{f.name}/{s}", f / s))
    else:
        for s in SPLITS:
            if (a.data / s).is_dir():
                targets.append((s, a.data / s))
        if not targets and (a.data / "_annotations.coco.json").exists():
            targets.append((a.data.name, a.data))
    if not targets:
        print(f"no COCO splits under {a.data}")
        return 2

    total, total_flagged, all_bad = 0, 0, []
    for label, d in targets:
        r = check_split(d)
        if r is None:
            continue
        n, flagged, bad = r
        total += n
        total_flagged += flagged
        all_bad += [(label, *b) for b in bad]
        print(f"  {label:<22} {n:>4} images, {flagged:>3} EXIF-oriented, "
              f"{len(bad):>3} problem(s)")

    print(f"\n{total} images checked, {total_flagged} carry an EXIF "
          f"orientation flag (handled), {len(all_bad)} problem(s)")
    if all_bad:
        print("\nFAIL - image frame does not match the COCO record:")
        for row in all_bad[:20]:
            print(f"  {row[0]}  {row[1]}  {row[2]}  coco={row[3]} file={row[4]}")
        print("\nThe annotations cannot be aligned to these images. Fix the "
              "export or the loader before spending GPU time.")
        return 1
    print("PASS: every image matches its COCO width/height")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
