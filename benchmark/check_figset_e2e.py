# -*- coding: utf-8 -*-
"""A1.23 end-to-end gate on a synthetic fold - no real mask needed.

LOCAL-ONLY (needs matplotlib; the figure tier never runs on the rented box).
Run it after touching render_overlays.py or error_profile.py:
  python check_figset_e2e.py

Proves: (1) figset is deterministic (manifest byte-identical across runs),
(2) compare writes the composite AND the 4 raw panels, (3) the decoy
{stem}.png does NOT win end to end (error_profile would report a near-full
prediction if it did), (4) every mechanism rule fires on the right image.
"""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

BENCH = Path(__file__).resolve().parent
PY = sys.executable
W, H = 1200, 900


def build(root: Path):
    test = root / "data" / "fold_TESTW" / "test"
    test.mkdir(parents=True)
    masks = root / "runs" / "a6_TESTW_s0" / "masks"
    masks.mkdir(parents=True)
    res = root / "results" / "benchmark"
    res.mkdir(parents=True)

    names = ["IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"]
    # img1: long crack, well detected | img2: crack + a fat FP bar (grid line)
    # img3: NO annotations (the empty-GT / written-number case)
    polys = {
        "IMG_1.jpg": [[100, 100, 700, 100, 700, 112, 100, 112]],
        "IMG_2.jpg": [[200, 400, 900, 400, 900, 410, 200, 410]],
        "IMG_3.jpg": [],
    }
    images, anns, aid = [], [], 1
    for i, n in enumerate(names, 1):
        img = np.full((H, W, 3), 200, np.uint8)
        cv2.imwrite(str(test / n), img)
        images.append({"id": i, "file_name": n, "height": H, "width": W})
        for pl in polys[n]:
            xs, ys = pl[0::2], pl[1::2]
            anns.append({"id": aid, "image_id": i, "category_id": 1,
                         "segmentation": [pl], "iscrowd": 0,
                         "area": (max(xs) - min(xs)) * (max(ys) - min(ys)),
                         "bbox": [min(xs), min(ys), max(xs) - min(xs),
                                  max(ys) - min(ys)]})
            aid += 1
    (test / "_annotations.coco.json").write_text(json.dumps(
        {"images": images, "annotations": anns,
         "categories": [{"id": 1, "name": "crack"}]}), encoding="utf-8")

    # predictions + the DECOY overlay figure infer_sam writes beside them
    preds = {}
    m1 = np.zeros((H, W), np.uint8)
    m1[100:112, 100:600] = 255                      # crack, right-truncated
    preds["IMG_1"] = m1
    m2 = np.zeros((H, W), np.uint8)
    m2[400:410, 200:900] = 255                      # crack, fully traced
    m2[600:620, 100:1100] = 255                     # 20k px isolated FP bar
    preds["IMG_2"] = m2
    m3 = np.zeros((H, W), np.uint8)
    m3[700:730, 800:900] = 255                      # 3k px on an empty-GT img
    preds["IMG_3"] = m3
    for stem, m in preds.items():
        cv2.imwrite(str(masks / f"{stem}_mask.png"), m)
        # the decoy: a mostly-white RGB "figure", different shape on purpose
        cv2.imwrite(str(masks / f"{stem}.png"),
                    np.full((H // 2, W // 2, 3), 255, np.uint8))

    # per-image metrics (only the columns figset reads)
    rows = [
        {"model": "a6", "fold": "TESTW", "seed": "0", "image": "IMG_1.jpg",
         "tp": 6000, "fp": 0, "fn": 1200, "cliou_4px": "0.720"},
        {"model": "a6", "fold": "TESTW", "seed": "0", "image": "IMG_2.jpg",
         "tp": 7000, "fp": 20000, "fn": 0, "cliou_4px": "0.910"},
        {"model": "a6", "fold": "TESTW", "seed": "0", "image": "IMG_3.jpg",
         "tp": 0, "fp": 3000, "fn": 0, "cliou_4px": "0.000"},
    ]
    with open(res / "per_image_metrics.csv", "w", encoding="utf-8",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (res / "eval_a6_TESTW_s0.csv").write_text("image\n", encoding="utf-8")
    (root / "marked.txt").write_text("# comment\nIMG_2.jpg\nIMG_3.jpg\n",
                                     encoding="utf-8")
    return names


def run(root, out, manifest):
    cmd = [PY, str(BENCH / "render_overlays.py"), "figset",
           "--fold", str(root / "data" / "fold_TESTW"),
           "--models", "a6", "--seed", "0",
           "--per-image", str(root / "results/benchmark/per_image_metrics.csv"),
           "--marked-list", str(root / "marked.txt"),
           "--runs-dir", str(root / "runs"),
           "--out", str(out), "--manifest", str(manifest), "--dpi", "80"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit("figset failed")
    return r.stdout


def main():
    root = Path(tempfile.mkdtemp(prefix="a123_"))
    try:
        build(root)
        m1, m2 = root / "man1.csv", root / "man2.csv"
        run(root, root / "figs1", m1)
        run(root, root / "figs2", m2)

        a, b = m1.read_bytes(), m2.read_bytes()
        assert a == b, "figset is NOT deterministic"
        rows = list(csv.DictReader(m1.open(encoding="utf-8")))
        rules = [r["rule"] for r in rows]
        assert rules.count("data_driven:worst") == 1
        assert rules.count("data_driven:median") == 1
        assert rules.count("data_driven:best") == 1
        by = {r["rule"]: r for r in rows}
        assert by["data_driven:worst"]["image"] == "IMG_3.jpg", by
        assert by["data_driven:best"]["image"] == "IMG_2.jpg", by
        # mechanism: grid_line must take the marked image WITH GT (IMG_2),
        # written_number the empty-GT one (IMG_3), dense_zone the widest GT
        assert by["mechanism:grid_line"]["image"] == "IMG_2.jpg", by
        assert by["mechanism:written_number"]["image"] == "IMG_3.jpg", by
        # dense_zone = most GT px (tp+fn): IMG_1 7200 > IMG_2 7000
        assert by["mechanism:dense_zone"]["image"] == "IMG_1.jpg", by
        assert by["mechanism:clear_crack"]["image"] == "IMG_2.jpg", by
        for r in rows:
            assert r["figure"], r
        # every figure has 4 raw panels beside it
        for sub in ("data_driven", "mechanism"):
            d = root / "figs1" / sub
            comps = sorted(d.glob("fig_compare_*.png"))
            panels = sorted(d.glob("*__a_photo.png"))
            assert len(comps) == len(panels) > 0, (sub, len(comps))
            for c in comps:
                stem = c.name[len("fig_compare_"):-4]
                for sfx in ("a_photo", "b_label", "c_pred", "d_agree"):
                    assert (d / f"{stem}__{sfx}.png").exists(), (stem, sfx)

        # error_profile end to end: if the decoy had won, pred_px would be
        # the whole frame instead of the planted counts
        cmd = [PY, str(BENCH / "error_profile.py"),
               "--results", str(root / "results/benchmark"),
               "--data-root", str(root / "data"),
               "--runs-dir", str(root / "runs"),
               "--marked-list", str(root / "marked.txt")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        ep = list(csv.DictReader(
            (root / "results/benchmark/error_profile.csv").open(
                encoding="utf-8")))
        assert len(ep) == 3, ep
        e = {x["image"]: x for x in ep}
        assert int(e["IMG_1.jpg"]["pred_px"]) == 12 * 500, e["IMG_1.jpg"]
        assert int(e["IMG_1.jpg"]["fn_broken_px"]) > 0, e["IMG_1.jpg"]
        assert int(e["IMG_1.jpg"]["fn_missed_px"]) == 0, e["IMG_1.jpg"]
        assert int(e["IMG_2.jpg"]["fp_isolated_px"]) == 20 * 1000, e["IMG_2.jpg"]
        assert int(e["IMG_2.jpg"]["fp_touching_px"]) == 0, e["IMG_2.jpg"]
        assert int(e["IMG_3.jpg"]["gt_px"]) == 0 and \
            int(e["IMG_3.jpg"]["fp_isolated_px"]) == 30 * 100, e["IMG_3.jpg"]
        assert e["IMG_2.jpg"]["marked"] == "1" and e["IMG_1.jpg"]["marked"] == "0"

        print("E2E PASS: figset deterministic (manifest byte-identical), "
              f"{len(rows)} figures all manifest-backed, 4 raw panels each, "
              "mechanism rules hit the declared images, decoy overlay lost "
              "end to end (planted pixel counts recovered exactly)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
