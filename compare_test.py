"""
compare_test.py — eyeball whether the model's segmentation matches YOUR labels.

For every image in the test split it draws your ground-truth crack polygons (the
ones you labelled in Roboflow) in GREEN, and — if the prediction overlays from
infer_sam.py are available — places the model output right next to it, so you can
flip through and see agreement / misses / false positives per image.

Inputs:
  TEST_DIR : a fold's test/ folder  (images + _annotations.coco.json)
  PRED_DIR : folder of prediction overlays produced by infer_sam.py (optional).
             A prediction file is matched to a test image by the same stem.
  OUT_DIR  : where the side-by-side panels are written.

If PRED_DIR is empty/missing you still get the GT overlays alone (useful to sanity
-check the labels themselves). Run infer_sam.py over the test images first to get
the predictions (see the command block in the chat).

EDIT the paths, then:  python compare_test.py
"""
import os, json, glob
import numpy as np
import cv2

# ----- config (edit) -----
TEST_DIR = r"D:\THESIS\03_annotation\folds\fold_RW20\test"
PRED_DIR = r"D:\THESIS\03_annotation\preds_RW20"     # infer_sam.py outputs; "" or missing = GT only
OUT_DIR  = r"D:\THESIS\03_annotation\compare_RW20"
PANEL_MAX_W = 1100      # downscale each panel half to this width (0 = no resize)
GT_COLOR = (0, 255, 0)  # BGR green for ground-truth label
# -------------------------

ANN = "_annotations.coco.json"
IMG_EXT = (".jpg", ".jpeg", ".png")


def stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def draw_gt(img, polys):
    """Draw GT polygons: translucent green fill + solid outline."""
    overlay = img.copy()
    for seg in polys:
        pts = np.array(seg, np.float32).reshape(-1, 2).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], GT_COLOR)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for seg in polys:
        pts = np.array(seg, np.float32).reshape(-1, 2).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], True, GT_COLOR, 2, cv2.LINE_AA)
    return img


def label(img, text, color=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def find_pred(pred_dir, img_stem):
    if not pred_dir or not os.path.isdir(pred_dir):
        return None
    for f in glob.glob(os.path.join(pred_dir, img_stem + ".*")):
        if f.lower().endswith(IMG_EXT):
            return f
    # infer_sam sometimes suffixes the output; match any file containing the stem
    cands = [f for f in glob.glob(os.path.join(pred_dir, "*")) if img_stem in os.path.basename(f)
             and f.lower().endswith(IMG_EXT)]
    return cands[0] if cands else None


def main():
    ann_path = os.path.join(TEST_DIR, ANN)
    if not os.path.exists(ann_path):
        print(f"❌ no {ANN} in {TEST_DIR}"); return
    coco = json.load(open(ann_path, "r", encoding="utf-8"))
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    os.makedirs(OUT_DIR, exist_ok=True)
    n_done, n_nogt, n_nopred = 0, 0, 0
    for im in coco["images"]:
        ip = os.path.join(TEST_DIR, im["file_name"])
        if not os.path.exists(ip):
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        polys = []
        for a in anns_by_img.get(im["id"], []):
            polys.extend(a.get("segmentation", []))
        if not polys:
            n_nogt += 1
        gt = label(draw_gt(img.copy(), polys), f"LABEL (GT): {len(polys)} crack(s)", GT_COLOR)

        pred_path = find_pred(PRED_DIR, stem(im["file_name"]))
        if pred_path:
            pred = cv2.imread(pred_path)
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            pred = label(pred, "MODEL PREDICTION")
            panel = np.hstack([gt, pred])
        else:
            n_nopred += 1
            panel = gt

        if PANEL_MAX_W and panel.shape[1] > PANEL_MAX_W:
            s = PANEL_MAX_W / panel.shape[1]
            panel = cv2.resize(panel, (PANEL_MAX_W, int(panel.shape[0] * s)))

        cv2.imwrite(os.path.join(OUT_DIR, stem(im["file_name"]) + ".png"), panel)
        n_done += 1

    print(f"✅ wrote {n_done} panels -> {OUT_DIR}")
    if n_nogt:   print(f"   ℹ️ {n_nogt} test image(s) had NO GT label (undamaged / nothing annotated)")
    if n_nopred: print(f"   ⚠️ {n_nopred} image(s) had no matching prediction in PRED_DIR "
                       f"(run infer_sam.py first, or check PRED_DIR)")


if __name__ == "__main__":
    main()
