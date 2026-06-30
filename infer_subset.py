"""
infer_subset.py — run infer_sam.py on a VIEW-FILTERED subset of the test images.

Real-world use uploads mostly 'overview' and 'close-up' shots, so for the report
it makes sense to evaluate those views. The Roboflow file names don't carry the
view, so we resolve it from coco_with_meta.csv by IMG-core (the IMG_<digits> part),
pick up to N images of the wanted views (spread across drift levels so it's not all
one damage state), and call infer_sam.py on each.

EDIT the config, then:  python infer_subset.py
Outputs prediction overlays into PRED_DIR, ready for compare_test.py.
"""
import os, re, csv, json, glob, subprocess, collections, random

# ----- config (edit) -----
TEST_DIR   = r"D:\THESIS\03_annotation\folds\fold_RW20\test"
META_CSV   = r"D:\THESIS\03_annotation\MetaData\coco_with_meta.csv"
PRED_DIR   = r"D:\THESIS\03_annotation\preds_RW20"
REPO_DIR   = r"D:\THESIS\03_annotation\Test_Crack"            # where infer_sam.py lives
CONFIG     = REPO_DIR + r"\configs\full_lora_config.yaml"
WEIGHTS    = REPO_DIR + r"\outputs\sam3_lora_full\best_lora_weights.pt"
WANT_VIEWS = ["overview", "close-up"]   # views to keep (case-insensitive, '-'/'_'/' ' ignored)
N_IMAGES   = 50                         # cap on how many to run
PROMPT     = "crack"
THRESHOLD  = 0.3
SEED       = 42
# -------------------------

# ----- ENV OVERRIDES (let run_all.sh drive this per fold) -----
TEST_DIR   = os.environ.get("IS_TEST_DIR",   TEST_DIR)
META_CSV   = os.environ.get("IS_META_CSV",   META_CSV)
PRED_DIR   = os.environ.get("IS_PRED_DIR",   PRED_DIR)
REPO_DIR   = os.environ.get("IS_REPO_DIR",   REPO_DIR)
CONFIG     = os.environ.get("IS_CONFIG",     CONFIG)
WEIGHTS    = os.environ.get("IS_WEIGHTS",    WEIGHTS)
N_IMAGES   = int(os.environ.get("IS_N_IMAGES", N_IMAGES))
WANT_VIEWS = os.environ.get("IS_WANT_VIEWS", " ".join(WANT_VIEWS)).split()
THRESHOLD  = float(os.environ.get("IS_THRESHOLD", THRESHOLD))
# --------------------------------------------------------------

CORE_RE = re.compile(r'(IMG_\d+)', re.IGNORECASE)
def core(name): 
    m = CORE_RE.match(os.path.basename(name)); return m.group(1).upper() if m else None
def norm(v):
    return re.sub(r'[-_ ]', '', (v or '')).lower()


def main():
    want = {norm(v) for v in WANT_VIEWS}
    # IMG-core -> view, from coco_with_meta.csv
    view_of = {}
    for r in csv.DictReader(open(META_CSV, newline="", encoding="utf-8")):
        c = core(r["coco_file_name"])
        if c: view_of[c] = r.get("view", "")

    coco = json.load(open(os.path.join(TEST_DIR, "_annotations.coco.json"), "r", encoding="utf-8"))
    # bucket test images by drift so the subset spans damage states, not one level
    drift_of = {}
    for r in csv.DictReader(open(META_CSV, newline="", encoding="utf-8")):
        c = core(r["coco_file_name"])
        if c: drift_of[c] = r.get("drift", "?")

    picked, by_view = [], collections.Counter()
    cand = []
    for im in coco["images"]:
        c = core(im["file_name"])
        v = norm(view_of.get(c, ""))
        if v in want and os.path.exists(os.path.join(TEST_DIR, im["file_name"])):
            cand.append((im["file_name"], view_of.get(c, ""), drift_of.get(c, "?")))

    # spread across drift levels: round-robin by drift bucket, deterministic
    rng = random.Random(SEED)
    by_drift = collections.defaultdict(list)
    for fn, v, d in cand:
        by_drift[d].append((fn, v))
    for d in by_drift:
        rng.shuffle(by_drift[d])
    drifts = sorted(by_drift)
    while len(picked) < N_IMAGES and any(by_drift.values()):
        for d in drifts:
            if by_drift[d] and len(picked) < N_IMAGES:
                fn, v = by_drift[d].pop()
                picked.append(fn); by_view[v] += 1

    print(f"test images total : {len(coco['images'])}")
    print(f"matching views {WANT_VIEWS} : {len(cand)}  ->  running {len(picked)}")
    print(f"view breakdown    : {dict(by_view)}")
    if not picked:
        print("❌ nothing matched — check WANT_VIEWS spelling vs coco_with_meta.csv 'view' column"); return

    os.makedirs(PRED_DIR, exist_ok=True)
    for i, fn in enumerate(picked, 1):
        out = os.path.join(PRED_DIR, os.path.splitext(fn)[0] + ".png")
        cmd = ["python", os.path.join(REPO_DIR, "infer_sam.py"),
               "--config", CONFIG, "--weights", WEIGHTS,
               "--image", os.path.join(TEST_DIR, fn),
               "--prompt", PROMPT, "--output", out, "--threshold", str(THRESHOLD)]
        print(f"[{i}/{len(picked)}] {fn}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"   ⚠️ infer_sam.py failed on {fn} (rc={r.returncode}) — continuing")
    print(f"\n✅ predictions -> {PRED_DIR}")
    print("   next: set compare_test.py PRED_DIR to this folder and run it")


if __name__ == "__main__":
    main()
