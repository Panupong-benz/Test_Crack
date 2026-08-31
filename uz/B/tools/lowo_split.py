"""
lowo_split.py — build Leave-One-Wall-Out folds in the layout the SAM3 trainer eats.

COCOSegmentDataset reads  <data_dir>/<split>/_annotations.coco.json  plus the
images in that folder, so every fold is:

    fold_<WALL>/
        train/  <images> + _annotations.coco.json
        valid/  <images> + _annotations.coco.json
        test/   <images> + _annotations.coco.json   (the held-out wall)

Wall + load_step come from coco_with_meta.csv (reconcile.py), NOT the file name
(Roboflow names IMG_..._rf.<hash>.jpg carry neither).

Splitting rules:
  - TEST_WALLS  : each held out once as test (one fold each).
  - TRAIN_ONLY  : never tested; always in the train pool.
  - valid is carved from the train pool, wall-stratified & deterministic.
  - GROUP-BY-LOAD-STEP: the atomic unit moved to valid is a whole load_step
    (wall|drift|sign|cycle), so the side / close-up / overview views of the SAME
    crack state never straddle train and valid (which would make valid look
    artificially easy). load_step never appears in both train and valid.
  - test is always a different wall, so none of this can leak into the reported
    metric — this only makes early-stopping / model-selection honest.

EDIT the config block, then:  python lowo_split.py
"""
import os, csv, json, shutil, random, collections

# ----- config (edit) -----
# Paths come from anno_paths (2026-08-28): the tree was reorganised and these
# four constants silently pointed at the pre-reorg locations. Never hardcode.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
try:
    import anno_paths as _AP
    COCO_JSON = str(_AP.uncropped(_AP.BASE_POOL) / _AP.ANN)
    IMG_DIR   = str(_AP.uncropped(_AP.BASE_POOL))   # the .rf.<hash>.jpg images
    META_CSV  = str(_AP.meta_dir() / "coco_with_meta.csv")
    OUT_ROOT  = str(_AP.folds_dir())
except ImportError:
    # runs off-tree too (e.g. on the GPU instance, where the pool is unpacked
    # somewhere else); every one of these is required as a CLI arg there
    COCO_JSON = IMG_DIR = META_CSV = OUT_ROOT = ""
TEST_WALLS  = ["RW20", "RW20T"]     # held out one at a time
TRAIN_ONLY  = ["RW40", "N40"]       # never tested; always in train pool
VAL_FRAC    = 0.15                  # ~fraction of the train pool carved into valid
SEED        = 42                    # deterministic valid carve
IMG_MODE    = "copy"                # "copy" (self-contained, ~2x disk) or "symlink"
GROUP_BY_STEP = True                # True = move whole load_steps to valid (recommended)
# -------------------------

ANN = "_annotations.coco.json"


def load_meta(path):
    """coco_file_name -> {wall, step, view} from coco_with_meta.csv."""
    m = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8-sig")):
        m[r["coco_file_name"]] = {
            "wall": r.get("wall", ""),
            "step": r.get("load_step_id", "") or f'{r.get("wall","")}|{r.get("drift","")}',
            "view": r.get("view", "") or "?",
        }
    return m


def carve_val(images, val_frac, seed, group_by_step):
    """Split images into (train, val), wall-stratified & deterministic. When
    group_by_step, whole load_steps move together so views of one crack state
    never straddle train/valid. Always keeps >=1 group per wall in train."""
    rng = random.Random(seed)
    train, val = [], []
    by_wall = collections.defaultdict(list)
    for im in images:
        by_wall[im["_wall"]].append(im)

    for wall in sorted(by_wall):
        ims = by_wall[wall]
        # unit = load_step (group_by_step) or each image on its own
        groups = collections.defaultdict(list)
        for im in ims:
            key = im["_step"] if group_by_step else f'{im["_step"]}#{im["id"]}'
            groups[key].append(im)
        order = sorted(groups)            # stable before shuffle
        rng.shuffle(order)
        target = int(round(len(ims) * val_frac))
        chosen, v_count = set(), 0
        for key in order[:-1]:            # never the last group -> >=1 stays in train
            if v_count >= target:
                break
            chosen.add(key); v_count += len(groups[key])
        for key in order:
            (val if key in chosen else train).extend(groups[key])
    return train, val


MATERIALIZED = collections.Counter()   # reported at the end: symlink vs copy


def materialize(src, dst, mode):
    if mode == "symlink":
        try:
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            MATERIALIZED["symlink"] += 1
            return
        except OSError:
            # Windows without privileges, or a filesystem with no symlinks.
            # Falling back is fine but must not be SILENT: it costs ~4x disk
            # (every fold holds the whole pool).
            MATERIALIZED["copy_fallback"] += 1
    else:
        MATERIALIZED["copy"] += 1
    shutil.copy2(src, dst)


def write_split(out_dir, imgs, coco, anns_by_img, img_dir, mode):
    os.makedirs(out_dir, exist_ok=True)
    missing = []
    for im in imgs:
        src = os.path.join(img_dir, im["file_name"])
        if not os.path.exists(src):
            missing.append(im["file_name"]); continue
        materialize(src, os.path.join(out_dir, im["file_name"]), mode)
    keep = {im["id"] for im in imgs}
    out_imgs = [{k: v for k, v in im.items() if not k.startswith("_")} for im in imgs]
    anns = [a for i in keep for a in anns_by_img.get(i, [])]
    out = {"images": out_imgs, "annotations": anns, "categories": coco["categories"]}
    for k in ("info", "licenses"):
        if k in coco:
            out[k] = coco[k]
    json.dump(out, open(os.path.join(out_dir, ANN), "w"))
    return len(out_imgs), len(anns), missing


def main(args=None):
    try:
        import sys as _sys
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # CLI overrides (2026-08-20, pipeline_rerun_spec prep) - defaults reproduce
    # the original behaviour byte-for-byte when no arguments are given.
    COCO_JSON_, IMG_DIR_, META_CSV_, OUT_ROOT_ = COCO_JSON, IMG_DIR, META_CSV, OUT_ROOT
    TEST_WALLS_, TRAIN_ONLY_, IMG_MODE_ = list(TEST_WALLS), list(TRAIN_ONLY), IMG_MODE
    STATS_ONLY = False
    if args is not None:
        COCO_JSON_ = args.coco or COCO_JSON_
        IMG_DIR_ = args.img_dir or IMG_DIR_
        META_CSV_ = args.meta or META_CSV_
        OUT_ROOT_ = args.out or OUT_ROOT_
        if args.test_walls:
            TEST_WALLS_ = args.test_walls.split(",")
        if args.train_only:
            TRAIN_ONLY_ = args.train_only.split(",")
        IMG_MODE_ = args.img_mode or IMG_MODE_
        STATS_ONLY = args.stats_only

    for p in (COCO_JSON_, META_CSV_):
        if not os.path.exists(p):
            print(f"❌ not found: {p}"); return
    if not os.path.isdir(IMG_DIR_) and not STATS_ONLY:
        print(f"❌ IMG_DIR not found: {IMG_DIR_}"); return

    coco = json.load(open(COCO_JSON_, "r", encoding="utf-8"))
    meta = load_meta(META_CSV_)
    for im in coco["images"]:
        info = meta.get(im["file_name"], {"wall": "", "step": "", "view": "?"})
        im["_wall"], im["_step"], im["_view"] = info["wall"], info["step"], info["view"]
    untagged = [im for im in coco["images"] if not im["_wall"]]

    anns_by_img = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    present = collections.Counter(im["_wall"] for im in coco["images"] if im["_wall"])
    print("images per wall:", dict(present))
    print(f"total {len(coco['images'])} images | {len(coco['annotations'])} annotations | "
          f"untagged {len(untagged)} | group_by_step={GROUP_BY_STEP}")
    for w in TEST_WALLS_:
        if present.get(w, 0) == 0:
            print(f"❌ TEST wall '{w}' has 0 images — check coco_with_meta.csv"); return
    stray = [w for w in present if w not in set(TEST_WALLS_) | set(TRAIN_ONLY_)]
    if stray:
        print(f"⚠️ wall(s) not in TEST_WALLS/TRAIN_ONLY -> default to train pool: {stray}")
    if untagged:
        print(f"⚠️ {len(untagged)} untagged image(s) -> placed in train of every fold")

    os.makedirs(OUT_ROOT_, exist_ok=True)
    summary = {}
    print(f"\nBUILDING FOLDS  (val_frac={VAL_FRAC}, seed={SEED}, "
          f"img_mode={IMG_MODE_}{', STATS-ONLY' if STATS_ONLY else ''})")
    for W in TEST_WALLS_:
        test_imgs = [im for im in coco["images"] if im["_wall"] == W]
        pool      = [im for im in coco["images"] if im["_wall"] and im["_wall"] != W]
        tr_imgs, val_imgs = carve_val(pool, VAL_FRAC, SEED, GROUP_BY_STEP)
        tr_imgs += untagged

        fold = os.path.join(OUT_ROOT_, f"fold_{W}")
        if STATS_ONLY:
            # no dirs, no copies: counts + missing-file check only. A pool
            # built with merge_pools --no-images holds just the json - then
            # the existence check is meaningless and is skipped with a note.
            pool_has_imgs = any(f.lower().endswith((".jpg", ".jpeg", ".png"))
                                for f in os.listdir(IMG_DIR_)) \
                if os.path.isdir(IMG_DIR_) else False
            if not pool_has_imgs:
                print("      (json-only pool: image-existence check skipped)")

            def stat_split(imgs):
                miss = [im["file_name"] for im in imgs if pool_has_imgs and not
                        os.path.exists(os.path.join(IMG_DIR_, im["file_name"]))]
                return len(imgs), sum(len(anns_by_img.get(im["id"], []))
                                      for im in imgs), miss
            nt, at, m1 = stat_split(tr_imgs)
            nv, av, m2 = stat_split(val_imgs)
            ns, asx, m3 = stat_split(test_imgs)
        else:
            nt, at, m1 = write_split(os.path.join(fold, "train"), tr_imgs,  coco, anns_by_img, IMG_DIR_, IMG_MODE_)
            nv, av, m2 = write_split(os.path.join(fold, "valid"), val_imgs, coco, anns_by_img, IMG_DIR_, IMG_MODE_)
            ns, asx, m3 = write_split(os.path.join(fold, "test"), test_imgs, coco, anns_by_img, IMG_DIR_, IMG_MODE_)

        # guards
        leak = [im["file_name"] for im in (tr_imgs + val_imgs) if im["_wall"] == W]
        tr_steps = {im["_step"] for im in tr_imgs}
        va_steps = {im["_step"] for im in val_imgs}
        straddle = tr_steps & va_steps                          # load_steps in BOTH train & valid
        miss = m1 + m2 + m3
        tr_walls = collections.Counter(im["_wall"] for im in tr_imgs)
        va_walls = collections.Counter(im["_wall"] for im in val_imgs)
        va_views = collections.Counter(im["_view"] for im in val_imgs)
        tr_views = collections.Counter(im["_view"] for im in tr_imgs)
        summary[W] = {"train": nt, "valid": nv, "test": ns,
                      "train_walls": dict(tr_walls), "valid_walls": dict(va_walls),
                      "valid_views": dict(va_views),
                      "leak_images": len(leak), "straddle_loadsteps": len(straddle),
                      "missing_images": len(miss)}
        flag = "LEAK!" if leak else ("STRADDLE!" if straddle else ("MISSING IMG!" if miss else "ok"))
        print(f"  fold_{W}: train {nt} / valid {nv} / test {ns}  [{flag}]")
        print(f"      train walls={dict(tr_walls)}  valid walls={dict(va_walls)}")
        print(f"      view balance  train={dict(tr_views)}  valid={dict(va_views)}")
        print(f"      load_steps straddling train/valid: {len(straddle)}  (must be 0)")
        if leak:     print(f"      ❌ {len(leak)} held-out image leaked into train/valid")
        if straddle: print(f"      ❌ steps in both: {list(straddle)[:3]}")
        if miss:     print(f"      ❌ {len(miss)} image file(s) not found: {miss[:3]}")

    sname = "folds_summary_dryrun.json" if STATS_ONLY else "folds_summary.json"
    json.dump(summary, open(os.path.join(OUT_ROOT_, sname), "w"), indent=2)
    print(f"\n✅ wrote {len(TEST_WALLS_)} folds -> {OUT_ROOT_}  ({sname})")
    if MATERIALIZED:
        print(f"   images materialized: {dict(MATERIALIZED)}")
        if MATERIALIZED.get("copy_fallback"):
            print(f"   ! symlink was requested but {MATERIALIZED['copy_fallback']}"
                  f" file(s) had to be COPIED (no symlink support here) —"
                  f" the folds cost ~4x the pool on disk")
    print("   each fold = train/ valid/ test/  (images + _annotations.coco.json)")
    print("   point run_lowo.py --lowo-root at this folder.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="LOWO fold builder; no args = original 4-wall behaviour")
    ap.add_argument("--coco"), ap.add_argument("--img-dir"), ap.add_argument("--meta")
    ap.add_argument("--out")
    ap.add_argument("--test-walls", help="comma list, e.g. RW20,RW20T,RW20C")
    ap.add_argument("--train-only", help="comma list, e.g. RW40,N40")
    ap.add_argument("--img-mode", choices=["copy", "symlink"])
    ap.add_argument("--stats-only", action="store_true",
                    help="compute splits + guards only; no dirs, no image copies")
    main(ap.parse_args())
