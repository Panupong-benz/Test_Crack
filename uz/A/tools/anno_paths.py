# -*- coding: utf-8 -*-
"""Single source of truth for every 03_annotation path.

Why this file exists: on 2026-08-28 the annotation tree was reorganised
(everything pushed into "Old version/", new 01_Uncropped/02_Cropped created)
and THREE scripts broke silently because they carried absolute paths -
lowo_split.py's config block, reconcile_newwall.py's WALLS dict and
crop_map.py's MANIFEST. merge_pools.py survived only because it derived its
root from __file__. Import this module instead of writing a path.

Layout (stage-based, so the SS8d resolution trap is visible in the path):
  01_Uncropped/<WALL>/train/   full-res Roboflow export -> segmentation pool
                               (merge_pools -> lowo_split -> SAM3/benchmark)
  02_Cropped/<WALL>/           manual crop + transformed COCO + crop_manifest
                               -> Stage A rectification (crop_map)
  _meta/                       coco_with_meta*.csv + reconcile/build_metadata
  _folds/                      lowo_split.py, merge_pools.py, fold_*, summaries
  _pool/                       merged POOL_<row> COCO
  Old version/                 pre-2026-08-28 archive (migration source)

Every accessor resolves the NEW layout first and falls back to the archive,
so the tree can be migrated wall by wall without breaking anything. Ask
explain() what actually resolved.

    from anno_paths import uncropped, cropped, meta_dir, photo_root
    coco = uncropped("RW20") / "_annotations.coco.json"
"""
from __future__ import annotations

from pathlib import Path

ANNO = Path(__file__).resolve().parent
UNCROPPED = ANNO / "01_Uncropped"
CROPPED = ANNO / "02_Cropped"
META = ANNO / "_meta"
FOLDS = ANNO / "_folds"
POOLS = ANNO / "_pool"
LEGACY = ANNO / "Old version"
PHOTO_ROOT = ANNO.parent / "02_PhotoData"

ANN = "_annotations.coco.json"

# The 4-wall export that merge_pools uses as BASE is ONE merged Roboflow
# download (253 imgs), not the per-wall dirs - it keeps its own name.
BASE_POOL = "BASE_4WALL"

# where each thing lived before the 2026-08-28 reorg
_LEGACY_UNCROPPED = {
    BASE_POOL: "N40_RW20_RW20T_RW40/train",
    "RW20": "RW20_finish/train",
    "RW20T": "RW20T_finish/train",
    "RW40": "RW40_finish/train",
    "N40": "N40_finish/train",
    "RW20C": "RW20C/train",
}
_LEGACY_CROPPED = {
    "RW20": "RW20_finish/Output croped",
    "RW20T": "RW20T_finish/Output croped",
    "RW40": "RW40_finish/Output croped",
    "N40": "N40_finish/Output croped",
    "RW20C": "RW20C/Output croped",
}
_LEGACY_MANIFEST = {
    "RW20": "RW20_finish/crop_manifest.csv",
    "RW20T": "RW20T_finish/crop_manifest.csv",
    "RW40": "RW40_finish/crop_manifest.csv",
    "N40": "N40_finish/crop_manifest.csv",
    "RW20C": "RW20C/crop_manifest.csv",
}

# 02_PhotoData dir names are numbered and inconsistent - pin them here
PHOTO_DIRS = {
    "RW20": "USE 5RW20(No3)",
    "RW20C": "8RW20C",
    "RW20L": "6RW20L(No4)",
    "RW20T": "7RW20T(No5)",
    "RW40": "9RW40",
    "N40": "11N40(No2)",
    "N20B": "10N20B(No1)",
    "NSW3": "1NSW3",
    "NSW4": "2NSW4",
    "NSW5": "3NSW5",
    "NSW6": "4NSW6",
}

_resolved: dict[str, str] = {}       # what explain() reports


def _pick(key: str, new: Path, legacy_rel: str | None):
    """New layout wins; fall back to the archive while migrating."""
    if new.exists():
        _resolved[key] = f"new: {new}"
        return new
    if legacy_rel:
        old = LEGACY / legacy_rel
        if old.exists():
            _resolved[key] = f"LEGACY: {old}"
            return old
    _resolved[key] = f"MISSING (new would be {new})"
    return new                        # caller reports the miss with context


def uncropped(wall: str) -> Path:
    """Full-res Roboflow export dir for one wall (holds images + COCO)."""
    return _pick(f"uncropped:{wall}", UNCROPPED / wall / "train",
                 _LEGACY_UNCROPPED.get(wall))


def cropped(wall: str) -> Path:
    """Manually cropped images + transformed COCO for one wall."""
    return _pick(f"cropped:{wall}", CROPPED / wall, _LEGACY_CROPPED.get(wall))


def crop_manifest(wall: str) -> Path:
    """crop_manifest.csv mapping full frame <-> crop for one wall."""
    return _pick(f"manifest:{wall}", CROPPED / wall / "crop_manifest.csv",
                 _LEGACY_MANIFEST.get(wall))


def meta_dir() -> Path:
    return _pick("meta", META, "MetaData (4 walls)")


def folds_dir() -> Path:
    return _pick("folds", FOLDS, "Fold 4 walls")


def pool_dir(row: str = "") -> Path:
    p = _pick("pools", POOLS, None)
    return p / f"POOL_{row}" if row else p


def photo_root(wall: str) -> Path:
    """02_PhotoData tree for one wall (load-step folders live under it)."""
    d = PHOTO_DIRS.get(wall)
    if d is None:
        raise KeyError(f"no 02_PhotoData dir mapped for wall {wall!r} - "
                       f"add it to PHOTO_DIRS")
    return PHOTO_ROOT / d


def explain() -> str:
    """What every accessor called so far actually resolved to."""
    if not _resolved:
        return "(nothing resolved yet)"
    return "\n".join(f"  {k:<22} {v}" for k, v in sorted(_resolved.items()))


if __name__ == "__main__":
    print(f"ANNO = {ANNO}\n")
    for w in ("RW20", "RW20C", "RW20L", "RW20T", "RW40", "N40", "N20B"):
        u, c = uncropped(w), cropped(w)
        print(f"{w:<6} uncropped {'OK ' if u.exists() else '-- '}{u}")
        print(f"{'':<6} cropped   {'OK ' if c.exists() else '-- '}{c}")
    uncropped(BASE_POOL)
    meta_dir()
    folds_dir()
    print("\nresolution report:")
    print(explain())
