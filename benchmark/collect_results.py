# -*- coding: utf-8 -*-
"""Archive EVERYTHING worth keeping before the vast.ai instance is destroyed.

Why this exists
---------------
`queue_runner.py --poweroff` calls `poweroff` the moment the last job ends.
The runbook said "tar the results back" in prose, but no code did it: an audit
on 2026-08-29 found zero tar/scp/rsync anywhere in the benchmark path. Every
artifact - checkpoints, masks, per-image eval CSVs, logs - lived only on the
instance disk that was about to be wiped. This script is now the LAST job in
the queue, so poweroff can only fire after the tarball exists.

Three classes of loss it prevents:
  1. things that cannot be recomputed at all (GPU timings, peak VRAM, the
     nnU-Net training curve, per-image predictions);
  2. things that live OUTSIDE runs/ and results/ and a naive tar would miss -
     $nnUNet_results (row A1's weights + progress.png + training_log), the
     generated configs/benchmark/*.yaml (the only record of what A6 actually
     trained with), /workspace/folds/folds_summary.json (the split-integrity
     proof from check_folds);
  3. provenance that is only knowable ON the instance - git SHA, pip freeze,
     nvidia-smi, torch/CUDA build.

Deliberately EXCLUDED: `last.pt` (optimizer state, ~3x best.pt, and a resume
checkpoint has no value once the instance is gone) and the nnU-Net
preprocessed arrays (regenerable from the pool; only their *.json plans and
fingerprints are kept, since those document A1's self-configuration).

Usage
-----
  python benchmark/collect_results.py                 # write the tarball
  python benchmark/collect_results.py --dry-run       # list only, no tar
  python benchmark/collect_results.py --out /workspace/bm.tar.gz

Exit code is 0 even when parts are missing - a partial archive after a failed
queue is exactly the situation this script exists for. Missing pieces are
listed loudly and recorded in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("BM_WORKSPACE", "/workspace"))

# (glob root, pattern, required?) - evaluated relative to REPO unless absolute
SPEC = [
    (REPO / "results", "**/*", True),
    (REPO / "runs", "*/run.json", False),
    (REPO / "runs", "*/train_log.csv", False),
    (REPO / "runs", "*/valid_log.csv", False),
    (REPO / "runs", "*/val_stats.json", False),
    (REPO / "runs", "*/DONE", False),
    (REPO / "runs", "*/best.pt", False),
    (REPO / "runs", "*/best_lora_weights.pt", False),
    (REPO / "runs", "*/masks/**/*", True),
    (REPO / "runs", "*.log", True),
    (REPO / "configs" / "benchmark", "*.yaml", False),
    (REPO, "jobs.yaml", False),
    (REPO, "queue_state.json", True),
    (REPO, "QUEUE_DONE", False),
    (REPO, "marked_line_images.txt", False),
    (WORK / "folds", "folds_summary.json", True),
]


def nnunet_spec():
    """Row A1 lives outside the repo, under the env dirs setup_benchmark set."""
    out = []
    res = os.environ.get("nnUNet_results")
    pre = os.environ.get("nnUNet_preprocessed")
    if res:
        out.append((Path(res), "**/*", False))
    if pre:
        out.append((Path(pre), "**/*.json", False))     # plans + fingerprint
    return out


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=120)
        return (r.stdout or r.stderr).strip()
    except Exception as e:                                # noqa: BLE001
        return f"<failed: {e}>"


def env_report():
    return "\n".join([
        "# environment captured by collect_results.py",
        f"utc            {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"repo           {REPO}",
        f"git_sha        {sh(f'git -C {REPO} rev-parse HEAD')}",
        f"git_status     {sh(f'git -C {REPO} status --porcelain') or '(clean)'}",
        f"python         {sys.version.split()[0]}",
        "",
        "## nvidia-smi", sh("nvidia-smi"),
        "",
        "## torch", sh(f'{sys.executable} -c "import torch;'
                       'print(torch.__version__, torch.version.cuda,'
                       'torch.cuda.get_device_name(0))"'),
        "",
        "## pip freeze", sh(f"{sys.executable} -m pip freeze"),
    ])


def collect(spec):
    """-> (list of (abs_path, arcname), list of missing group labels)"""
    files, missing = [], []
    for root, pat, required in spec:
        if not root.exists():
            if required:
                missing.append(f"{root} (root absent)")
            continue
        hits = [p for p in sorted(root.glob(pat)) if p.is_file()]
        if not hits and required:
            missing.append(f"{root}/{pat}")
        for p in hits:
            try:
                rel = p.relative_to(root.parent)
            except ValueError:
                rel = Path(root.name) / p.relative_to(root)
            files.append((p, str(rel).replace("\\", "/")))
    # de-dup (results/** and runs/*/masks/** can overlap with other patterns)
    seen, uniq = set(), []
    for p, arc in files:
        if arc in seen:
            continue
        seen.add(arc)
        uniq.append((p, arc))
    return uniq, sorted(set(missing))


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be archived; write nothing")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    out = a.out or (WORK if WORK.exists() else REPO) / f"bm_results_{stamp}.tar.gz"

    files, missing = collect(SPEC + nnunet_spec())
    total = sum(p.stat().st_size for p, _ in files)
    groups = {}
    for p, arc in files:
        groups[arc.split("/")[0]] = groups.get(arc.split("/")[0], 0) + p.stat().st_size
    print(f"{len(files)} files, {human(total)}")
    for g, sz in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {g:<24} {human(sz):>10}")
    if missing:
        print("\nMISSING (archiving anyway — a partial archive beats none):")
        for m in missing:
            print(f"  ! {m}")

    if a.dry_run:
        print(f"\n[dry-run] would write {out}")
        return 0

    md5 = []
    for _i, (p, arc) in enumerate(files, 1):
        print(f"\r[md5 {_i}/{len(files)}] {arc[:70]:<70}", end="",
              file=sys.stderr, flush=True)
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        md5.append(f"{h.hexdigest()}  {arc}")

    manifest = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(files), "bytes": total,
        "missing": missing,
        "groups": {g: sz for g, sz in groups.items()},
    }
    extras = {
        "env_report.txt": env_report(),
        "md5sums.txt": "\n".join(md5) + "\n",
        "collect_manifest.json": json.dumps(manifest, indent=2),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    print(file=sys.stderr)
    # compresslevel=1, not the default 9: the payload is mostly .pt/.png,
    # already incompressible - identical tar contents, 3-5x faster gzip.
    # Per-file counter (A1.16): this used to be the last silent stretch
    # before poweroff.
    _done = 0
    with tarfile.open(out, "w:gz", compresslevel=1) as tf:
        for _i, (p, arc) in enumerate(files, 1):
            tf.add(p, arcname=arc)
            _done += p.stat().st_size
            print(f"\r[tar {_i}/{len(files)}] {human(_done)} / {human(total)}   ",
                  end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        for name, text in extras.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(time.time())
            import io
            tf.addfile(info, io.BytesIO(data))

    size = out.stat().st_size
    print(f"\nwrote {out}  ({human(size)})")
    print("DOWNLOAD THIS BEFORE DESTROYING THE INSTANCE:")
    print(f"  scp -P <port> root@<host>:{out} .")
    print(f"  # or:  vastai copy <instance>:{out} .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
