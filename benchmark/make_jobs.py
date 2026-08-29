# -*- coding: utf-8 -*-
"""Generate the full benchmark job queue (jobs.yaml) + per-run A6 configs.

Encodes the money-safe ladder of docs/vastai_runbook.md SS4:
  smoke -> A6 fold_RW20 seed0 (kill-gate) -> remaining A6 -> A5 -> A2/A3/A4
  (train -> predict -> eval each) -> A1 nnU-Net per fold.
Every trained run is followed immediately by its inference + eval job, so a
dead instance loses at most the run in progress and finished rows are always
already scored (no idle, no unscored checkpoints).

A6 configs: the production trainer takes ONLY --config; data_dir / seed /
output_dir live in the YAML. This script clones the base config per
(fold, seed) with those three keys overridden — nothing else is touched.

Run on the instance AFTER the smoke hour (batch sizes measured there):
  python benchmark/make_jobs.py --data-root data --out jobs.yaml \
      --base-config configs/full_lora_config.yaml \
      --batch 8 --gate-eval-first
Then:  python benchmark/queue_runner.py --jobs jobs.yaml --poweroff

nnU-Net seed note: nnUNetv2_train exposes no seed flag; the 3-seed policy for
row A1 is decided at smoke hour (trainer subclass or a declared 1-seed row in
Amendment A1) — jobs here are emitted per fold only, loudly commented.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

FOLD_WALLS = ["RW20", "RW20C", "RW20L", "RW20T"]
SEG_ARCHS = ["unet", "deeplabv3p", "segformer"]


def override_keys(cfg: dict, data_dir: str, seed: int, out_dir: str) -> dict:
    """Recursively set data_dir / seed / output_dir wherever they occur —
    tolerant of the config's section layout (the base YAML nests them)."""
    cfg = copy.deepcopy(cfg)

    def walk(node):
        if isinstance(node, dict):
            for k in list(node):
                if k == "data_dir":
                    node[k] = data_dir
                elif k == "seed":
                    node[k] = seed
                elif k == "output_dir":
                    node[k] = out_dir
                else:
                    walk(node[k])
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(cfg)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data",
                    help="dir holding fold_<wall>/ (symlink made by setup_benchmark.sh)")
    ap.add_argument("--out", type=Path, default=Path("jobs.yaml"))
    ap.add_argument("--base-config", type=Path,
                    default=Path("configs/full_lora_config.yaml"))
    ap.add_argument("--config-dir", type=Path,
                    default=Path("configs/benchmark"))
    ap.add_argument("--folds", nargs="+", default=FOLD_WALLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=8,
                    help="A2-A4 batch size (from the smoke-hour sweep)")
    ap.add_argument("--epochs", type=int, default=40, help="A2-A4 epochs")
    ap.add_argument("--marked-list", default="marked_line_images.txt")
    ap.add_argument("--results", default="results/benchmark")
    ap.add_argument("--nnunet-id-base", type=int, default=501)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.config_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results).mkdir(parents=True, exist_ok=True)

    jobs = []

    def add(name, cmd, after=None, optional=False):
        j = {"name": name, "cmd": cmd}
        if after:
            j["after"] = after
        if optional:
            j["optional"] = True
        jobs.append(j)
        return name

    def eval_cmd(fold, pred_dir, tag):
        return (f"python benchmark/eval_masks.py "
                f"--gt {args.data_root}/fold_{fold}/test "
                f"--pred {pred_dir} "
                f"--out {args.results}/eval_{tag}.csv "
                f"--marked-list {args.marked_list}")

    # ---- Phase 4a: smoke (eval unit test + 50-step per arch) -------------
    prev = add("smoke_eval_unit", "python benchmark/eval_masks.py --selftest")
    for arch in SEG_ARCHS:
        prev = add(f"smoke_{arch}",
                   f"python benchmark/train_seg.py --arch {arch} "
                   f"--data {args.data_root}/fold_{args.folds[0]} "
                   f"--out runs/smoke_{arch} --seed 0 --batch {args.batch} "
                   f"--smoke 50", after=prev)

    # ---- A6 SAM3-LoRA: kill-gate run first, then the rest ----------------
    def a6_jobs(fold, seed, after):
        tag = f"a6_{fold}_s{seed}"
        cfg = override_keys(base_cfg, f"{args.data_root}/fold_{fold}",
                            seed, f"runs/{tag}")
        cfg_path = args.config_dir / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False),
                            encoding="utf-8")
        t = add(tag, f"python train_sam3_lora_native_claude.py "
                     f"--config {cfg_path.as_posix()}", after=after)
        # weight filename confirmed at smoke hour; find keeps it layout-proof
        i = add(f"infer_{tag}",
                f"python benchmark/run_a5_zeroshot.py "
                f"--fold {args.data_root}/fold_{fold} --out runs/{tag}/masks "
                f"--weights \"$(find runs/{tag} -name best_lora_weights.pt "
                f"| head -1)\"", after=t)
        return add(f"eval_{tag}",
                   eval_cmd(fold, f"runs/{tag}/masks", tag), after=i)

    gate = a6_jobs(args.folds[0], args.seeds[0], prev)
    # GATE: queue_runner stops here automatically if the gate run fails;
    # the clDice-vs-old-checkpoint judgement (runbook 4b) is a manual read
    # of eval_a6_<fold0>_s0 before letting the queue continue overnight.
    last = gate
    for fold in args.folds:
        for seed in args.seeds:
            if fold == args.folds[0] and seed == args.seeds[0]:
                continue
            last = a6_jobs(fold, seed, last)

    # ---- A5 zero-shot (no training; cheap, right after the gate) ---------
    for fold in args.folds:
        tag = f"a5_{fold}"
        i = add(tag, f"python benchmark/run_a5_zeroshot.py "
                     f"--fold {args.data_root}/fold_{fold} "
                     f"--out runs/{tag}/masks", after=last)
        last = add(f"eval_{tag}", eval_cmd(fold, f"runs/{tag}/masks", tag),
                   after=i)

    # ---- A2/A3/A4 --------------------------------------------------------
    for arch in SEG_ARCHS:
        for fold in args.folds:
            for seed in args.seeds:
                tag = f"{arch}_{fold}_s{seed}"
                t = add(tag,
                        f"python benchmark/train_seg.py --arch {arch} "
                        f"--data {args.data_root}/fold_{fold} "
                        f"--out runs/{tag} --seed {seed} "
                        f"--batch {args.batch} --epochs {args.epochs} "
                        f"--resume", after=last)
                p = add(f"pred_{tag}",
                        f"python benchmark/predict_seg.py --run runs/{tag} "
                        f"--fold {args.data_root}/fold_{fold} "
                        f"--out runs/{tag}/masks", after=t)
                last = add(f"eval_{tag}",
                           eval_cmd(fold, f"runs/{tag}/masks", tag), after=p)

    # ---- A1 nnU-Net (per fold; seed policy decided at smoke hour) --------
    for k, fold in enumerate(args.folds):
        did = args.nnunet_id_base + k
        name = f"BM_{fold}"
        tag = f"a1_{fold}"
        c = add(f"nnraw_{tag}",
                f"python benchmark/to_nnunet.py "
                f"--fold {args.data_root}/fold_{fold} --dataset-id {did} "
                f"--name {name} --raw \"$nnUNet_raw\"", after=last)
        p1 = add(f"nnplan_{tag}",
                 f"nnUNetv2_plan_and_preprocess -d {did} -c 2d "
                 f"--verify_dataset_integrity", after=c)
        t = add(tag, f"nnUNetv2_train {did} 2d all "
                     f"-tr nnUNetTrainer_250epochs", after=p1)
        p2 = add(f"pred_{tag}",
                 f"nnUNetv2_predict "
                 f"-i \"$nnUNet_raw\"/Dataset{did}_{name}/imagesTs "
                 f"-o runs/{tag}/masks -d {did} -c 2d -f all "
                 f"-tr nnUNetTrainer_250epochs", after=t)
        last = add(f"eval_{tag}", eval_cmd(fold, f"runs/{tag}/masks", tag),
                   after=p2)

    # ---- final jobs: summary tables, then ARCHIVE -----------------------
    add("summarize",
        f"python benchmark/summarize_benchmark.py --results {args.results} "
        f"--folds {' '.join(args.folds)} "
        f"--seeds {' '.join(map(str, args.seeds))} --strict", after=last)
    # queue_runner --poweroff destroys the box right after the last job, so
    # the tarball MUST be the last job (Amendment A1.4). optional: a failed
    # archive must not be the thing that stops us from powering off, and the
    # runbook tells you to run it by hand if the queue dies earlier.
    add("collect", "python benchmark/collect_results.py",
        after="summarize", optional=True)

    args.out.write_text(yaml.safe_dump({"jobs": jobs}, sort_keys=False,
                                       width=1000), encoding="utf-8")
    n_train = sum(1 for j in jobs if j["name"].startswith(
        ("a6_", "unet_", "deeplabv3p_", "segformer_", "a1_")))
    print(f"{args.out}: {len(jobs)} jobs ({n_train} training runs, "
          f"{len(args.folds)} folds x seeds {args.seeds})")
    print(f"A6 configs -> {args.config_dir}/  (data_dir/seed/output_dir "
          f"overridden; everything else = base config)")


if __name__ == "__main__":
    main()
