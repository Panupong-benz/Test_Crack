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
  python3 benchmark/make_jobs.py --data-root data --out jobs.yaml \
      --base-config configs/full_lora_config.yaml \
      --batch 8 --gate-eval-first
Then:  python3 benchmark/queue_runner.py --jobs jobs.yaml --poweroff

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
ALL_ROWS = ["a1", "unet", "deeplabv3p", "segformer", "a5", "a6"]


def override_keys(cfg: dict, data_dir: str, seed: int, out_dir: str,
                  gpus: int = 1) -> dict:
    """Recursively set data_dir / seed / output_dir wherever they occur —
    tolerant of the config's section layout (the base YAML nests them).

    gpus > 1 (A1.27 item 150): divide gradient_accumulation_steps by gpus so
    the EFFECTIVE batch (micro x accum x ranks) and the optimizer steps per
    epoch stay exactly what the single-GPU run had. Refuses a non-integer
    split rather than silently changing the experiment."""
    cfg = copy.deepcopy(cfg)
    if gpus > 1:
        tr = cfg.setdefault("training", {})
        acc = int(tr.get("gradient_accumulation_steps", 1))
        if acc % gpus:
            raise SystemExit(f"gradient_accumulation_steps={acc} is not divisible "
                             f"by --gpus {gpus}: the effective batch could not be "
                             f"kept - pick a divisor or change the base config")
        tr["gradient_accumulation_steps"] = acc // gpus

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
    ap.add_argument("--a6-epochs", type=int, default=None,
                    help="A6 num_epochs override; default None = the base "
                         "config's value (30). Exists for REPRODUCIBILITY, "
                         "not budget cutting (A1.21 item 114): "
                         "configs/benchmark/ is gitignored, so the budget "
                         "must live in a committable command - a hand-"
                         "edited YAML is silently erased by the next "
                         "make_jobs run.")
    ap.add_argument("--a6-ext", type=int, default=None,
                    help="A6 extended budget when the pilot fold is "
                         "budget_limited (A1.22 item 117). Default None = 2 x "
                         "the base budget. Whole row moves together.")
    ap.add_argument("--marked-list", default="marked_line_images.txt")
    ap.add_argument("--results", default="results/benchmark")
    ap.add_argument("--nnunet-id-base", type=int, default=501)
    ap.add_argument("--gpus", type=int, default=1,
                    help="GPUs per A6 training (A1.27). 1 = byte-identical "
                         "queue; N>1 adds --device 0..N-1 (the trainer self-"
                         "launches torchrun) and divides grad accumulation by N "
                         "so the effective batch stays 16.")
    ap.add_argument("--rows", nargs="+", default=ALL_ROWS,
                    choices=ALL_ROWS, metavar="ROW",
                    help="which benchmark rows to emit (default: all six). "
                         "Interim A6-only rental (Amendment A1.8): "
                         "--rows a6 a5 --seeds 0. Rows left out are SHELVED, "
                         "not cancelled - a later rental adds them with the "
                         "same pool/folds and the eval CSVs merge in results/.")
    args = ap.parse_args()
    rows = set(args.rows)
    seg_rows = [a for a in SEG_ARCHS if a in rows]

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

    def eval_cmd(fold, pred_dir, tag, out_name=None):
        # out_name overrides the eval_<tag>.csv convention. summarize_benchmark
        # globs eval_*.summary.json, so the SMOKE eval must not use it or every
        # real run would carry a "WARN: unrecognized tag eval_smoke" line.
        out = out_name or f"eval_{tag}.csv"
        return (f"python3 benchmark/eval_masks.py "
                f"--gt {args.data_root}/fold_{fold}/test "
                f"--pred {pred_dir} "
                f"--out {args.results}/{out} "
                f"--marked-list {args.marked_list}")

    # ---- Phase 4a: smoke (eval unit test + 50-step per SELECTED arch) ----
    prev = add("smoke_eval_unit", "python3 benchmark/eval_masks.py --selftest")
    for arch in seg_rows:
        prev = add(f"smoke_{arch}",
                   f"python3 benchmark/train_seg.py --arch {arch} "
                   f"--data {args.data_root}/fold_{args.folds[0]} "
                   f"--out runs/smoke_{arch} --seed 0 --batch {args.batch} "
                   f"--smoke 50", after=prev)
    # predict_seg is the ONLY script whose first execution would otherwise be
    # hours into the paid queue (A2-A4 inference; A1.5). Exercise the whole
    # tail here on 3 images off the 50-step probe checkpoint - the numbers are
    # meaningless, the pipe is what is under test. Spelled exactly like the
    # real jobs below so the smoke tests the same command, not a similar one.
    # (Only when a seg row is selected at all - an A6-only queue has no
    # predict_seg jobs to protect.)
    if seg_rows:
        prev = add("smoke_predict",
                   f"python3 benchmark/predict_seg.py "
                   f"--run runs/smoke_{seg_rows[0]} "
                   f"--fold {args.data_root}/fold_{args.folds[0]} "
                   f"--out runs/smoke_{seg_rows[0]}/masks --limit 3",
                   after=prev)
        prev = add("smoke_eval_real",
                   eval_cmd(args.folds[0], f"runs/smoke_{seg_rows[0]}/masks",
                            "smoke", out_name="smoke_eval.csv"), after=prev)

    # ---- A6 SAM3-LoRA: kill-gate run first, then the rest ----------------
    _adaptive_common = (f"--results {args.results} "
                        f"--config-dir {args.config_dir.as_posix()} "
                        f"--base-config {args.base_config.as_posix()} "
                        f"--data-root {args.data_root}"
                        + (f" --gpus {args.gpus}" if args.gpus > 1 else ""))
    # --device 0..N-1 only when N>1: a single-GPU queue must stay byte-
    # identical to every queue generated before A1.27.
    _dev = (" --device " + " ".join(str(i) for i in range(args.gpus))
            if args.gpus > 1 else "")

    def a6_jobs(fold, seed, after, gate=False, adaptive=False):
        """gate=True keeps the whole chain non-optional and returns the EVAL
        (the kill-gate: a bad gate must stop the queue before the expensive
        rest). Otherwise infer/eval are optional leaves and the TRAIN is
        returned, so the chain advances through trainings only (A1.5).

        adaptive=True (every A6 run except the pilot, A1.22 item 117): the
        config is written at RUN time by `a6_adaptive.py train` at the budget
        the pilot's `decide` job recorded - it hard-fails without a decision
        file, never defaulting silently."""
        tag = f"a6_{fold}_s{seed}"
        if adaptive:
            t = add(tag, f"python3 benchmark/a6_adaptive.py train "
                         f"--fold {fold} --seed {seed} {_adaptive_common}",
                    after=after)
        else:
            cfg = override_keys(base_cfg, f"{args.data_root}/fold_{fold}",
                                seed, f"runs/{tag}", gpus=args.gpus)
            if args.a6_epochs:
                # set AFTER override_keys - its key-name walk would clobber
                # every num_epochs in the tree (A1.21 item 114)
                cfg["training"]["num_epochs"] = int(args.a6_epochs)
            cfg_path = args.config_dir / f"{tag}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False),
                                encoding="utf-8")
            # --resume on EVERY A6 train (A1.22 item 120): no ckpt_state.pt =
            # fresh start, so the identical command re-run by queue_runner
            # after a kill continues from the last completed epoch instead of
            # epoch 0 (the A1.21 item 107 risk, closed).
            t = add(tag, f"python3 train_sam3_lora_native_claude.py "
                         f"--config {cfg_path.as_posix()} --resume{_dev}",
                    after=after)
        # weight filename confirmed at smoke hour; find keeps it layout-proof
        i = add(f"infer_{tag}",
                f"python3 benchmark/run_a5_zeroshot.py "
                f"--fold {args.data_root}/fold_{fold} --out runs/{tag}/masks "
                f"--weights \"$(find runs/{tag} -name best_lora_weights.pt "
                f"| head -1)\"", after=t, optional=not gate)
        e = add(f"eval_{tag}", eval_cmd(fold, f"runs/{tag}/masks", tag),
                after=i, optional=not gate)
        return e if gate else t

    last = prev
    if "a6" in rows:
        gate = a6_jobs(args.folds[0], args.seeds[0], prev, gate=True)
        # GATE: queue_runner stops here automatically if the gate run fails;
        # the clDice-vs-old-checkpoint judgement (runbook 4b) is a manual read
        # of eval_a6_<fold0>_s0 before letting the queue continue overnight.

        # ---- A1.22 upward-only budget rule, as static jobs ---------------
        # The pilot's B-epoch curve is judged by the FROZEN A1.7 criterion;
        # budget_limited -> the WHOLE row goes to EXT: the pilot continues via
        # --resume (same program: prefix property), later folds train at EXT
        # from scratch. Both decide and extend are non-optional: an
        # unreadable curve must stop the queue, and a failed extension must
        # not let later folds proceed at a budget the pilot never reached.
        f0, s0 = args.folds[0], args.seeds[0]
        pilot = f"a6_{f0}_s{s0}"
        B = int(args.a6_epochs or base_cfg["training"]["num_epochs"])
        EXT = int(args.a6_ext or 2 * B)
        dec = add(f"decide_{pilot}",
                  f"python3 benchmark/a6_adaptive.py decide "
                  f"--pilot-run runs/{pilot} --base {B} --ext {EXT} "
                  f"{_adaptive_common}", after=gate)
        ext = add(f"extend_{pilot}",
                  f"python3 benchmark/a6_adaptive.py extend --run runs/{pilot} "
                  f"{_adaptive_common}", after=dec)
        # post-extension infer/eval: no-ops unless EXTENDED exists. Masks go
        # to masks_x and are swapped in only on success, so a mid-way failure
        # never leaves a half-overwritten mask dir beside a stale eval CSV.
        guard = f"[ -f runs/{pilot}/EXTENDED ] || exit 0; "
        ix = add(f"inferx_{pilot}",
                 guard + f"python3 benchmark/run_a5_zeroshot.py "
                 f"--fold {args.data_root}/fold_{f0} --out runs/{pilot}/masks_x "
                 f"--weights \"$(find runs/{pilot} -name best_lora_weights.pt "
                 f"| head -1)\" && rm -rf runs/{pilot}/masks_B{B} "
                 f"&& mv runs/{pilot}/masks runs/{pilot}/masks_B{B} "
                 f"&& mv runs/{pilot}/masks_x runs/{pilot}/masks",
                 after=ext, optional=True)
        add(f"evalx_{pilot}",
            guard + eval_cmd(f0, f"runs/{pilot}/masks", pilot),
            after=ix, optional=True)
        last = ext
        for fold in args.folds:
            for seed in args.seeds:
                if fold == f0 and seed == s0:
                    continue
                last = a6_jobs(fold, seed, last, adaptive=True)

    # ---- A5 zero-shot (no training; cheap, right after the gate) ---------
    # no training here, so these are leaves off the last A6 train: one failed
    # zero-shot fold costs its own row and nothing else. `last` is deliberately
    # NOT advanced, so the A2-A4 block still chains to a training job.
    if "a5" in rows:
        for fold in args.folds:
            tag = f"a5_{fold}"
            i = add(tag, f"python3 benchmark/run_a5_zeroshot.py "
                         f"--fold {args.data_root}/fold_{fold} "
                         f"--out runs/{tag}/masks", after=last, optional=True)
            add(f"eval_{tag}", eval_cmd(fold, f"runs/{tag}/masks", tag),
                after=i, optional=True)

    # ---- A2/A3/A4 --------------------------------------------------------
    for arch in seg_rows:
        for fold in args.folds:
            for seed in args.seeds:
                tag = f"{arch}_{fold}_s{seed}"
                t = add(tag,
                        f"python3 benchmark/train_seg.py --arch {arch} "
                        f"--data {args.data_root}/fold_{fold} "
                        f"--out runs/{tag} --seed {seed} "
                        f"--batch {args.batch} --epochs {args.epochs} "
                        f"--resume", after=last)
                # pred/eval hang off their OWN train as optional leaves, and
                # `last` advances through the TRAIN. Training is the expensive
                # irreplaceable half; masks and scores are cheap and can be
                # regenerated. Before A1.5 a broken eval took the remaining 35
                # runs plus all of A1 down with it.
                p = add(f"pred_{tag}",
                        f"python3 benchmark/predict_seg.py --run runs/{tag} "
                        f"--fold {args.data_root}/fold_{fold} "
                        f"--out runs/{tag}/masks", after=t, optional=True)
                add(f"eval_{tag}", eval_cmd(fold, f"runs/{tag}/masks", tag),
                    after=p, optional=True)
                last = t

    # ---- A1 nnU-Net (per fold; seed policy decided at smoke hour) --------
    for k, fold in enumerate(args.folds if "a1" in rows else []):
        did = args.nnunet_id_base + k
        name = f"BM_{fold}"
        tag = f"a1_{fold}"
        c = add(f"nnraw_{tag}",
                f"python3 benchmark/to_nnunet.py "
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
                 f"-tr nnUNetTrainer_250epochs", after=t, optional=True)
        add(f"eval_{tag}", eval_cmd(fold, f"runs/{tag}/masks", tag),
            after=p2, optional=True)
        last = t

    # ---- final jobs: summary tables, then ARCHIVE -----------------------
    # ---- Axis B (external transfer, both directions) ---------------------
    # ONE optional job that discovers whatever external data was downloaded
    # onto the instance and skips the rest (Amendment A1.6). Before this, axis
    # B was not in the queue at all and ran only if someone remembered eight
    # commands by hand; now downloading the data is enough, and not
    # downloading it costs nothing but a recorded skip in axis_b_auto.json.
    add("axis_b",
        f"python3 benchmark/axis_b.py auto --data-root {args.data_root} "
        f"--folds {' '.join(args.folds)} "
        f"--marked-list {args.marked_list} --results {args.results}",
        after=last, optional=True)

    # ---- was the epoch budget enough? ------------------------------------
    # Reads every run's validation curve against the criterion pre-registered
    # in epoch_saturation.py's docstring, and must run BEFORE summarize, which
    # picks up epoch_saturation.csv to mark budget-limited rows in the table
    # itself (Amendment A1.7).
    add("epoch_saturation",
        f"python3 benchmark/epoch_saturation.py --runs runs "
        f"--results {args.results} --config-dir {args.config_dir.as_posix()}",
        after=last, optional=True)

    # optional BY DESIGN: --strict exits 1 when a run is missing, which is
    # exactly what happens once any optional leaf was skipped. summarize
    # regenerates bit-identically on the local machine from the eval CSVs
    # (runbook SS5 item 4), so it must never be the thing that blocks the
    # archive - the exit code still lands in queue_state.json to be read.
    add("summarize",
        f"python3 benchmark/summarize_benchmark.py --results {args.results} "
        f"--folds {' '.join(args.folds)} "
        f"--seeds {' '.join(map(str, args.seeds))} "
        f"--models {' '.join(r for r in ALL_ROWS if r in rows)} "
        f"--strict", after=last,
        optional=True)
    # resource usage summary (Amendment A1.9): peaks + MIN disk free = the
    # number the next rental's container size comes from. Must be a queue job
    # (poweroff fires before control returns to bash) and must precede collect
    # so the summary lands inside the tar.
    add("resource_report",
        "python3 benchmark/resource_monitor.py --report", optional=True)

    # queue_runner --poweroff destroys the box right after the last job, so
    # the tarball MUST be the last job (Amendment A1.4) and it carries NO
    # "after": with `after: summarize` a failing summarize skipped the collect
    # and poweroff then destroyed the night's results (Amendment A1.5).
    # optional as well: a failed archive must not stop the poweroff, and the
    # runbook tells you to run it by hand if the queue died earlier.
    add("collect", "python3 benchmark/collect_results.py", optional=True)

    args.out.write_text(yaml.safe_dump({"jobs": jobs}, sort_keys=False,
                                       width=1000), encoding="utf-8")
    n_train = sum(1 for j in jobs if j["name"].startswith(
        ("a6_", "unet_", "deeplabv3p_", "segformer_", "a1_")))
    print(f"{args.out}: {len(jobs)} jobs ({n_train} training runs, "
          f"{len(args.folds)} folds x seeds {args.seeds}, "
          f"rows {sorted(rows)})")
    print(f"A6 configs -> {args.config_dir}/  (data_dir/seed/output_dir "
          f"overridden; everything else = base config)")


if __name__ == "__main__":
    main()
