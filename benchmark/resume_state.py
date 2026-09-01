# -*- coding: utf-8 -*-
"""Exact-resume state for the A6 SAM3-LoRA trainer (Amendment A1.22).

Importable WITHOUT sam3 / bitsandbytes / triton, so the mechanism can be proven
on a dev box that cannot import the trainer, and then exercised for real on the
rental in `smoke_resume` (AdamW8bit round-trip on the GPU).

What "exact" means here, and what it does not
---------------------------------------------
A run with num_epochs = 2B is the same PROGRAM as the first B epochs of a run
with num_epochs = 2B plus B more, provided that at the epoch boundary we
restore: the LoRA weights, the optimizer moments, `best_val_loss`, the epoch
counter, and all four RNG streams the epoch consumes -
python `random`, numpy (tile_dataset augmentation), torch CPU (DataLoader's
per-iterator `base_seed`, ColorJitter) and torch CUDA (dropout / DropPath).
Data ORDER needs nothing: the sampler seeds a private generator from
(seed, epoch). This holds only with `persistent_workers=False`, because a
persistent iterator draws `base_seed` once and its workers' streams then advance
across epochs in state the main process cannot see.

Not claimed: bitwise-identical floats. `cudnn.benchmark = True` lets kernel
selection vary. The claim is "same data order, same LR, same RNG streams".

Crash windows handled
---------------------
* kill mid-write of the ~112 MB state file -> atomic tmp + os.replace, the
  previous state survives.
* kill between the val_stats.json append and the state save -> val_stats has
  one epoch more than the checkpoint; `truncate_val_stats` drops it, else the
  resumed run appends a duplicate epoch and epoch_saturation's block splitter
  discards the whole first run (the A1.21 item 111 pathology, reintroduced).
* an attempt launched WITHOUT --resume rotates the curve away (A1.21 item 111)
  -> `assert_contiguous` refuses to resume onto a curve that is not 1..E and
  prints the recovery hint instead of silently producing a 10..30 curve.

Selftest (`python resume_state.py --selftest`): a toy model + torch.optim.AdamW
+ a Dataset that draws augmentation from GLOBAL np.random through a 2-worker,
non-persistent DataLoader. Four epochs straight must equal two epochs + save +
a FRESH SUBPROCESS + load + two epochs: identical parameters, identical
per-epoch data fingerprints. Plus the truncate / contiguity / atomicity /
key-consumption guards.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

STATE_NAME = "ckpt_state.pt"
FORMAT = 1


# ------------------------------------------------------------------ RNG -----
def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# A1.27 item 152(c): under DDP every rank owns its own four streams. The
# single-GPU format keeps `rng` (flat, rank 0) so old files and old runs are
# byte-identical; multi-GPU adds `rng_ranks` = {rank: streams} gathered on
# rank 0 at save time, and each rank restores ITS OWN entry.
def gather_rng_all(rank: int, world_size: int) -> dict:
    """{rank: capture_rng()} for every rank (all_gather_object); {0: ...} when
    not distributed. Every rank must call this (it is a collective)."""
    mine = capture_rng()
    if world_size <= 1:
        return {0: mine}
    import torch.distributed as dist
    buf = [None] * world_size
    dist.all_gather_object(buf, mine)
    return {r: st for r, st in enumerate(buf)}


def rng_for_rank(st: dict, rank: int) -> dict:
    """The streams THIS rank must restore: rng_ranks[rank] when the file
    carries per-rank streams, else the flat (legacy / single-GPU) `rng`."""
    per = st.get("rng_ranks")
    if per:
        if rank not in per:
            raise SystemExit(f"resume: ckpt has RNG for ranks {sorted(per)}, "
                             f"not rank {rank} - world size changed?")
        return per[rank]
    return st["rng"]


# ------------------------------------------------------------ save / load ---
def save_state(path: Path, *, lora_sd: dict, optimizer_sd: dict,
               epoch_completed: int, best_val_loss: float, meta: dict,
               rng_ranks: dict = None) -> None:
    """Atomic: a kill mid-write leaves the previous file intact.
    rng_ranks = gather_rng_all(...) under DDP; None = single process."""
    path = Path(path)
    payload = {
        "format": FORMAT,
        "epoch_completed": int(epoch_completed),
        "best_val_loss": float(best_val_loss),
        "lora": {k: v.detach().cpu() for k, v in lora_sd.items()},
        "optimizer": optimizer_sd,
        "rng": (rng_ranks[0] if rng_ranks else capture_rng()),
        "meta": dict(meta),
    }
    if rng_ranks and len(rng_ranks) > 1:
        payload["rng_ranks"] = dict(rng_ranks)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_state(path: Path) -> dict:
    # weights_only=False: the numpy RNG state holds an ndarray, which the
    # torch>=2.7 default (weights_only=True) refuses to unpickle.
    st = torch.load(Path(path), map_location="cpu", weights_only=False)
    if st.get("format") != FORMAT:
        raise SystemExit(f"resume: {path} has format {st.get('format')}, "
                         f"expected {FORMAT}")
    return st


def check_meta(st: dict, expected: dict) -> None:
    """Refuse to resume a run onto a different fold / seed / output dir."""
    got = st.get("meta", {})
    bad = {k: (got.get(k), v) for k, v in expected.items()
           if k in got and got.get(k) != v}
    if bad:
        raise SystemExit("resume: checkpoint belongs to a different run - "
                         + ", ".join(f"{k}: ckpt={a!r} vs config={b!r}"
                                     for k, (a, b) in bad.items()))


# ------------------------------------------------------------ val_stats -----
def _read_records(path: Path):
    recs = []
    if not path.exists():
        return recs
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def truncate_val_stats(path: Path, epoch_completed: int) -> tuple[int, int]:
    """Keep only records with epoch <= epoch_completed. Returns (kept, dropped)."""
    path = Path(path)
    recs = _read_records(path)
    keep = [r for r in recs if int(r.get("epoch", 0)) <= epoch_completed]
    if len(keep) != len(recs):
        path.write_text("".join(json.dumps(r) + "\n" for r in keep),
                        encoding="utf-8")
    return len(keep), len(recs) - len(keep)


def assert_contiguous(path: Path, epoch_completed: int) -> None:
    """After truncation the curve must be exactly epochs 1..E."""
    epochs = [int(r.get("epoch", 0)) for r in _read_records(Path(path))]
    want = list(range(1, epoch_completed + 1))
    if epochs != want:
        raise SystemExit(
            f"resume: {path} holds epochs {epochs[:5]}..{epochs[-3:] if epochs else []} "
            f"but the checkpoint completed epoch {epoch_completed}; expected "
            f"1..{epoch_completed}. Likely an attempt was launched WITHOUT "
            f"--resume and rotated the curve to val_stats.prev*.json. Recovery: "
            f"restore the .prev file that ends at epoch {epoch_completed} back to "
            f"val_stats.json (and best_lora_weights.prev*.pt to "
            f"best_lora_weights.pt), then relaunch with --resume.")


# ------------------------------------------------------------ LoRA keys -----
def assert_lora_keys_consumed(sd_keys, model_lora_keys, unexpected) -> None:
    """model.load_state_dict(strict=False) discards nothing loudly. This makes
    a key drift (e.g. a changed lora: section) a hard failure instead of a
    silent 'loaded 0 tensors, printed success'."""
    sd_keys, model_lora_keys = set(sd_keys), set(model_lora_keys)
    if unexpected:
        raise SystemExit(f"resume: {len(unexpected)} checkpoint keys not in the "
                         f"model (lora: section changed?): {sorted(unexpected)[:3]}")
    if sd_keys != model_lora_keys:
        raise SystemExit(f"resume: checkpoint LoRA keys != model LoRA keys "
                         f"({len(sd_keys)} vs {len(model_lora_keys)})")


def fingerprint(t) -> float:
    """Exact, float-noise-immune identity of a batch: the sum of its raw image
    tensor BEFORE it moves to the device. Two runs on the same data with the
    same RNG streams produce the same number; any divergence in augmentation
    or ordering changes it."""
    return float(torch.as_tensor(t).double().sum().item())


# --------------------------------------------------------------- selftest --
class _ToyDS(torch.utils.data.Dataset):
    """Draws its 'augmentation' from the GLOBAL numpy RNG inside workers,
    exactly as tile_dataset does. Module-level so Windows spawn can pickle it."""
    def __len__(self):
        return 16

    def __getitem__(self, i):
        jitter = np.random.rand()                     # worker-side global draw
        return torch.tensor([float(i), jitter], dtype=torch.float32)


def _toy_run(n_epochs, start_epoch, out_dir: Path, resume: bool):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    best, fps = float("inf"), []
    state_f = out_dir / STATE_NAME
    stats_f = out_dir / "val_stats.json"
    if resume:
        st = load_state(state_f)
        model.load_state_dict(st["lora"], strict=True)
        opt.load_state_dict(st["optimizer"])
        best = st["best_val_loss"]
        start_epoch = st["epoch_completed"]
        truncate_val_stats(stats_f, start_epoch)
        assert_contiguous(stats_f, start_epoch)
        restore_rng(st["rng"])
    for epoch in range(start_epoch, n_epochs):
        dl = torch.utils.data.DataLoader(_ToyDS(), batch_size=4, shuffle=False,
                                         num_workers=2, persistent_workers=False)
        fp = None
        for step, x in enumerate(dl):               # iter() draws base_seed
            if step == 0:
                fp = fingerprint(x)
            loss = (model(x) ** 2).mean() + float(torch.rand(()))  # torch CPU draw
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = float(loss.item())
        best = min(best, val)
        with open(stats_f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": epoch + 1, "val_loss": val,
                                 "fingerprint": fp}) + "\n")
        fps.append(fp)
        save_state(state_f, lora_sd=model.state_dict(),
                   optimizer_sd=opt.state_dict(), epoch_completed=epoch + 1,
                   best_val_loss=best, meta={"seed": 0})
    return {k: v.detach().clone() for k, v in model.state_dict().items()}, fps


def _selftest() -> int:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root / "straight"
        b = root / "split"
        a.mkdir()
        b.mkdir()
        params_a, fps_a = _toy_run(4, 0, a, resume=False)
        _toy_run(2, 0, b, resume=False)
        # the resumed half runs in a FRESH process: nothing survives but files
        out = subprocess.run([sys.executable, __file__, "--_resume-half",
                              str(b)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        res = torch.load(b / "resumed_result.pt", weights_only=False)
        for k in params_a:
            assert torch.equal(params_a[k], res["params"][k]), (
                f"param {k} differs after resume")
        fps_b = [r["fingerprint"] for r in _read_records(b / "val_stats.json")]
        assert fps_b == fps_a, ("per-epoch data fingerprints differ: "
                                f"{fps_b} vs {fps_a}")
        assert len(fps_b) == 4 and fps_b[2] != fps_b[3] != fps_b[0], fps_b
        print(f"  exactness: 4 straight == 2 + save + subprocess + 2 "
              f"(params bitwise equal, fingerprints {[round(f, 3) for f in fps_a]})")

        # crash window: val_stats one epoch AHEAD of the checkpoint
        c = root / "crash"
        c.mkdir()
        _toy_run(3, 0, c, resume=False)
        with open(c / "val_stats.json", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"epoch": 4, "val_loss": 0.0}) + "\n")
        kept, dropped = truncate_val_stats(c / "val_stats.json", 3)
        assert (kept, dropped) == (3, 1), (kept, dropped)
        assert_contiguous(c / "val_stats.json", 3)
        print("  crash window: 1 orphan epoch truncated, curve 1..3 contiguous")

        # a rotated-away curve must be refused, with the hint
        (c / "val_stats.json").write_text(
            "".join(json.dumps({"epoch": e, "val_loss": 1.0}) + "\n"
                    for e in (2, 3)), encoding="utf-8")
        try:
            assert_contiguous(c / "val_stats.json", 3)
            raise AssertionError("non-contiguous curve was accepted")
        except SystemExit as e:
            assert "WITHOUT --resume" in str(e)
        print("  non-contiguous curve refused with recovery hint")

        # atomic save: simulate a kill mid-write - the old file must survive
        st_f = c / STATE_NAME
        before = st_f.read_bytes()
        tmp = st_f.with_suffix(".pt.tmp")
        tmp.write_bytes(b"garbage")            # a torn write that never replaced
        assert st_f.read_bytes() == before
        load_state(st_f)
        print("  atomic save: torn .tmp leaves the previous state loadable")

        # A1.27 item 152(c): per-rank RNG round-trip through save/load, and
        # the legacy flat file still restores on every rank
        r0, r1 = capture_rng(), None
        random.random(); np.random.rand(); torch.rand(1)     # advance -> differs
        r1 = capture_rng()
        assert r0["torch_cpu"].tolist() != r1["torch_cpu"].tolist()
        pr = c / "per_rank.pt"
        save_state(pr, lora_sd={}, optimizer_sd={}, epoch_completed=1,
                   best_val_loss=1.0, meta={}, rng_ranks={0: r0, 1: r1})
        st2 = load_state(pr)
        assert st2["rng_ranks"] and sorted(st2["rng_ranks"]) == [0, 1]
        assert rng_for_rank(st2, 1)["torch_cpu"].tolist() == r1["torch_cpu"].tolist()
        assert rng_for_rank(st2, 0)["torch_cpu"].tolist() == r0["torch_cpu"].tolist()
        assert st2["rng"]["torch_cpu"].tolist() == r0["torch_cpu"].tolist(), (
            "flat rng must stay rank 0 so single-GPU readers are unchanged")
        try:
            rng_for_rank(st2, 2)
            raise AssertionError("rank outside the saved set was accepted")
        except SystemExit:
            pass
        legacy = load_state(st_f)                    # written by _toy_run (1 proc)
        assert "rng_ranks" not in legacy
        assert rng_for_rank(legacy, 0) is legacy["rng"]
        assert rng_for_rank(legacy, 1) is legacy["rng"], "flat file serves every rank"
        assert gather_rng_all(0, 1).keys() == {0}
        print("  per-rank RNG: {0,1} round-trip, rank 0 == flat rng, unknown rank refused,"
              " legacy flat file restores on any rank")

        # meta + key guards
        try:
            check_meta({"meta": {"seed": 0}}, {"seed": 1})
            raise AssertionError("meta mismatch accepted")
        except SystemExit:
            pass
        try:
            assert_lora_keys_consumed({"a", "b"}, {"a", "b"}, unexpected=["x"])
            raise AssertionError("unexpected key accepted")
        except SystemExit:
            pass
        try:
            assert_lora_keys_consumed({"a"}, {"a", "b"}, unexpected=[])
            raise AssertionError("missing key accepted")
        except SystemExit:
            pass
        assert_lora_keys_consumed({"a", "b"}, {"a", "b"}, unexpected=[])
        print("  guards: wrong seed / unexpected key / missing key all refused")
    print("selftest PASS")
    return 0


def _resume_half(out_dir: str) -> int:
    params, _ = _toy_run(4, 0, Path(out_dir), resume=True)
    torch.save({"params": params}, Path(out_dir) / "resumed_result.pt")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--_resume-half" in sys.argv:
        raise SystemExit(_resume_half(sys.argv[sys.argv.index("--_resume-half") + 1]))
    print(__doc__)
