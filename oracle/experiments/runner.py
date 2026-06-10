"""Parallel experiment runner — farm run specs across GPU workers.

Each experiment module exposes `get_runs()` returning JSON specs; this script
collects them and executes with a multiprocessing pool (spawn — CUDA-safe).
The tiny grokking models use a small slice of a modern GPU, so several runs
share one device profitably; `sweep.execute` is idempotent (skips specs whose
result.json exists), so the whole sweep is resumable by re-running.

Usage (on the GPU box, from the repo root):
  python -m modular_addition.oracle.experiments.runner --dry-run
  python -m modular_addition.oracle.experiments.runner --workers 10
  python -m modular_addition.oracle.experiments.runner \
      --exps exp06 --workers 6 --gpus 0,1

Notes:
- exp00 (smoke) and exp03 (analysis-only, derived from exp01) contribute no
  runs. Big exp06 runs are scheduled first so the pool drains evenly.
- After training: run each experiment file (or just its summary cells) for
  summary.json, then experiments/make_figures.py for figures, then
  oracle/push_to_hf.py to archive checkpoints.
"""
import os

# Must precede any torch/cuBLAS init in this process and is inherited by
# spawned workers: required for CUDA determinism (helpers.set_seed enables
# torch.use_deterministic_algorithms).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib
import sys
import time
from pathlib import Path

try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

EXPERIMENTS = ["exp01_uptake", "exp02_1_delayed", "exp02_2_amplitude",
               "exp04_reliability", "exp05_answer_hint", "exp06_nfreqs"]
_PKG = "modular_addition.oracle.experiments"


def collect_specs(exp_filter=None):
    specs = []
    for mod_name in EXPERIMENTS:
        mod = importlib.import_module(f"{_PKG}.{mod_name}")
        for s in mod.get_runs():
            if exp_filter and s["exp"] not in exp_filter:
                continue
            specs.append(s)
    # big models first → better pool packing
    specs.sort(key=lambda s: -(s["p"] ** 2) * s["d_model"] * s["num_epochs"])
    return specs


def _worker(args):
    spec, device, use_wandb, force = args
    os.environ.setdefault("OMP_NUM_THREADS", "2")   # n workers share the CPUs
    from modular_addition.oracle import sweep       # import inside spawn
    t0 = time.time()
    try:
        res = sweep.execute(spec, device=device, use_wandb=use_wandb,
                            force=force, verbose=False)
        ge = res.get("grok_epoch")
        return (spec["exp"], spec["label"], "ok",
                f"grok={ge} {time.time() - t0:.0f}s")
    except Exception as e:  # noqa: BLE001 — one failed run must not kill the sweep
        return (spec["exp"], spec["label"], "FAIL", repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exps", nargs="*", default=None,
                    help="experiment keys to run (e.g. exp01 exp06); default all")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent training runs (per host, across GPUs)")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated CUDA indices to round-robin (e.g. 0,1)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run specs whose result.json already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs = collect_specs(set(args.exps) if args.exps else None)
    by_exp = {}
    for s in specs:
        by_exp[s["exp"]] = by_exp.get(s["exp"], 0) + 1
    print(f"{len(specs)} runs: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_exp.items())))
    if args.dry_run:
        from modular_addition.oracle import sweep
        done = sum(sweep.result_path(s).exists() for s in specs)
        print(f"already complete: {done}; remaining: {len(specs) - done}")
        return

    import torch
    if args.gpus:
        gpus = [f"cuda:{g}" for g in args.gpus.split(",")]
    elif torch.cuda.is_available():
        gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        gpus = [None]   # sweep.pick_device() decides (mps/cpu)
    jobs = [(s, gpus[i % len(gpus)], not args.no_wandb, args.force)
            for i, s in enumerate(specs)]

    import multiprocessing as mp
    from tqdm.auto import tqdm
    ctx = mp.get_context("spawn")
    t0, ok, fail = time.time(), 0, 0
    bar = tqdm(total=len(jobs), desc="sweep", unit="run", smoothing=0.05)
    with ctx.Pool(processes=args.workers) as pool:
        for exp, label, status, info in pool.imap_unordered(_worker, jobs):
            ok += status == "ok"
            fail += status != "ok"
            bar.update(1)
            bar.set_postfix(ok=ok, fail=fail)
            tqdm.write(f"[{ok + fail:>4}/{len(jobs)}] {exp}/{label}: {status} ({info})")
    bar.close()
    print(f"\ndone in {(time.time() - t0) / 3600:.2f}h — {ok} ok, {fail} failed")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
