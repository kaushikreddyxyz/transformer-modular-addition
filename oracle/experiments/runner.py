"""Parallel experiment runner — farm run specs across GPU workers.

Experiment modules are AUTO-DISCOVERED: any `experiments/exp*.py` that defines
`get_runs()` (returning JSON run specs) is collected — define a new experiment
following that pattern and the runner picks it up with no changes here.
(exp00 is excluded — it's the laptop smoke test, run it directly; cell-style
scripts without get_runs, e.g. exp10, are never imported.)

Each invocation writes to its OWN timestamped directory,
`oracle/results/run_<YYYYmmdd_HHMMSS>/` (checkpoints, JSONL, result.json,
summaries — everything), and refreshes the `oracle/results/latest` symlink,
which make_figures.py and push_to_hf.py follow by default. Because skipping
and the exp01 ↔ exp02 cross-references work per-directory, use
`--results-dir oracle/results/run_.../` to RESUME or EXTEND a previous run
(e.g. train exp01 now, add exp02_1 into the same dir tomorrow); a fresh
timestamped dir always starts from zero.

Execution: a multiprocessing pool (spawn — CUDA-safe). The tiny grokking
models use a small slice of a modern GPU, so several runs share one device
profitably (`--workers` processes round-robined over `--gpus`).

Usage (on the GPU box, from the repo root):
  python -m modular_addition.oracle.experiments.runner --dry-run
  python -m modular_addition.oracle.experiments.runner --workers 10
  python -m modular_addition.oracle.experiments.runner \
      --exps exp06 --workers 6 --gpus 0,1
  python -m modular_addition.oracle.experiments.runner \
      --results-dir modular_addition/oracle/results/run_20260610_120000

After training: run each experiment file's summary cells for summary.json,
then experiments/make_figures.py, then oracle/push_to_hf.py.
"""
import os

# Must precede any torch/cuBLAS init in this process and is inherited by
# spawned workers: required for CUDA determinism (helpers.set_seed enables
# torch.use_deterministic_algorithms).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Workers are long-lived and train many different model shapes back-to-back;
# expandable segments lets the caching allocator grow/shrink without
# fragmenting, which is what turns "plenty of free VRAM" into OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

_PKG = "modular_addition.oracle.experiments"
EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_BASE = EXPERIMENTS_DIR.parent / "results"


def discover_modules():
    """Importable experiment modules: exp*.py defining get_runs().

    The check is textual so that cell-style scripts that train at import time
    (no get_runs) are never imported. exp00 is the import-side-effect smoke
    test — always excluded, run it directly instead.
    """
    mods = []
    for p in sorted(EXPERIMENTS_DIR.glob("exp*.py")):
        if p.stem.startswith("exp00"):
            continue
        if "def get_runs(" not in p.read_text():
            continue
        mods.append(p.stem)
    return mods


def collect_specs(exp_filter=None):
    specs = []
    for mod_name in discover_modules():
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
        already = sweep.result_path(spec).exists() and not force
        res = sweep.execute(spec, device=device, use_wandb=use_wandb,
                            force=force, verbose=False)
        if already:
            return (spec["exp"], spec["label"], "skip", "result existed")
        ge = res.get("grok_epoch")
        return (spec["exp"], spec["label"], "trained",
                f"grok={ge} {time.time() - t0:.0f}s")
    except Exception as e:  # noqa: BLE001 — one failed run must not kill the sweep
        return (spec["exp"], spec["label"], "FAIL", repr(e))
    finally:
        # Hand freed blocks back to the driver between specs so concurrent
        # workers can use them (the allocator otherwise keeps this process's
        # peak reserved forever).
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _point_latest(run_dir):
    latest = RESULTS_BASE / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
    except OSError as e:   # e.g. exotic filesystems — purely a convenience link
        print(f"(could not update {latest}: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exps", nargs="*", default=None,
                    help="experiment keys to run (e.g. exp01 exp06); default all")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent training runs (per host, across GPUs)")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated CUDA indices to round-robin (e.g. 0,1)")
    ap.add_argument("--results-dir", default=None,
                    help="write into / resume this directory instead of a new "
                         "timestamped one (needed to extend a previous run)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run specs whose result.json already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_dir = (Path(args.results_dir) if args.results_dir
               else RESULTS_BASE / f"run_{datetime.now():%Y%m%d_%H%M%S}")
    # Workers (spawn) and our own sweep import both read this env var.
    os.environ["ORACLE_RESULTS_DIR"] = str(run_dir.resolve())

    specs = collect_specs(set(args.exps) if args.exps else None)
    by_exp = {}
    for s in specs:
        by_exp[s["exp"]] = by_exp.get(s["exp"], 0) + 1
    print(f"{len(specs)} models to train: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_exp.items())))
    print(f"results dir: {run_dir}"
          + ("" if args.results_dir else "  (new; --results-dir to resume an old one)"))

    from modular_addition.oracle import sweep
    done = sum(sweep.result_path(s).exists() for s in specs)
    if done:
        print(f"already in this dir: {done} (will be skipped; --force to redo)")
    if args.dry_run:
        print(f"dry run — nothing trained ({len(specs) - done} would train)")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    _point_latest(run_dir)

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
    t0 = time.time()
    counts = {"trained": 0, "skip": 0, "FAIL": 0}
    bar = tqdm(total=len(jobs), desc="models", unit="model", smoothing=0.05)
    with ctx.Pool(processes=args.workers) as pool:
        for exp, label, status, info in pool.imap_unordered(_worker, jobs):
            counts[status] += 1
            bar.update(1)
            bar.set_postfix(trained=counts["trained"], skipped=counts["skip"],
                            failed=counts["FAIL"])
            tqdm.write(f"[{bar.n:>4}/{len(jobs)}] {exp}/{label}: {status} ({info})")
    bar.close()
    print(f"\ndone in {(time.time() - t0) / 3600:.2f}h — "
          f"{counts['trained']} trained, {counts['skip']} skipped, "
          f"{counts['FAIL']} failed → {run_dir}")
    if counts["FAIL"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
