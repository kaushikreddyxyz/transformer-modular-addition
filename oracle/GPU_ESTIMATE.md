# GPU sizing for the oracle experiment suite

Estimates for running every experiment (exp01, exp02.1, exp02.2, exp04, exp05,
exp06) at full grid via `experiments/runner.py`. exp00 is a laptop smoke test;
exp03 trains nothing (derived from exp01's grid).

## Workload

| | runs | model | params | train tokens/epoch | FLOPs/run (30k ep) |
|---|---|---|---|---|---|
| exp01 | 28 | p=113, d=128 | ~226k (0.9 MB) | ~11.5k | ~0.5 PFLOP |
| exp02.1 | 48 | p=113, d=128 | " | " | " |
| exp02.2 | 96 | p=113, d=128 | " | " | " |
| exp04 | 120 | p=113, d=128 | " | " | " |
| exp05 | 16 | p=113, d=128 | " | " | " |
| exp06 | 48 | p=211, d=256 | ~896k (3.6 MB) | ~40k | ~6.4 PFLOP (~13× small) |

(FLOPs ≈ 6 · params · tokens · epochs; full-batch GD on the 30% train split.
Uptake snapshots every 2k epochs add ~30–60 s per run — full-grid forwards +
DFTs, included in the wall-time figures below.)

- **Small run (p=113):** ~2–3 min on a 4090-class GPU (matches the measured
  "30k epochs well under two minutes" plus snapshot overhead).
- **Big run (exp06):** ~8–15 min (13× FLOPs, but larger matmuls utilize the
  GPU better).

**Sequential total: ~22 GPU-hours** (308 small ≈ 13 h + 48 big ≈ 9 h).

## Why parallelism is nearly free here

These models occupy a tiny fraction of a modern GPU's SMs and memory; a single
run is bound by kernel-launch overhead, not compute. Independent processes
(the runner uses spawn — one CUDA context each) interleave kernels and scale
close to linearly until SM occupancy or VRAM runs out.

Per-worker VRAM (weights + grads + Adam states + full-batch activations;
uptake snapshots add a transient that is freed back to the driver afterwards):

- small run: ~0.4–0.6 GB
- exp06 run: ~1 GB steady, ~2–2.5 GB briefly during a snapshot

Do not remove the memory hygiene that makes these numbers true: snapshots run
under `no_grad` with chunked grid forwards and a targeted MLP hook
(`analysis._mlp_post_acts`), logit coefficients are computed per-frequency
(`analysis._coefficients_lowmem` — the upstream vectorized version
materializes a (p//2, p, p, p) cube, ~8 GB at p=211, which OOM'd multi-worker
sweeps), `empty_cache` runs after snapshots/specs, and the runner sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Reading nvidia-smi: "GPU-Util" is the fraction of time *any* kernel is
resident, not SM occupancy — one tiny run can show ~99% while using a sliver
of the chip (that's why concurrent workers still scale). Long ~30% stretches
are snapshot windows (launch-bound analysis ops). VRAM% reflects what the
caching allocators have *reserved*, which sits above live usage.

## Recommended specs

**One 24 GB GPU (RTX 4090 / A10G / L4), ≥12 physical cores, 32 GB RAM, ~20 GB
free disk.**

- `--workers 10` for the small experiments → ~1.5 h
- `--workers 6` for exp06 → ~1.5–2 h
- **Everything: ~3.5–4 h wall**, e.g.
  `runner --exps exp01 exp02_1 exp02_2 exp04 exp05 --workers 10` then
  `runner --exps exp06 --workers 6` (or one invocation with `--workers 8`;
  big runs are scheduled first automatically).

Faster options:
- **A100 80 GB / H100:** `--workers 12–16` → ~2–2.5 h total. Helps mostly via
  more concurrent workers, not per-run speed.
- **2× 4090 with `--gpus 0,1 --workers 16`** → ~2 h.

CPU matters: each worker burns ~1–2 cores on Python/launch overhead
(`OMP_NUM_THREADS=2` is set per worker). Don't run 10 workers on 4 vCPUs.

## Output & progress

Every runner invocation creates its own
`modular_addition/oracle/results/run_<YYYYmmdd_HHMMSS>/` containing everything
that invocation produced: per-run JSONL + result.json, `checkpoints/`,
per-experiment `summary.json`, and `figures/` once make_figures runs. A
`results/latest` symlink always points at the newest run — summary cells,
make_figures, and any sweep code follow it automatically, so "train → summarize
→ plot" needs no path plumbing. Run the runner dozens of times; each run is
its own folder. Use `--results-dir results/run_<ts>` to resume/extend a
previous run (skip-if-done and the exp01↔exp02 cross-references are
per-directory), and `ORACLE_RESULTS_DIR=/scratch/...` to redirect entirely.
Pre-sweep single-seed data is archived in `results_legacy/`. push_to_hf
defaults to the whole `results/` base, so all stamped runs accumulate in the
HF repo under their `run_<ts>/` prefixes (already-pushed files are skipped).

Progress: the runner prints the model count per experiment up front, then a
tqdm bar over all models (ETA + trained/skipped/failed counts) with one
completion line per model — equally informative for 1 model or 356. Per-epoch
bars appear in sequential/verbose mode (experiment files run directly). wandb
charts update live per run either way; its console banner/summary is silenced
(harness sets `WANDB_SILENT=true` — chatter goes to `wandb/run-*/logs/debug.log`,
verbose mode still prints each run's URL; export `WANDB_SILENT=false` to undo).

## Other resources

- **Disk:** checkpoints ≈ 9 × 1 MB × 308 + 9 × 3.6 MB × 48 ≈ **4.5 GB**, plus
  JSONL/result files (~1 GB) and wandb cache.
- **wandb:** ~356 runs in project `oracle-encodings`, grouped per experiment.
- **Determinism:** the runner exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
  torch loads and `helpers.set_seed` enables deterministic algorithms, so a
  given spec reproduces bit-identically on the same hardware/stack.
- Resumability: re-running the runner skips any spec whose result.json exists
  (`--force` to override), so spot/preemptible instances are fine.

## Not done (deliberately)

- `torch.compile` could plausibly 2–4× the small runs but adds warmup,
  version fragility, and (historically) nondeterminism foot-guns — revisit if
  the suite grows another order of magnitude.
- CUDA MPS / green contexts: unnecessary at these sizes; plain process
  parallelism already saturates the GPU.
