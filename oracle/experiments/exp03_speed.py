# %% [markdown]
# # Exp 03 — Grokking speed with oracle features
# Hypothesis: injecting the right Fourier features lets the model skip building its
# own embedding circuit, so it groks earlier. Compare grok_epoch (first epoch with
# test_acc >= 0.99) for baseline vs oracle across seeds.

# %% imports + path bootstrap
import sys
from pathlib import Path
try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

import dataclasses
import json
import os
import numpy as np
import torch as t

from modular_addition import transformer
from modular_addition.oracle import inject, harness

device = t.device("cuda" if t.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2]
FREQS = [17, 34]
AMP = 1.0
RUN_DIR = f"{_root}/modular_addition/oracle/results/exp03"
os.makedirs(RUN_DIR, exist_ok=True)

rows = []
for seed in SEEDS:
    cfg = dataclasses.replace(transformer.Config(), device=device, seed=seed, save_models=False)
    # baseline (no snapshots -> fast; stop shortly after grok)
    mb, db = harness.setup(cfg, oracle_fn=None)
    rb = harness.train(cfg, mb, db, num_epochs=30_000, eval_every=200, snapshot_every=10**9,
                       run_dir=RUN_DIR, label=f"baseline_s{seed}")
    # oracle
    mo, do = harness.setup(cfg, oracle_fn=inject.make_fourier_oracle(cfg, FREQS, amp=AMP))
    ro = harness.train(cfg, mo, do, num_epochs=30_000, eval_every=200, snapshot_every=10**9,
                       run_dir=RUN_DIR, label=f"oracle_s{seed}")
    row = dict(seed=seed, baseline_grok=rb["grok_epoch"], oracle_grok=ro["grok_epoch"])
    row["speedup"] = (rb["grok_epoch"] / ro["grok_epoch"]
                      if rb["grok_epoch"] and ro["grok_epoch"] else None)
    rows.append(row)
    print("SEED", seed, row)

def _stats(xs):
    xs = [x for x in xs if x is not None]
    return dict(mean=float(np.mean(xs)), std=float(np.std(xs)), vals=xs) if xs else None

summary = dict(seeds=SEEDS, freqs=FREQS, amp=AMP, rows=rows,
               baseline_grok=_stats([r["baseline_grok"] for r in rows]),
               oracle_grok=_stats([r["oracle_grok"] for r in rows]),
               speedup=_stats([r["speedup"] for r in rows]))
with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Exp C (speed) summary ===")
for r in rows:
    print(f"  seed {r['seed']}: baseline grok @ {r['baseline_grok']}  oracle grok @ {r['oracle_grok']}  "
          f"speedup x{r['speedup']:.2f}" if r["speedup"] else f"  seed {r['seed']}: {r}")
print(f"  baseline grok: {summary['baseline_grok']}")
print(f"  oracle   grok: {summary['oracle_grok']}")
print(f"  speedup: {summary['speedup']}")
print("\n✅ exp03 done")
