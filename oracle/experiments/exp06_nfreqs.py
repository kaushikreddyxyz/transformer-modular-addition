# %% [markdown]
# # Exp 06 — How many injected frequencies? (completeness / speed)
# Exp 01 showed a 2-frequency oracle is *used* but the model plateaus below full
# grokking. Here we sweep the NUMBER of injected frequencies to ask: does a more
# complete oracle basis let the model grok fully and faster? This is the real test
# of the "faster grokking with oracle features" hypothesis.

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
import torch as t

from modular_addition import transformer
from modular_addition.oracle import inject, analysis, harness

device = t.device("cuda" if t.cuda.is_available() else "cpu")
NUM_EPOCHS = 30_000
AMP = 1.0
FREQ_POOL = [17, 34, 9, 25, 43, 50, 13, 47]   # first 2 == Exp01's [17,34]
N_LIST = [1, 2, 3, 5, 8]
RUN_DIR = f"{_root}/modular_addition/oracle/results/exp06"
os.makedirs(RUN_DIR, exist_ok=True)
cfg = dataclasses.replace(transformer.Config(), device=device, num_epochs=NUM_EPOCHS, save_models=False)

_, dshared = harness.setup(cfg, oracle_fn=None)
ctx = analysis.metric_context(cfg, dshared["train_pairs"])

def rec_of(res, freqs):
    s = res["snapshots"][-1]
    abl = s.get("ablation_test") or {}
    return dict(n_inject=len(freqs), freqs=list(freqs), grok_epoch=res["grok_epoch"],
                final_test_acc=round(res["history"][-1]["test_acc"], 4),
                n_key_freqs=len(s["key_freqs"]), key_freqs=s["key_freqs"],
                injected_in_key=s["injected_in_key_freqs"],
                ablation_delta=round(abl.get("delta", float("nan")), 4) if freqs else None,
                we_total_norm=round(s["we_total_norm"], 3),
                we_power_injected=round(float(sum(s["we_freq_power_injected"])), 3) if freqs else 0.0)

records = {}

# %% baseline (n=0)
mb, db = harness.setup(cfg, oracle_fn=None)
rb = harness.train(cfg, mb, db, num_epochs=NUM_EPOCHS, eval_every=200, snapshot_every=2000,
                   snapshot_fn=lambda m, e: analysis.uptake_snapshot(m, cfg, ctx, injected_freqs=[], data=db),
                   run_dir=RUN_DIR, label="nfreq0_baseline", stop_after_grok=1000)
records["0"] = rec_of(rb, [])
print("N=0", records["0"])

# %% sweep number of injected frequencies
for n in N_LIST:
    freqs = FREQ_POOL[:n]
    orc = inject.make_fourier_oracle(cfg, freqs, amp=AMP)
    m, d = harness.setup(cfg, oracle_fn=orc)
    snap_fn = lambda model, epoch, _f=freqs, _d=d: analysis.uptake_snapshot(
        model, cfg, ctx, injected_freqs=_f, data=_d)
    res = harness.train(cfg, m, d, num_epochs=NUM_EPOCHS, eval_every=200, snapshot_every=2000,
                        snapshot_fn=snap_fn, run_dir=RUN_DIR, label=f"nfreq{n}", stop_after_grok=1000)
    records[str(n)] = rec_of(res, freqs)
    print(f"N={n}", records[str(n)])

# %% summary
with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(records, f, indent=2)
print("\n=== Exp 06 (n injected freqs) summary ===")
print("n_inj | grok_epoch | test_acc | #key_freqs | injected∈key | abl ΔCE | |W_E| | W_E pow@inj")
for k in ["0"] + [str(n) for n in N_LIST]:
    r = records[k]
    print(f"  {r['n_inject']:>4} | {str(r['grok_epoch']):>9} | {r['final_test_acc']:.3f} | "
          f"{r['n_key_freqs']:>3} | {str(len(r['injected_in_key']))+'/'+str(r['n_inject']):>6} | "
          f"{str(r['ablation_delta']):>7} | {r['we_total_norm']:>6} | {r['we_power_injected']:>7}")
print("\n✅ exp06 done")
