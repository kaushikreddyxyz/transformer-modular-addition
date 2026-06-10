# %% [markdown]
# # Exp 05 — Weakly-informative answer hint
# Inject a weak feature about the answer c=(i+j) mod p at the "=" position
# (c % 10 or c // 10). Does the model then solve the task with fewer frequencies?
# Compare #key_freqs (and uptake) of hinted models vs a no-hint baseline.

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
RUN_DIR = f"{_root}/modular_addition/oracle/results/exp05"
os.makedirs(RUN_DIR, exist_ok=True)
cfg = dataclasses.replace(transformer.Config(), device=device, num_epochs=NUM_EPOCHS, save_models=False)

_, dshared = harness.setup(cfg, oracle_fn=None)
ctx = analysis.metric_context(cfg, dshared["train_pairs"])
# no injected Fourier freqs here; we only care about #key_freqs the model uses
snap_fn = lambda data: (lambda model, epoch: analysis.uptake_snapshot(
    model, cfg, ctx, injected_freqs=[], data=data))

CONFIGS = [
    ("baseline", None),
    ("hint_mod10_onehot", inject.make_answer_hint_oracle(cfg, hint="mod", modulus=10, amp=1.0, code="onehot")),
    ("hint_div10_onehot", inject.make_answer_hint_oracle(cfg, hint="div", modulus=10, amp=1.0, code="onehot")),
    ("hint_mod10_fourier", inject.make_answer_hint_oracle(cfg, hint="mod", modulus=10, amp=1.0, code="fourier")),
]

records = {}
for name, orc in CONFIGS:
    m, d = harness.setup(cfg, oracle_fn=orc)
    res = harness.train(cfg, m, d, num_epochs=NUM_EPOCHS, eval_every=200, snapshot_every=1000,
                        snapshot_fn=snap_fn(d), run_dir=RUN_DIR, label=name)
    s = res["snapshots"][-1]
    abl = s.get("ablation_test") or {}
    rec = dict(name=name, grok_epoch=res["grok_epoch"],
               final_test_acc=round(res["history"][-1]["test_acc"], 4),
               n_key_freqs=len(s["key_freqs"]), key_freqs=s["key_freqs"],
               we_total_norm=round(s["we_total_norm"], 3),
               ablation_delta=round(abl.get("delta", float("nan")), 4) if orc is not None else None)
    records[name] = rec
    print(name, rec)

with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(records, f, indent=2)

print("\n=== Exp E (answer hint) summary ===")
print("config | grok | testacc | #key_freqs | |W_E| | abl ΔCE")
for name, _ in CONFIGS:
    r = records[name]
    print(f"  {name:<20} | {str(r['grok_epoch']):>5} | {r['final_test_acc']:.3f} | "
          f"{r['n_key_freqs']:>3} | {r['we_total_norm']:>6} | {r['ablation_delta']}")
print(f"\nHypothesis: hinted models use FEWER key frequencies than baseline "
      f"({records['baseline']['n_key_freqs']}).")
print("\n✅ exp05 done")
