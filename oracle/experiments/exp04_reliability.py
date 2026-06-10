# %% [markdown]
# # Exp 04 — Unreliable / variable-frequency oracle
# The oracle's frequency varies per example: with prob `reliability` it is the true
# base frequency, else a random frequency. Sweep reliability and watch the progress
# measures + W_E: where is the threshold past which the model disregards the oracle
# and rebuilds its own embedding?

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
NUM_EPOCHS = 22_000
BASE_FREQ = 17
AMP = 1.0
RELIABILITIES = [1.0, 0.75, 0.5, 0.25, 0.0]
RUN_DIR = f"{_root}/modular_addition/oracle/results/exp04"
os.makedirs(RUN_DIR, exist_ok=True)
cfg = dataclasses.replace(transformer.Config(), device=device, num_epochs=NUM_EPOCHS, save_models=False)

_, dshared = harness.setup(cfg, oracle_fn=None)
ctx = analysis.metric_context(cfg, dshared["train_pairs"])

records = {}
for rel in RELIABILITIES:
    fm = inject.freq_map_corrupt(cfg, BASE_FREQ, reliability=rel, seed=0)
    orc = inject.make_perexample_freq_oracle(cfg, fm, amp=AMP)
    m, d = harness.setup(cfg, oracle_fn=orc)
    snap_fn = lambda model, epoch: analysis.uptake_snapshot(
        model, cfg, ctx, injected_freqs=[BASE_FREQ], data=d)
    res = harness.train(cfg, m, d, num_epochs=NUM_EPOCHS, eval_every=200, snapshot_every=1000,
                        snapshot_fn=snap_fn, run_dir=RUN_DIR, label=f"rel{rel}",
                        stop_after_grok=2000)
    s = res["snapshots"][-1]
    abl = s.get("ablation_test") or {}
    # W_E power on base freq vs the strongest "other" frequency
    full = s["we_freq_power_full"]
    other = max((p for k, p in enumerate(full, start=1) if k != BASE_FREQ), default=0.0)
    rec = dict(reliability=rel, grok_epoch=res["grok_epoch"],
               final_test_acc=round(res["history"][-1]["test_acc"], 4),
               key_freqs=s["key_freqs"], base_in_key=(BASE_FREQ in s["key_freqs"]),
               ablation_delta=round(abl.get("delta", float("nan")), 4),
               we_power_base=round(float(full[BASE_FREQ - 1]), 3),
               we_power_other_max=round(float(other), 3),
               we_total_norm=round(s["we_total_norm"], 3))
    records[f"rel{rel}"] = rec
    print("REL", rel, rec)

with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(records, f, indent=2)

print("\n=== Exp D (reliability) summary ===")
print("reliability | grok | testacc | abl ΔCE | base∈key | W_E pow@base | W_E pow@other")
for rel in RELIABILITIES:
    r = records[f"rel{rel}"]
    print(f"  {rel:>4} | {str(r['grok_epoch']):>5} | {r['final_test_acc']:.3f} | {r['ablation_delta']:>7} | "
          f"{str(r['base_in_key']):>5} | {r['we_power_base']:>7} | {r['we_power_other_max']:>7}")
print("\n✅ exp04 done")
