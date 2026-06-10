# %% [markdown]
# # Exp 02 — W_E laziness + delayed injection
# (A) Amplitude sweep: as the oracle gets louder, does the trainable W_E offload
#     work onto it (lower norm / lower Fourier power at the injected freqs)?
# (B) Delayed injection: turn the oracle on only at epoch T. Does the model abandon
#     the embedding structure it already built, and still adopt the oracle?

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
FREQS = [17, 34]
RUN_DIR = f"{_root}/modular_addition/oracle/results/exp02"
os.makedirs(RUN_DIR, exist_ok=True)
cfg = dataclasses.replace(transformer.Config(), device=device, num_epochs=NUM_EPOCHS, save_models=False)

_, dshared = harness.setup(cfg, oracle_fn=None)
ctx = analysis.metric_context(cfg, dshared["train_pairs"])

def snap_fn(data):
    return lambda model, epoch: analysis.uptake_snapshot(model, cfg, ctx, injected_freqs=FREQS, data=data)

def rec_of(res):
    s = res["snapshots"][-1]
    abl = s.get("ablation_test") or {}
    return dict(label=res["label"], grok_epoch=res["grok_epoch"],
                we_total_norm=round(s["we_total_norm"], 3), we_gini=round(s["we_gini"], 3),
                we_power_injected=round(float(sum(s["we_freq_power_injected"])), 3),
                key_freqs=s["key_freqs"], injected_in_key=s["injected_in_key_freqs"],
                ablation_delta=round(abl.get("delta", float("nan")), 4),
                final_test_acc=round(res["history"][-1]["test_acc"], 4))

records = {}

# %% (A) amplitude sweep
for amp in [0.5, 1.0, 2.0, 4.0]:
    orc = inject.make_fourier_oracle(cfg, FREQS, amp=amp)
    m, d = harness.setup(cfg, oracle_fn=orc)
    res = harness.train(cfg, m, d, num_epochs=NUM_EPOCHS, eval_every=200, snapshot_every=1000,
                        snapshot_fn=snap_fn(d), run_dir=RUN_DIR, label=f"amp{amp}")
    records[f"amp{amp}"] = rec_of(res)
    print("AMP", amp, records[f"amp{amp}"])

# %% (B) delayed injection sweep (amp=1.0); T=0 is the amp1.0 run above
for T in [4000, 8000]:
    orc = inject.make_fourier_oracle(cfg, FREQS, amp=1.0)
    m, d = harness.setup(cfg, oracle_fn=orc)
    res = harness.train(cfg, m, d, num_epochs=NUM_EPOCHS, eval_every=200, snapshot_every=500,
                        snapshot_fn=snap_fn(d), run_dir=RUN_DIR, label=f"delay{T}",
                        inject_from_epoch=T)
    # W_E power at injected freqs just-before vs just-after injection turns on
    snaps = res["snapshots"]
    before = next((s for s in reversed(snaps) if s["epoch"] < T), None)
    after = next((s for s in snaps if s["epoch"] >= T), None)
    rec = rec_of(res)
    rec["we_power_injected_before_T"] = round(float(sum(before["we_freq_power_injected"])), 3) if before else None
    rec["we_power_injected_after_T"] = round(float(sum(after["we_freq_power_injected"])), 3) if after else None
    rec["key_freqs_before_T"] = before["key_freqs"] if before else None
    records[f"delay{T}"] = rec
    print("DELAY", T, rec)

# %% summary
with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(records, f, indent=2)
print("\n=== Exp B summary ===")
print("amplitude sweep (expect we_power_injected to DROP as amp rises if model goes lazy):")
for amp in [0.5, 1.0, 2.0, 4.0]:
    r = records[f"amp{amp}"]
    print(f"  amp={amp}: |W_E|={r['we_total_norm']} we_power@inj={r['we_power_injected']} "
          f"grok={r['grok_epoch']} ablΔ={r['ablation_delta']} key_freqs={r['key_freqs']}")
print("delayed injection (does it adopt after T, and was pre-T structure abandoned?):")
for T in [4000, 8000]:
    r = records[f"delay{T}"]
    print(f"  T={T}: grok={r['grok_epoch']} injected_in_key={r['injected_in_key']} "
          f"we_power@inj before_T={r['we_power_injected_before_T']} after/end={r['we_power_injected']}")
print("\n✅ exp02 done")
