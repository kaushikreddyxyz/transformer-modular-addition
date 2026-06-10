# %% [markdown]
# # Exp 01 — Injection uptake (baseline vs oracle)
# Does the model *use* injected frequencies? Train a baseline and an injected model
# (frozen Fourier oracle at freqs [17,34]) to full grokking; at each snapshot record
# the uptake metrics (key_freqs, excluded/trig loss, ablation ΔCE, W_E spectrum).
# Also serves as the W_E-trajectory source for Exp 02.

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
AMP = 1.0
RUN_DIR = f"{_root}/modular_addition/oracle/results/exp01"
os.makedirs(RUN_DIR, exist_ok=True)
cfg = dataclasses.replace(transformer.Config(), device=device, num_epochs=NUM_EPOCHS, save_models=False)
print(f"Exp01 uptake | epochs={NUM_EPOCHS} freqs={FREQS} amp={AMP} dir={RUN_DIR}")

# %% shared metric context (baseline & oracle share the same seed => same train split)
_, dshared = harness.setup(cfg, oracle_fn=None)
ctx = analysis.metric_context(cfg, dshared["train_pairs"])

def make_snap(data):
    def snap(model, epoch):
        return analysis.uptake_snapshot(model, cfg, ctx, injected_freqs=FREQS, data=data)
    return snap

# %% baseline run
mb, db = harness.setup(cfg, oracle_fn=None)
res_b = harness.train(cfg, mb, db, num_epochs=NUM_EPOCHS, eval_every=200, snapshot_every=1000,
                      snapshot_fn=make_snap(db), run_dir=RUN_DIR, label="baseline")

# %% oracle run (inject true Fourier features at FREQS)
orc = inject.make_fourier_oracle(cfg, FREQS, amp=AMP)
mo, do = harness.setup(cfg, oracle_fn=orc)
res_o = harness.train(cfg, mo, do, num_epochs=NUM_EPOCHS, eval_every=200, snapshot_every=1000,
                      snapshot_fn=make_snap(do), run_dir=RUN_DIR,
                      label=f"oracle_f{'_'.join(map(str, FREQS))}_amp{AMP}")

# %% summary
fo = res_o["snapshots"][-1]
fb = res_b["snapshots"][-1]
summary = dict(
    freqs=FREQS, amp=AMP, num_epochs=NUM_EPOCHS,
    baseline_grok_epoch=res_b["grok_epoch"], oracle_grok_epoch=res_o["grok_epoch"],
    baseline_final_test_acc=res_b["history"][-1]["test_acc"],
    oracle_final_test_acc=res_o["history"][-1]["test_acc"],
    baseline_key_freqs=fb["key_freqs"], oracle_key_freqs=fo["key_freqs"],
    oracle_injected_in_key_freqs=fo["injected_in_key_freqs"],
    oracle_ablation_test_delta=fo["ablation_test"]["delta"],
    oracle_excluded_loss_injected=fo["excluded_loss_injected"],
    oracle_trig_loss_injected=fo["trig_loss_injected"],
    oracle_we_freq_power_injected=fo["we_freq_power_injected"],
    baseline_we_freq_power_injected=fb["we_freq_power_injected"],
    oracle_we_total_norm=fo["we_total_norm"], baseline_we_total_norm=fb["we_total_norm"],
)
with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Exp A (uptake) summary ===")
print(f"grok_epoch: baseline={summary['baseline_grok_epoch']}  oracle={summary['oracle_grok_epoch']}")
print(f"oracle injected_in_key_freqs={summary['oracle_injected_in_key_freqs']} (injected {FREQS})")
print(f"oracle ablation test ΔCE={summary['oracle_ablation_test_delta']:.4f}")
print(f"oracle excluded_loss@injected={[round(x,3) for x in summary['oracle_excluded_loss_injected']]}")
print(f"oracle trig_loss@injected={summary['oracle_trig_loss_injected']:.4f}")
print(f"W_E |.|: baseline={summary['baseline_we_total_norm']:.2f}  oracle={summary['oracle_we_total_norm']:.2f}")
print(f"W_E power@injected: baseline={[round(x,2) for x in summary['baseline_we_freq_power_injected']]}  "
      f"oracle={[round(x,2) for x in summary['oracle_we_freq_power_injected']]}")
print("\n✅ exp01 done")
