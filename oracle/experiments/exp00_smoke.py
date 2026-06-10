# %% [markdown]
# # Exp 00 — smoke test
# Validates the oracle library end-to-end: reproducibility, that injection changes
# the forward pass and the `inject` gate works, training speed, and that every
# analysis function runs. NOT a grokking run (few epochs).

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
import torch as t

from modular_addition import transformer
from modular_addition.oracle import inject, analysis, harness

device = t.device("cuda" if t.cuda.is_available() else "cpu")
cfg = dataclasses.replace(transformer.Config(), device=device, num_epochs=400, save_models=False)
print(f"device={cfg.device} p={cfg.p} d_model={cfg.d_model} d_mlp={cfg.d_mlp} frac_train={cfg.frac_train}")
FREQS = [17, 34]

# %% Test 1 — reproducibility: two baseline setups must be identical
m1, d1 = harness.setup(cfg, oracle_fn=None)
m2, d2 = harness.setup(cfg, oracle_fn=None)
we_match = t.allclose(m1.embed.W_E, m2.embed.W_E)
l1, _ = harness.loss_acc(m1, d1["train_x"], d1["train_y"], cfg)
l2, _ = harness.loss_acc(m2, d2["train_x"], d2["train_y"], cfg)
print(f"[repro] W_E identical={we_match}  train_loss {l1:.6f} vs {l2:.6f}  match={abs(l1-l2)<1e-9}")
assert we_match and abs(l1 - l2) < 1e-9, "reproducibility broken"

# %% Test 2 — oracle changes forward; inject gate + ablation identity
orc = inject.make_fourier_oracle(cfg, FREQS, amp=1.0)
mo, do = harness.setup(cfg, oracle_fn=orc)
x = do["train_x"][:8]
mo.inject = True;  on = mo(x)[:, -1]
mo.inject = False; off = mo(x)[:, -1]
mb, _ = harness.setup(cfg, oracle_fn=None)
print(f"[oracle] ||on-off||={(on-off).norm().item():.4f} (want >0)  "
      f"off==baseline={t.allclose(off, mb(x)[:, -1], atol=1e-5)}")
print(f"[oracle] per-token oracle norm={orc.table[0].norm().item():.3f} (=amp*sqrt(#freqs))")
assert (on - off).norm().item() > 1e-3

# %% Test 3 — short training baseline vs injected (mechanics + timing)
mb, db = harness.setup(cfg, oracle_fn=None)
res_b = harness.train(cfg, mb, db, num_epochs=cfg.num_epochs, eval_every=50,
                      snapshot_every=10_000, label="smoke_baseline", verbose=True)
mo, do = harness.setup(cfg, oracle_fn=inject.make_fourier_oracle(cfg, FREQS, amp=1.0))
res_o = harness.train(cfg, mo, do, num_epochs=cfg.num_epochs, eval_every=50,
                      snapshot_every=10_000, label="smoke_oracle", verbose=True)
print("baseline final:", {k: round(v, 4) for k, v in res_b["history"][-1].items() if isinstance(v, float)})
print("oracle   final:", {k: round(v, 4) for k, v in res_o["history"][-1].items() if isinstance(v, float)})
print(f"[speed] baseline ~{res_b['wall_s']/cfg.num_epochs*1000:.2f} ms/epoch")

# %% Test 4 — analysis functions all run on the (under-trained) injected model
ctx = analysis.metric_context(cfg, do["train_pairs"])
kf = analysis.key_freqs(mo, cfg, ctx)
we = analysis.we_fourier_power(mo.embed.W_E, cfg, ctx["fourier_basis"])
abl = analysis.ablation_ce(mo, do["test_x"], do["test_y"], cfg)
snap = analysis.uptake_snapshot(mo, cfg, ctx, injected_freqs=FREQS, data=do)
print(f"[analysis] #key_freqs={len(kf)}  we_gini={we['gini']:.3f}  we_norm={we['total_norm']:.2f}")
print(f"[analysis] ablation ΔCE={abl['delta']:.4f} (ce_on={abl['ce_on']:.3f} ce_off={abl['ce_off']:.3f})")
print(f"[analysis] snapshot keys: {list(snap.keys())}")
print(f"[analysis] injected_in_key_freqs={snap['injected_in_key_freqs']} "
      f"we_freq_power_injected={[round(v,3) for v in snap['we_freq_power_injected']]}")
print("\n✅ smoke test passed")
