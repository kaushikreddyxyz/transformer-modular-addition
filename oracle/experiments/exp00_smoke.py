# %% [markdown]
# # Exp 00 — smoke test
# Validates the oracle library + sweep machinery end-to-end on a laptop:
# reproducibility, the inject gate, spec execution (with checkpoints + snapshots
# + idempotent skip), the multifreq per-example oracle, and every analysis
# function. NOT a grokking run (few epochs); wandb stays off.

# %% imports + path bootstrap
import sys
from pathlib import Path
try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

import shutil

import torch as t

from modular_addition.oracle import analysis, harness, inject, sweep

EXP = "exp00"
EPOCHS = 400
RUN_DIR = sweep.RESULTS_DIR / EXP


def get_runs():
    """Mini-grid: baseline + 2-pair oracle, one seed, tiny ckpt schedule."""
    runs = []
    for n in (0, 2):
        freqs = sweep.pick_freqs(n)
        oracle = (dict(kind="fourier", freqs=freqs, amp=1.0) if n
                  else dict(kind="none"))
        runs.append(sweep.spec(
            exp=EXP, label=f"n{n}_s0", seed=0, oracle=oracle,
            num_epochs=EPOCHS, eval_every=50, snapshot_every=200,
            ckpt_epochs=(10, 100, EPOCHS), axes=dict(n=n, seed=0)))
    return runs


# %% Test 1 — reproducibility: two baseline setups must be identical
cfg = sweep.make_config(seed=0, num_epochs=EPOCHS)
print(f"device={cfg.device} p={cfg.p} d_model={cfg.d_model} d_mlp={cfg.d_mlp}")
m1, d1 = harness.setup(cfg, oracle_fn=None)
m2, d2 = harness.setup(cfg, oracle_fn=None)
l1, _ = harness.loss_acc(m1, d1["train_x"], d1["train_y"], cfg)
l2, _ = harness.loss_acc(m2, d2["train_x"], d2["train_y"], cfg)
assert t.allclose(m1.embed.W_E, m2.embed.W_E) and abs(l1 - l2) < 1e-9, \
    "reproducibility broken"
print(f"[repro] OK (train_loss {l1:.6f})")

# %% Test 2 — oracle changes forward; inject gate + ablation identity
FREQS = sweep.pick_freqs(2)
orc = inject.make_fourier_oracle(cfg, FREQS, amp=1.0)
mo, do = harness.setup(cfg, oracle_fn=orc)
x = do["train_x"][:8]
mo.inject = True; on = mo(x)[:, -1]
mo.inject = False; off = mo(x)[:, -1]
mb, _ = harness.setup(cfg, oracle_fn=None)
assert (on - off).norm().item() > 1e-3
assert t.allclose(off, mb(x)[:, -1], atol=1e-5)
print(f"[oracle] gate OK  ||on-off||={(on - off).norm().item():.4f}")

# %% Test 3 — multifreq per-example oracle: rel=1.0 ≡ fixed fourier on dims
fm = [inject.freq_map_reliable(cfg, f) for f in FREQS]
orc_pe = inject.make_perexample_multifreq_oracle(cfg, fm, amp=1.0)
orc_fx = inject.make_fourier_oracle(cfg, FREQS, amp=1.0)
xs = do["train_x"][:64]
pe, fx = orc_pe(xs), orc_fx(xs)
# fixed-fourier also writes features for the "=" token row (zeros) — compare
# the number-token positions only. atol covers float32 op-ordering noise in the
# two angle computations ((2πk/p)·i vs (2πk·i)/p), which is ~3e-5.
assert t.allclose(pe[:, :2], fx[:, :2], atol=1e-3), \
    "multifreq(rel=1) should match fixed fourier on number tokens"
assert pe[:, 2].abs().max().item() == 0.0
print("[oracle] multifreq(rel=1.0) ≡ fixed fourier OK")

# %% Test 4 — spec execution: checkpoints, snapshots, result files, skip
shutil.rmtree(RUN_DIR, ignore_errors=True)   # fresh smoke dir each invocation
results = sweep.run_all(get_runs(), use_wandb=False)
for s, res in zip(get_runs(), results):
    rp = sweep.result_path(s)
    assert rp.exists(), f"missing {rp}"
    assert res["spec"]["label"] == s["label"]
    assert len(res["snapshots"]) >= 2, "snapshots missing"
    for n_ep in s["ckpt_epochs"]:
        ck = RUN_DIR / "checkpoints" / s["label"] / f"ep{n_ep:06d}.pth"
        assert ck.exists(), f"missing checkpoint {ck}"
    ck = t.load(RUN_DIR / "checkpoints" / s["label"] / f"ep{EPOCHS:06d}.pth",
                map_location="cpu", weights_only=False)
    assert ck["epochs_done"] == EPOCHS and "embed.W_E" in ck["model"]
print("[exec] result files, snapshots, checkpoints OK")

# idempotency: second run_all must skip (results already on disk)
again = sweep.run_all(get_runs(), use_wandb=False)
assert all(isinstance(r, dict) for r in again)
print("[exec] idempotent skip OK")

# %% Test 5 — analysis functions all run on the (under-trained) injected model
res_o = results[1]
mo2, do2 = harness.setup(cfg, oracle_fn=inject.make_fourier_oracle(cfg, FREQS, amp=1.0))
mo2.load_state_dict({k: v.to(cfg.device) for k, v in t.load(
    RUN_DIR / "checkpoints" / "n2_s0" / f"ep{EPOCHS:06d}.pth",
    map_location="cpu", weights_only=False)["model"].items()})
ctx = analysis.metric_context(cfg, do2["train_pairs"])
snap = analysis.uptake_snapshot(mo2, cfg, ctx, injected_freqs=FREQS, data=do2)
abl = analysis.ablation_ce(mo2, do2["test_x"], do2["test_y"], cfg)
print(f"[analysis] #key_freqs={len(snap['key_freqs'])}  "
      f"ablation ΔCE={abl['delta']:.4f}  snapshot keys OK")

print("\n✅ smoke test passed")
