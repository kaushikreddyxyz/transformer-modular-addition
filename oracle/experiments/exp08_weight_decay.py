# %% [markdown]
# # Exp 08 — Weight decay vs oracle-assisted grokking (p=113, n=5)
# Grokking on modular addition is classically *driven* by weight decay (the Nanda
# recipe uses wd=1.0, this project's Config default). Here we fix n=5 injected
# Fourier pairs and a trainable W_E, then sweep weight decay across the LOW regime
# (1e-4 … 1e-2, all far below the 1.0 default) over 4 seeds. Question: does a
# strong injected oracle make generalization robust when the weight-decay pressure
# that normally produces grokking is weak — and where is the wd floor below which
# even the oracle can't get the model to grok within the budget?
#
# Early stopping: halt STOP_AFTER_GROK epochs after the first epoch with
# test_acc >= GROK_ACC (grokked model checkpointed at the stop epoch); 25k-epoch
# hard cap for the low-wd runs that may never grok. frac_train=0.3, d_model=128.

# %% imports + path bootstrap
import sys
from pathlib import Path
try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

from modular_addition.oracle import sweep

EXP = "exp08"
P = 113                       # d_vocab = 114
D_MODEL = 128
AMP = 1.0
N_FIXED = 5                   # injected frequency pairs (held constant)
# Half-decade log sweep 1e-4 → 1e-2. NOTE: the directive wrote
# "[10-4, 50*10-4, 10-3, 50*10-3, 10-2]"; read as 1,5,1,5,1 per decade (i.e.
# 50*10-4 = 5.0e-4). If instead the literal 50×10^-4 = 5e-3 was meant, the set is
# {1e-4, 1e-3, 5e-3, 1e-2, 5e-2} — change this one list to switch.
WEIGHT_DECAYS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
NUM_EPOCHS = 25_000           # hard cap for runs that never grok
STOP_AFTER_GROK = 1_000       # early-stop budget past the grok epoch
GROK_ACC = 0.99
CKPT_EPOCHS = (10, 100, 500, 1000, 2500, 5000, 7500,
               10_000, 15_000, 20_000, 25_000)

FREQS = sweep.pick_freqs(N_FIXED, p=P)   # canonical p=113 pool prefix


def get_runs():
    runs = []
    for wd in WEIGHT_DECAYS:
        oracle = dict(kind="fourier", freqs=FREQS, amp=AMP)
        for s in sweep.SEEDS:
            runs.append(sweep.spec(
                exp=EXP, label=f"wd{wd:g}_s{s}", seed=s, oracle=oracle,
                p=P, d_model=D_MODEL, num_epochs=NUM_EPOCHS,
                ckpt_epochs=CKPT_EPOCHS,
                config=dict(weight_decay=wd),
                stop_after_grok=STOP_AFTER_GROK, grok_acc=GROK_ACC,
                axes=dict(weight_decay=wd, seed=s, n=N_FIXED, amp=AMP,
                          freqs=FREQS, p=P, d_model=D_MODEL)))
    return runs


# %% run (sequential; use experiments/runner.py to parallelize)
if __name__ == "__main__" or "ipykernel" in sys.modules:
    results = sweep.run_all(get_runs())

    # %% summary — aggregate across seeds, report per weight decay
    recs = [sweep.final_record(r) for r in results]
    agg = sweep.mean_std(
        recs, keys=["grok_epoch", "final_test_acc", "ablation_delta",
                    "we_power_injected", "n_key_freqs", "injected_in_key"],
        group_keys=["ax_weight_decay"])
    sweep.write_summary(EXP, dict(
        grid=dict(p=P, d_model=D_MODEL, n=N_FIXED, weight_decays=WEIGHT_DECAYS,
                  seeds=sweep.SEEDS, amp=AMP, freqs=FREQS,
                  num_epochs=NUM_EPOCHS, stop_after_grok=STOP_AFTER_GROK,
                  grok_acc=GROK_ACC),
        per_run=recs,
        by_weight_decay={str(k[0]): v for k, v in agg.items()}))

    print(f"\n=== Exp 08 (p={P}, n={N_FIXED}; grokking vs weight decay, mean±std) ===")
    print("     wd | grok_epoch (n=#grokked) | test_acc           | abl ΔCE")
    for (wd,), a in sorted(agg.items()):
        print(f"  {wd:>5g} | {sweep.fmt_stat(a['grok_epoch']):>23} | "
              f"{sweep.fmt_stat(a['final_test_acc'], 3):>18} | "
              f"{sweep.fmt_stat(a['ablation_delta'], 3, plus=True)}")
    print("\n✅ exp08 done")
