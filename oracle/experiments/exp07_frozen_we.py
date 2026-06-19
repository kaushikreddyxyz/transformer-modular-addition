# %% [markdown]
# # Exp 07 — Frozen token embedding: can injected freqs carry grokking alone?
# Same p=113 setup as exp01, but `W_E` (the token embedding) is held FIXED at its
# random init for the whole run. A trainable W_E is how the baseline transformer
# normally builds its own Fourier embedding of the inputs; freezing it removes
# that route, so any generalization must ride on the *injected* oracle frequencies
# in the residual stream. We sweep n ∈ {0, 3, 6, 8} injected pairs over 4 seeds
# (n=0 = frozen W_E with no oracle, a floor control that should never grok):
# does more injected structure compensate for an embedding that can't adapt, and
# how does the completeness threshold compare to exp01's trainable-W_E curve?
#
# Early stopping: training halts STOP_AFTER_GROK epochs after the first epoch with
# test_acc >= GROK_ACC (the grokked model is checkpointed at the stop epoch), with
# a 25k-epoch hard cap for runs that never grok. Everything else (frac_train=0.3,
# weight_decay=1.0, d_model=128) matches the exp01 regime for a clean comparison.

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

EXP = "exp07"
P = 113                       # d_vocab = 114 (set by sweep.make_config)
D_MODEL = 128                 # → d_mlp = 512, d_head = 32 (project default)
AMP = 1.0
N_LIST = [0, 3, 6, 8]         # injected pairs; 0 = frozen-W_E + no-oracle floor
                              # control (a fixed random embedding with no injected
                              # structure — expected never to grok)
FREEZE = ["embed.W_E"]        # hold the token embedding at random init
NUM_EPOCHS = 25_000           # hard cap for runs that never grok
STOP_AFTER_GROK = 1_000       # early-stop budget past the grok epoch
GROK_ACC = 0.99               # project grok threshold (test acc)
# Dense early schedule (oracle-assisted grokking is fast) covering the 25k cap;
# harness also drops a checkpoint at the dynamic early-stop epoch.
CKPT_EPOCHS = (10, 100, 500, 1000, 2500, 5000, 7500,
               10_000, 15_000, 20_000, 25_000)


def get_runs():
    runs = []
    for n in N_LIST:
        freqs = sweep.pick_freqs(n, p=P)
        oracle = (dict(kind="fourier", freqs=freqs, amp=AMP) if n
                  else dict(kind="none"))   # n=0: frozen W_E, no oracle
        for s in sweep.SEEDS:
            runs.append(sweep.spec(
                exp=EXP, label=f"n{n}_s{s}", seed=s, oracle=oracle,
                p=P, d_model=D_MODEL, num_epochs=NUM_EPOCHS,
                ckpt_epochs=CKPT_EPOCHS,
                freeze=FREEZE, stop_after_grok=STOP_AFTER_GROK, grok_acc=GROK_ACC,
                axes=dict(n=n, seed=s, amp=AMP, freqs=freqs, p=P,
                          d_model=D_MODEL, frozen_we=True)))
    return runs


# %% run (sequential; use `python -m modular_addition.oracle.runner` to parallelize)
if __name__ == "__main__" or "ipykernel" in sys.modules:
    results = sweep.run_all(get_runs())

    # %% summary — aggregate across seeds, report per n
    recs = [sweep.final_record(r) for r in results]
    agg = sweep.mean_std(
        recs, keys=["grok_epoch", "final_test_acc", "ablation_delta",
                    "we_power_injected", "n_key_freqs", "injected_in_key"],
        group_keys=["ax_n"])
    sweep.write_summary(EXP, dict(
        grid=dict(p=P, d_model=D_MODEL, n_list=N_LIST, seeds=sweep.SEEDS,
                  amp=AMP, frozen_we=True, freeze=FREEZE,
                  num_epochs=NUM_EPOCHS, stop_after_grok=STOP_AFTER_GROK,
                  grok_acc=GROK_ACC),
        per_run=recs,
        by_n={str(k[0]): v for k, v in agg.items()}))

    print(f"\n=== Exp 07 (p={P}, FROZEN W_E; uptake vs n, mean±std) ===")
    print("   n | grok_epoch (n=#grokked) | test_acc           | abl ΔCE          | inj∈key")
    for (n,), a in sorted(agg.items()):
        print(f"  {n:>2} | {sweep.fmt_stat(a['grok_epoch']):>23} | "
              f"{sweep.fmt_stat(a['final_test_acc'], 3):>18} | "
              f"{sweep.fmt_stat(a['ablation_delta'], 3, plus=True):>16} | "
              f"{sweep.fmt_stat(a['injected_in_key'], 1)}")
    print("\n✅ exp07 done")
