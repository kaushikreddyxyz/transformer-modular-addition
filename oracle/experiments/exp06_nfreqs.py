# %% [markdown]
# # Exp 06 — Completeness at scale: n injected pairs on a larger model
# The small-p completeness story (exp01/exp03: ≥3 pairs → fast full grokking)
# rebuilt at scale: larger modulus p=211 (d_vocab=212, 44.5k examples) and
# d_model=256 (d_mlp=1024), sweeping n ∈ 0..11 injected frequency pairs over
# 4 seeds. Does the completeness threshold move with task/model size, or stay
# at "a handful of irreps is enough"?

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

EXP = "exp06"
P = 211          # prime modulus → d_vocab = 212 (set by sweep.make_config)
D_MODEL = 256    # → d_mlp = 1024, d_head = 64
AMP = 1.0
N_LIST = list(range(12))   # 0 (baseline) .. 11 injected pairs
FREQ_POOL = sweep.pick_freqs(max(N_LIST), p=P)   # nested deterministic pool


def get_runs():
    runs = []
    for n in N_LIST:
        freqs = FREQ_POOL[:n]
        oracle = (dict(kind="fourier", freqs=freqs, amp=AMP) if n
                  else dict(kind="none"))
        for s in sweep.SEEDS:
            runs.append(sweep.spec(
                exp=EXP, label=f"n{n}_s{s}", seed=s, oracle=oracle,
                p=P, d_model=D_MODEL,
                axes=dict(n=n, seed=s, amp=AMP, freqs=freqs, p=P,
                          d_model=D_MODEL)))
    return runs


# %% run (sequential; use experiments/runner.py to parallelize)
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
                  amp=AMP, freq_pool=FREQ_POOL),
        per_run=recs,
        by_n={str(k[0]): v for k, v in agg.items()}))

    print(f"\n=== Exp 06 (p={P}, d_model={D_MODEL}; uptake vs n, mean±std) ===")
    print("   n | grok_epoch          | test_acc           | abl ΔCE          | inj∈key")
    for (n,), a in sorted(agg.items()):
        print(f"  {n:>2} | {sweep.fmt_stat(a['grok_epoch']):>19} | "
              f"{sweep.fmt_stat(a['final_test_acc'], 3):>18} | "
              f"{sweep.fmt_stat(a['ablation_delta'], 3, plus=True):>16} | "
              f"{sweep.fmt_stat(a['injected_in_key'], 1)}")
    print("\n✅ exp06 done")
