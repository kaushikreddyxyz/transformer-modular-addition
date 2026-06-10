# %% [markdown]
# # Exp 01 — Injection uptake across n injected frequency pairs
# Does the model *use* injected frequencies, and how does uptake scale with how
# many pairs we inject? Grid: n ∈ sweep.N_LIST (0 = baseline) × 4 seeds, frozen
# Fourier oracle at amp 1.0, 30k epochs, uptake snapshots every 2k epochs.
# This grid is also the T=0 / amp=1.0 reference for exp02_1/exp02_2 and the
# data source for exp03 (grok speed vs n).

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

EXP = "exp01"
AMP = 1.0


def get_runs():
    runs = []
    for n in sweep.N_LIST:
        freqs = sweep.pick_freqs(n)
        oracle = (dict(kind="fourier", freqs=freqs, amp=AMP) if n
                  else dict(kind="none"))
        for s in sweep.SEEDS:
            runs.append(sweep.spec(
                exp=EXP, label=f"n{n}_s{s}", seed=s, oracle=oracle,
                axes=dict(n=n, seed=s, amp=AMP, freqs=freqs)))
    return runs


# %% run (sequential; use experiments/runner.py to parallelize)
if __name__ == "__main__" or "ipykernel" in sys.modules:
    results = sweep.run_all(get_runs())

    # %% summary — aggregate across seeds, report per n
    recs = [sweep.final_record(r) for r in results]
    agg = sweep.mean_std(
        recs, keys=["grok_epoch", "final_test_acc", "ablation_delta",
                    "we_power_injected", "we_total_norm", "n_key_freqs",
                    "injected_in_key"],
        group_keys=["ax_n"])
    sweep.write_summary(EXP, dict(
        grid=dict(n_list=sweep.N_LIST, seeds=sweep.SEEDS, amp=AMP),
        per_run=recs,
        by_n={str(k[0]): v for k, v in agg.items()}))

    fmt = sweep.fmt_stat
    print("\n=== Exp 01 (uptake vs n, mean±std over seeds) ===")
    print("   n | grok_epoch          | test_acc           | abl ΔCE          | inj∈key")
    for (n,), a in sorted(agg.items()):
        print(f"  {n:>2} | {fmt(a['grok_epoch']):>19} | "
              f"{fmt(a['final_test_acc'], 3):>18} | "
              f"{fmt(a['ablation_delta'], 3, plus=True):>16} | "
              f"{fmt(a['injected_in_key'], 1)}")
    print("\n✅ exp01 done")
