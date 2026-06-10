# %% [markdown]
# # Exp 02.2 — Amplitude sweep (W_E laziness)
# As the oracle gets louder, does the trainable W_E offload work onto it
# (lower norm / lower own-power at the injected freqs), and does performance
# survive? Grid: amp ∈ {0.5, 1, 2, 4} × n ∈ N_LIST>0 × 4 seeds. The amp=1.0
# row duplicates exp01's grid deliberately so this experiment is self-contained.

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

EXP = "exp02_2"
AMPS = [0.5, 1.0, 2.0, 4.0]
N_LIST = [n for n in sweep.N_LIST if n > 0]   # amplitude needs an oracle


def get_runs():
    runs = []
    for amp in AMPS:
        for n in N_LIST:
            freqs = sweep.pick_freqs(n)
            oracle = dict(kind="fourier", freqs=freqs, amp=amp)
            for s in sweep.SEEDS:
                runs.append(sweep.spec(
                    exp=EXP, label=f"amp{amp:g}_n{n}_s{s}", seed=s,
                    oracle=oracle,
                    axes=dict(amp=amp, n=n, seed=s, freqs=freqs)))
    return runs


# %% run (sequential; use experiments/runner.py to parallelize)
if __name__ == "__main__" or "ipykernel" in sys.modules:
    results = sweep.run_all(get_runs())

    # %% summary — aggregate across seeds per (amp, n)
    recs = [sweep.final_record(r) for r in results]
    agg = sweep.mean_std(
        recs, keys=["grok_epoch", "final_test_acc", "ablation_delta",
                    "we_power_injected", "we_total_norm", "we_gini",
                    "n_key_freqs"],
        group_keys=["ax_amp", "ax_n"])
    sweep.write_summary(EXP, dict(
        grid=dict(amps=AMPS, n_list=N_LIST, seeds=sweep.SEEDS),
        per_run=recs,
        by_amp_n={f"amp{k[0]:g}_n{k[1]}": v for k, v in agg.items()}))

    print("\n=== Exp 02.2 (amplitude × n, mean±std over seeds) ===")
    print("  amp  |  n | test_acc           | |W_E|            | W_E@inj")
    for (amp, n), a in sorted(agg.items()):
        print(f"  {amp:<4g} | {n:>2} | {sweep.fmt_stat(a['final_test_acc'], 3):>18} | "
              f"{sweep.fmt_stat(a['we_total_norm'], 1):>16} | "
              f"{sweep.fmt_stat(a['we_power_injected'], 1)}")
    print("\n✅ exp02_2 done")
