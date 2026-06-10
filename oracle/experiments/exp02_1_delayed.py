# %% [markdown]
# # Exp 02.1 — Delayed injection
# Turn the oracle on only at epoch T. Does the model abandon the embedding
# structure it already built and adopt the oracle — or ignore it? Grid:
# T ∈ {4000, 8000} × n ∈ N_LIST>0 × 4 seeds at amp 1.0. The T=0 reference is
# exp01's grid (same labels n{n}_s{s}, same freqs/amp) — run exp01 first.

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

EXP = "exp02_1"
AMP = 1.0
DELAYS = [4000, 8000]
N_LIST = [n for n in sweep.N_LIST if n > 0]   # delay needs an oracle


def get_runs():
    runs = []
    for T in DELAYS:
        for n in N_LIST:
            freqs = sweep.pick_freqs(n)
            oracle = dict(kind="fourier", freqs=freqs, amp=AMP)
            for s in sweep.SEEDS:
                runs.append(sweep.spec(
                    exp=EXP, label=f"delay{T}_n{n}_s{s}", seed=s,
                    oracle=oracle, inject_from_epoch=T,
                    snapshot_every=1000,   # finer cadence around the T switch
                    axes=dict(delay=T, n=n, seed=s, amp=AMP, freqs=freqs)))
    return runs


def adoption_around_T(res):
    """W_E power @ injected just before vs just after injection turns on."""
    T = res["spec"]["inject_from_epoch"]
    snaps = res.get("snapshots") or []
    before = next((s for s in reversed(snaps) if s["epoch"] < T), None)
    after = next((s for s in snaps if s["epoch"] >= T), None)
    pick = lambda s: (float(sum(s["we_freq_power_injected"]))
                      if s and s.get("we_freq_power_injected") else None)
    return pick(before), pick(after)


# %% run (sequential; use experiments/runner.py to parallelize)
if __name__ == "__main__" or "ipykernel" in sys.modules:
    results = sweep.run_all(get_runs())

    # %% summary — aggregate across seeds per (T, n)
    recs = []
    for r in results:
        rec = sweep.final_record(r)
        rec["we_power_injected_before_T"], rec["we_power_injected_after_T"] = \
            adoption_around_T(r)
        recs.append(rec)
    agg = sweep.mean_std(
        recs, keys=["grok_epoch", "final_test_acc", "ablation_delta",
                    "we_power_injected", "injected_in_key",
                    "we_power_injected_before_T", "we_power_injected_after_T"],
        group_keys=["ax_delay", "ax_n"])
    sweep.write_summary(EXP, dict(
        grid=dict(delays=DELAYS, n_list=N_LIST, seeds=sweep.SEEDS, amp=AMP,
                  t0_reference="exp01"),
        per_run=recs,
        by_delay_n={f"delay{k[0]}_n{k[1]}": v for k, v in agg.items()}))

    print("\n=== Exp 02.1 (delayed injection, mean±std over seeds) ===")
    print("  T     |  n | test_acc           | abl ΔCE          | W_E@inj (final)")
    for (T, n), a in sorted(agg.items()):
        print(f"  {T:<5} | {n:>2} | {sweep.fmt_stat(a['final_test_acc'], 3):>18} | "
              f"{sweep.fmt_stat(a['ablation_delta'], 3, plus=True):>16} | "
              f"{sweep.fmt_stat(a['we_power_injected'], 1)}")
    print("(compare against exp01 summary = T=0 reference)")
    print("\n✅ exp02_1 done")
