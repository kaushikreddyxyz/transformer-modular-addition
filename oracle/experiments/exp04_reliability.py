# %% [markdown]
# # Exp 04 — Unreliable / variable-frequency oracle
# The oracle's frequency varies per example: with prob `reliability` it is the
# true base frequency, else a random frequency (independently per injected
# pair). Grid: reliability × n pairs × 4 seeds. Where is the threshold past
# which the model disregards the oracle — and does a more complete (higher-n)
# basis buy robustness to corruption?

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

EXP = "exp04"
AMP = 1.0
RELIABILITIES = [1.0, 0.75, 0.5, 0.25, 0.0]
N_LIST = [n for n in sweep.N_LIST if n > 0]   # corruption needs an oracle


def get_runs():
    runs = []
    for rel in RELIABILITIES:
        for n in N_LIST:
            freqs = sweep.pick_freqs(n)
            for s in sweep.SEEDS:
                # map_seed = model seed → corruption draw varies across seeds,
                # so the error bars include corruption randomness, not just
                # init/split randomness.
                oracle = dict(kind="perexample_corrupt", freqs=freqs,
                              reliability=rel, amp=AMP, map_seed=s)
                runs.append(sweep.spec(
                    exp=EXP, label=f"rel{rel:g}_n{n}_s{s}", seed=s,
                    oracle=oracle,
                    axes=dict(rel=rel, n=n, seed=s, amp=AMP, freqs=freqs)))
    return runs


# %% run (sequential; use experiments/runner.py to parallelize)
if __name__ == "__main__" or "ipykernel" in sys.modules:
    results = sweep.run_all(get_runs())

    # %% summary — aggregate across seeds, report per (reliability, n)
    recs = [sweep.final_record(r) for r in results]
    agg = sweep.mean_std(
        recs, keys=["grok_epoch", "final_test_acc", "ablation_delta",
                    "we_power_injected", "n_key_freqs"],
        group_keys=["ax_rel", "ax_n"])
    sweep.write_summary(EXP, dict(
        grid=dict(reliabilities=RELIABILITIES, n_list=N_LIST,
                  seeds=sweep.SEEDS, amp=AMP),
        per_run=recs,
        by_rel_n={f"rel{k[0]:g}_n{k[1]}": v for k, v in agg.items()}))

    print("\n=== Exp 04 (reliability × n, mean±std over seeds) ===")
    print("  rel  |  n | test_acc           | abl ΔCE          | grok_epoch")
    for (rel, n), a in sorted(agg.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        print(f"  {rel:<4g} | {n:>2} | {sweep.fmt_stat(a['final_test_acc'], 3):>18} | "
              f"{sweep.fmt_stat(a['ablation_delta'], 3, plus=True):>16} | "
              f"{sweep.fmt_stat(a['grok_epoch'])}")
    print("\n✅ exp04 done")
