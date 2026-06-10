# %% [markdown]
# # Exp 05 — Weakly-informative answer hint
# Inject a weak feature about the answer c=(i+j) mod p at the "=" position
# (c % 10 or c // 10). Does the model then solve the task with fewer
# frequencies? Hints aren't frequency pairs, so the project-wide n-sweep does
# not apply here; the grid is hint-config × 4 seeds against a no-hint baseline.

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

EXP = "exp05"
AMP = 1.0
CONFIGS = [
    ("baseline", dict(kind="none")),
    ("hint_mod10_onehot", dict(kind="answer_hint", hint="mod", modulus=10,
                               code="onehot", amp=AMP)),
    ("hint_div10_onehot", dict(kind="answer_hint", hint="div", modulus=10,
                               code="onehot", amp=AMP)),
    ("hint_mod10_fourier", dict(kind="answer_hint", hint="mod", modulus=10,
                                code="fourier", amp=AMP)),
]


def get_runs():
    runs = []
    for name, oracle in CONFIGS:
        for s in sweep.SEEDS:
            runs.append(sweep.spec(
                exp=EXP, label=f"{name}_s{s}", seed=s, oracle=oracle,
                axes=dict(hint=name, seed=s, amp=AMP)))
    return runs


# %% run (sequential; use experiments/runner.py to parallelize)
if __name__ == "__main__" or "ipykernel" in sys.modules:
    results = sweep.run_all(get_runs())

    # %% summary — aggregate across seeds, report per hint config
    recs = [sweep.final_record(r) for r in results]
    agg = sweep.mean_std(
        recs, keys=["grok_epoch", "final_test_acc", "n_key_freqs",
                    "ablation_delta", "we_total_norm"],
        group_keys=["ax_hint"])
    sweep.write_summary(EXP, dict(
        grid=dict(configs=[c for c, _ in CONFIGS], seeds=sweep.SEEDS, amp=AMP),
        per_run=recs,
        by_hint={k[0]: v for k, v in agg.items()}))

    print("\n=== Exp 05 (answer hints, mean±std over seeds) ===")
    print("  config              | grok_epoch          | test_acc           | #key_freqs   | abl ΔCE")
    for (name,), a in sorted(agg.items()):
        print(f"  {name:<19} | {sweep.fmt_stat(a['grok_epoch']):>19} | "
              f"{sweep.fmt_stat(a['final_test_acc'], 3):>18} | "
              f"{sweep.fmt_stat(a['n_key_freqs'], 1):>12} | "
              f"{sweep.fmt_stat(a['ablation_delta'], 3, plus=True)}")
    print("\n✅ exp05 done")
