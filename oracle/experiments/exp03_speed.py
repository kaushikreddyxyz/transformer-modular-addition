# %% [markdown]
# # Exp 03 — Grokking speed vs #injected frequency pairs (derived from Exp 01)
# Hypothesis: injecting the right Fourier features lets the model skip building
# its own embedding circuit, so it groks earlier. Exp 01's grid
# (n ∈ N_LIST × 4 seeds, 30k epochs, no early stop) already contains every run
# this question needs, so exp03 TRAINS NOTHING: it reads exp01's result files
# and reports grok-speed statistics per n. Run exp01 (or the runner) first.

# %% imports + path bootstrap
import sys
from pathlib import Path
try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

import json

from modular_addition.oracle import sweep

EXP = "exp03"
SOURCE = "exp01"


def get_runs():
    """exp03 is analysis-only; it contributes no training runs."""
    return []


def load_source_records():
    recs = []
    for p in sorted((sweep.RESULTS_DIR / SOURCE).glob("*.result.json")):
        recs.append(sweep.final_record(json.load(open(p))))
    return recs


# %% analyze
if __name__ == "__main__" or "ipykernel" in sys.modules:
    recs = load_source_records()
    if not recs:
        sys.exit(f"no {SOURCE} results found — run exp01 first")

    agg = sweep.mean_std(recs, keys=["grok_epoch", "final_test_acc"],
                         group_keys=["ax_n"])
    base = agg.get((0,), {}).get("grok_epoch")
    by_n = {}
    for (n,), a in sorted(agg.items()):
        ge = a["grok_epoch"]
        n_total = a["_n_runs"]
        row = dict(n=n, grok_epoch=ge, final_test_acc=a["final_test_acc"],
                   frac_grokked=(ge["n"] / n_total) if ge else 0.0,
                   speedup_vs_baseline=(base["mean"] / ge["mean"]
                                        if base and ge else None))
        by_n[str(n)] = row
    sweep.write_summary(EXP, dict(source=SOURCE, by_n=by_n))

    print("\n=== Exp 03 (grok speed vs n, derived from exp01) ===")
    print("   n | grok_epoch          | grokked | speedup vs n=0")
    for k, r in by_n.items():
        sp = f"x{r['speedup_vs_baseline']:.2f}" if r["speedup_vs_baseline"] else "—"
        print(f"  {k:>2} | {sweep.fmt_stat(r['grok_epoch']):>19} | "
              f"{r['frac_grokked']:.0%}    | {sp}")
    print("\n✅ exp03 done")
