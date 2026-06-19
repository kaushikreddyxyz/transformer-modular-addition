# %% [markdown]
# # Exp 09 — Organic vs synthetic oracle
# exp01 injects a CLEAN *synthetic* oracle: cos/sin Fourier features at hand-picked
# frequencies, unit amplitude. Here the oracle is *organic* — the final embedding
# `W_E` of a NO-ORACLE baseline that already grokked the task (exp01's n0 cells),
# injected verbatim as the frozen additive per-token feature. Same OracleTransformer
# mechanism, same 30k-epoch / trainable-W_E / wd=1.0 regime as exp01: the ONLY
# changed variable is the oracle's *content* — idealized Fourier pairs vs the messy
# thing a model actually learned (its own ~5 key freqs at amp≈0.65, their relative
# weights, the non-Fourier residue, and the learned "=" embedding).
#
# Scale: RAW — `W_E` is injected exactly as learned (per-token L2 ≈ 1.5, per-freq
# amplitude ≈ 0.65, i.e. ~0.65× the synthetic n6 reference at amp=1.0). That natural
# strength gap is recorded (injected_freqs + we_freq_power), not normalized away.
#
# Grid: 4 donor baselines (n0_s0..n0_s3) × 3 fresh training seeds = 12 models.
# Compare against exp01's n6 cells — the matched-frequency-count synthetic reference
# (a grokked p=113 model uses ~5 key freqs, so n6 is the closest synthetic analogue).

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

EXP = "exp09"
SOURCE_RUN = "run_20260612_200000"     # the exp01 sweep whose baselines we mine
SOURCE_EXP = "exp01"
SOURCE_EPOCH = 30_000                  # final (fully grokked) baseline checkpoint
DONORS = ["n0_s0", "n0_s1", "n0_s2", "n0_s3"]   # the 4 no-oracle baselines
TRAIN_SEEDS = [0, 1, 2]                # 3 fresh seeds per donor -> 12 models
SCALE = 1.0                            # raw: inject W_E exactly as learned
KEY_AMP_FRAC = 0.5                     # donor "key freqs" = amp >= 0.5 * max amp


def get_runs():
    runs = []
    for donor in DONORS:
        oracle = dict(kind="organic_we", source_run=SOURCE_RUN,
                      source_exp=SOURCE_EXP, source_label=donor,
                      source_epoch=SOURCE_EPOCH, scale=SCALE,
                      include_eq=True, key_amp_frac=KEY_AMP_FRAC)
        for s in TRAIN_SEEDS:
            runs.append(sweep.spec(
                exp=EXP, label=f"{donor}_t{s}", seed=s, oracle=oracle,
                axes=dict(donor=donor, src_seed=int(donor.split("_s")[1]),
                          seed=s, kind="organic_we", scale=SCALE)))
    return runs


# %% run (sequential; use oracle/runner.py to parallelize / stack)
if __name__ == "__main__" or "ipykernel" in sys.modules:
    results = sweep.run_all(get_runs())

    # %% summary — aggregate across the 3 training seeds, report per donor
    recs = [sweep.final_record(r) for r in results]
    agg = sweep.mean_std(
        recs, keys=["grok_epoch", "final_test_acc", "ablation_delta",
                    "we_power_injected", "we_total_norm", "n_key_freqs",
                    "injected_in_key"],
        group_keys=["ax_donor"])
    sweep.write_summary(EXP, dict(
        grid=dict(source_run=SOURCE_RUN, source_exp=SOURCE_EXP,
                  source_epoch=SOURCE_EPOCH, donors=DONORS,
                  train_seeds=TRAIN_SEEDS, scale=SCALE,
                  key_amp_frac=KEY_AMP_FRAC),
        per_run=recs,
        by_donor={str(k[0]): v for k, v in agg.items()}))

    fmt = sweep.fmt_stat
    print("\n=== Exp 09 (organic oracle, mean±std over 3 train seeds) ===")
    print(" donor | grok_epoch          | test_acc           | abl ΔCE          | inj∈key")
    for (donor,), a in sorted(agg.items()):
        print(f" {donor:>5} | {fmt(a['grok_epoch']):>19} | "
              f"{fmt(a['final_test_acc'], 3):>18} | "
              f"{fmt(a['ablation_delta'], 3, plus=True):>16} | "
              f"{fmt(a['injected_in_key'], 1)}")
    print("\n✅ exp09 done")
