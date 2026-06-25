# %% [markdown]
# # Exp 08 figures — weight decay vs oracle-assisted grokking
# Trainable W_E, n=5 injected Fourier pairs held fixed, amp=1.0, p=113, 4 seeds.
# Weight decay swept across the LOW regime {1e-4 .. 1e-2}, all far below the
# wd=1.0 default that normally *drives* grokking. Question: does a strong
# injected oracle keep generalization (and embedding uptake) intact when the
# weight-decay pressure is weak, and where is the wd floor below which even the
# oracle can't get the model to grok within 25k epochs?
#
#   headline:  we_spectrum_by_wd.png   final W_E amplitude spectrum per wd (the ask)
#   readout:   wl_spectrum_by_wd.png   final W_L amplitude spectrum per wd (mirror)
#   uptake:    uptake_vs_wd.png         final W_E power frac on injected freqs
#   ablation:  dependence_vs_wd.png     final acc oracle ON vs OFF (live dep.)
#   the floor: grok_dynamics_by_wd.png  test_acc vs epoch, all seeds, per wd
#              grok_summary_vs_wd.png    grok rate + median grok epoch vs wd
#
# Sweep axis is weight_decay (n fixed), so this is a bespoke plotter on
# plot_common, like exp04/exp05 — NOT the n-keyed plot_suite.build().

# %% imports + data
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc  # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[1] / "results" / "run_20260616_110846")
EXP = "exp08"
FIG = RES / "figures" / EXP
FIG.mkdir(parents=True, exist_ok=True)

pc.set_style()

runs = pc.load_exp(RES, EXP)
WDS = sorted(pc.axis_values(runs, "weight_decay"))     # 1e-4 .. 1e-2
SEEDS = pc.axis_values(runs, "seed")
COL = pc.color_map(WDS, baseline=None)                 # color by wd (no baseline)

# injected freqs + Fourier-bin count are fixed across the sweep (n=5)
INJ = (pc.final_snap(runs[0]).get("injected_freqs")
       or runs[0]["_axes"].get("freqs", []))
L = len(pc.final_snap(runs[0])["we_freq_power_full"])  # p//2 bins
P = 2 * L + 1
N_FIXED = runs[0]["_axes"].get("n", len(INJ))
CENSOR = max(r["num_epochs"] for r in runs)

n_total = len(runs)
n_grok = sum(pc.groked(r) for r in runs)
print(f"exp08: {n_total} runs ({len(WDS)} wd x {len(SEEDS)} seeds), "
      f"{n_grok} grokked, {n_total - n_grok} never grokked; p={P}, n={N_FIXED}, "
      f"injected={INJ}")

WD_NOTE = (f"p={P}, n={N_FIXED} injected pairs (fixed), amp=1.0, frac_train=0.3, "
           f"trainable W_E, {CENSOR // 1000}k-epoch cap; wd=1.0 is the usual "
           "grokking default — every wd here is far below it.")


def wd_tag(sel):
    """Per-panel grok-rate suffix, '' when every seed groks."""
    gk = sum(pc.groked(r) for r in sel)
    return "" if gk == len(sel) else f"  ({gk}/{len(sel)} grok)"


def _amp_spectrum(power_full, p):
    """Per-frequency amplitude = sqrt(power / p), in W_E's native residual units
    — directly comparable to the oracle amp (matches exp02_2's convention)."""
    return pc.amp_spectrum(power_full, p)          # shared convention helper


# %% ----------------------------------------------------------------- #
# FIG 1 (headline) — final W_E amplitude spectrum, one panel per weight decay
# --------------------------------------------------------------------- #
ncol = min(len(WDS), 3)
nrow = int(np.ceil(len(WDS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 2.9 * nrow),
                         sharey=True, squeeze=False)
freqs = np.arange(1, L + 1)
for ax, wd in zip(axes.flat, WDS):
    sel = pc.select(runs, weight_decay=wd)
    specs = [_amp_spectrum(pc.final_snap(r)["we_freq_power_full"], P)
             for r in sel if pc.final_snap(r).get("we_freq_power_full")]
    if not specs:
        continue
    for j, f in enumerate(INJ):
        ax.axvline(f, color="green", ls="--", lw=1.0, alpha=0.6,
                   label="injected" if j == 0 else None)
    for sp in specs:
        ax.plot(freqs, sp, color=COL[wd], alpha=0.3, lw=0.9)
    ax.plot(freqs, np.mean(specs, 0), color=COL[wd], lw=1.8)
    ax.set_ylim(bottom=0)                       # amplitude is non-negative
    ax.set_title(f"wd = {wd:g}{wd_tag(sel)}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True)
for ax in axes.flat[len(WDS):]:
    ax.set_visible(False)
fig.supxlabel(f"Fourier frequency index (1..{L})")
fig.supylabel("final W_E amplitude  √(power/p)  (residual units)")
fig.suptitle(f"Exp08 — final W_E amplitude spectrum vs weight decay (p={P}, "
             f"n={N_FIXED} fixed injected pairs)", fontsize=12.5)
pc.save(fig, FIG / "we_spectrum_by_wd.png", cap=(
    f"Final W_E per-frequency amplitude √(power/p) ({L} freqs, residual units — "
    "directly comparable to the oracle amp=1.0) per weight decay; 4 seeds faint + "
    "mean bold, shared y-axis, green dashed = the (fixed) injected frequencies. "
    "With a trainable embedding the amplitude concentrates on the injected sites; "
    "this panel shows whether weakening weight decay flattens that concentration "
    f"or raises off-target leakage. {WD_NOTE}"))


# %% ----------------------------------------------------------------- #
# FIG 1b (readout mirror) — final W_L = W_out^T W_U neuron-logit-map amplitude
# spectrum, one panel per weight decay. The output-side mirror of FIG 1. Needs
# the wl_* snapshot fields; for runs trained before analysis.py recorded them,
# run  backfill_wl.py RESULTS_DIR exp08  to recompute W_L from the final
# checkpoints (weight-only, no retraining) before plotting.
# --------------------------------------------------------------------- #
if any(pc.final_snap(r).get("wl_freq_power_full") for r in runs):
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 2.9 * nrow),
                             sharey=True, squeeze=False)
    for ax, wd in zip(axes.flat, WDS):
        sel = pc.select(runs, weight_decay=wd)
        specs = [_amp_spectrum(pc.final_snap(r)["wl_freq_power_full"], P)
                 for r in sel if pc.final_snap(r).get("wl_freq_power_full")]
        if not specs:
            continue
        for j, f in enumerate(INJ):
            ax.axvline(f, color="green", ls="--", lw=1.0, alpha=0.6,
                       label="injected" if j == 0 else None)
        for sp in specs:
            ax.plot(freqs, sp, color=COL[wd], alpha=0.3, lw=0.9)
        ax.plot(freqs, np.mean(specs, 0), color=COL[wd], lw=1.8)
        ax.set_ylim(bottom=0)                       # amplitude is non-negative
        ax.set_title(f"wd = {wd:g}{wd_tag(sel)}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7.5, frameon=True)
    for ax in axes.flat[len(WDS):]:
        ax.set_visible(False)
    fig.supxlabel(f"Fourier frequency index (1..{L})")
    fig.supylabel("final W_L amplitude  √(power/p)  (logit units)")
    fig.suptitle(f"Exp08 — final W_L amplitude spectrum vs weight decay (p={P}, "
                 f"n={N_FIXED} fixed injected pairs)", fontsize=12.5)
    pc.save(fig, FIG / "wl_spectrum_by_wd.png", cap=(
        f"Final W_L = W_out^T W_U neuron-logit-map per-frequency amplitude "
        f"√(power/p) ({L} freqs, logit units) per weight decay; the readout-side "
        "mirror of we_spectrum_by_wd.png. 4 seeds faint + mean bold, shared "
        "y-axis, green dashed = the (fixed) injected frequencies. Shows whether "
        "the output side keeps concentrating on the oracle's frequencies as the "
        "weight-decay pressure weakens. W_L is weight-only, so this is recomputed "
        f"post-hoc from each cell's final checkpoint (backfill_wl.py). {WD_NOTE}"))
else:
    print("  [wl_spectrum] no wl_freq_power_full in snapshots -- run "
          "backfill_wl.py RESULTS_DIR exp08 to populate it; skipping.")


# %% ----------------------------------------------------------------- #
# shared: a final-snapshot scalar vs wd (log-x), grokked-only,
# jittered seed dots + connected mean line. `series` = [(fn,color,label),...]
# --------------------------------------------------------------------- #
def final_vs_wd(series, ylabel, title, fname, cap, ylim=None, only_groked=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for fn, color, lab in series:
        mwd, means = [], []
        for wd in WDS:
            sub = [r for r in pc.select(runs, weight_decay=wd)
                   if (pc.groked(r) or not only_groked)]
            vals = [fn(pc.final_snap(r)) for r in sub]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            xj = wd * np.exp(rng.uniform(-0.06, 0.06, len(vals)))   # log-jitter
            ax.scatter(xj, vals, color=color, alpha=0.45, s=30, zorder=2)
            means.append(np.mean(vals))
            mwd.append(wd)
        if means:
            ax.plot(mwd, means, color=color, lw=1.9, marker="o", ms=5,
                    zorder=3, label=lab)
    ax.set_xscale("log")
    ax.set(xlabel="weight decay (log)", ylabel=ylabel, title=title)
    ax.set_xticks(WDS)
    ax.set_xticklabels([f"{w:g}" for w in WDS])
    if ylim:
        ax.set_ylim(*ylim)
    if any(lab for _, _, lab in series):
        ax.legend(fontsize=9, framealpha=0.9)
    pc.save(fig, FIG / fname, cap=cap)


# %% FIG 2 — embedding uptake of injected freqs vs wd
final_vs_wd(
    [(pc.frac_we_power_injected, COL[WDS[-1]], None)],
    "final W_E power fraction on injected freqs",
    "Exp08 — embedding uptake of injected freqs vs weight decay (grokked only)",
    "uptake_vs_wd.png",
    cap=("Fraction of final W_E Fourier power sitting on the injected freqs vs "
         "weight decay (grokked seeds; dots = seeds, line = mean). Tests whether "
         "weak weight decay still lets the trainable embedding concentrate power "
         f"on the oracle's frequencies. {WD_NOTE}"),
    ylim=(-0.03, 1.03))


# %% FIG 3 — dependence on the live oracle (acc ON vs OFF) vs wd  [ablation]
final_vs_wd(
    [(lambda s: pc.ablation(s, "acc_on"), "#1f77b4", "oracle ON"),
     (lambda s: pc.ablation(s, "acc_off"), "#d62728", "oracle OFF")],
    "final test accuracy",
    "Exp08 — dependence on the live oracle vs weight decay (grokked only)",
    "dependence_vs_wd.png",
    cap=("Final test accuracy with the live oracle ON (blue) vs switched OFF at "
         "inference (red) vs weight decay (grokked seeds; dots = seeds, line = "
         "mean). The ON-OFF gap is how much the model leans on the live signal "
         "rather than having internalised it into weights — the ablation effect "
         f"as wd varies. {WD_NOTE}"),
    ylim=(-0.03, 1.03))


# %% ----------------------------------------------------------------- #
# FIG 4 — test_acc vs epoch, small-multiples by wd, ALL seeds (incl fails)
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, len(WDS), figsize=(3.3 * len(WDS), 5.0),
                         sharex=True, sharey=True, squeeze=False)
for ax, wd in zip(axes.flat, WDS):
    sel = pc.select(runs, weight_decay=wd)
    for r in sel:
        ep, acc = pc.hist_series(r, "test_acc")
        m = ep > 0                                  # drop epoch 0 for log-x
        ax.plot(ep[m], acc[m], color=COL[wd],
                alpha=0.85 if pc.groked(r) else 0.5,
                lw=1.4 if pc.groked(r) else 0.9,
                ls="-" if pc.groked(r) else (0, (2, 1.5)))
    gk = sum(pc.groked(r) for r in sel)
    ax.axhline(1.0 / P, color="0.6", ls=":", lw=0.8)
    ax.set_title(f"wd = {wd:g}   ({gk}/{len(sel)} grok)", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(left=8)
    ax.set_ylim(-0.02, 1.02)
axes.flat[-1].legend(handles=[
    Line2D([], [], color="0.4", lw=1.6, ls="-", label="grokked"),
    Line2D([], [], color="0.4", lw=1.0, ls=(0, (2, 1.5)),
           label="never grokked")], fontsize=8, loc="lower right",
    framealpha=0.9)
fig.supxlabel("epoch (log)")
fig.supylabel("test accuracy")
fig.suptitle("Exp08 — grokking dynamics by weight decay (all 4 seeds; "
             "dashed = never grokked, dotted = chance)")
pc.save(fig, FIG / "grok_dynamics_by_wd.png", cap=(
    f"Test accuracy vs epoch, one panel per weight decay, all {len(SEEDS)} "
    "seeds (dashed = never grokked within the cap, dotted = chance 1/"
    f"{P}). Reveals the wd floor: the smallest weight decay at which the oracle "
    f"can still drive grokking in the budget. {WD_NOTE}"))


# %% ----------------------------------------------------------------- #
# FIG 5 — grok rate + median grok epoch vs wd (the floor, summarised)
# --------------------------------------------------------------------- #
rate = [pc.grok_rate(pc.select(runs, weight_decay=wd)) for wd in WDS]
med = [(np.median([pc.grok_epoch(r) for r in pc.select(runs, weight_decay=wd)
                   if pc.groked(r)])
        if any(pc.groked(r) for r in pc.select(runs, weight_decay=wd))
        else np.nan) for wd in WDS]

fig, (axr, axe) = plt.subplots(1, 2, figsize=(12, 4.6))
axr.plot(WDS, rate, color="#3b528b", lw=2, marker="o", ms=7, zorder=3)
axr.set_xscale("log")
axr.set(xlabel="weight decay (log)", ylabel="grok rate (fraction of 4 seeds)",
        title="(a) grok success vs weight decay", ylim=(-0.05, 1.08))
axr.set_xticks(WDS)
axr.set_xticklabels([f"{w:g}" for w in WDS])
for wd, rt in zip(WDS, rate):
    axr.annotate(f"{int(round(rt * len(SEEDS)))}/{len(SEEDS)}", (wd, rt),
                 textcoords="offset points", xytext=(0, 8), ha="center",
                 fontsize=8.5, color="0.25")

axe.plot(WDS, med, color="#b5367a", lw=2, marker="s", ms=7, zorder=3)
axe.set_xscale("log")
axe.set(xlabel="weight decay (log)", ylabel="median grok epoch (grokked seeds)",
        title="(b) time-to-grok vs weight decay")
axe.set_xticks(WDS)
axe.set_xticklabels([f"{w:g}" for w in WDS])
if np.isfinite(med).any() and np.nanmax(med) / max(np.nanmin(med), 1) > 20:
    axe.set_yscale("log")
fig.suptitle("Exp08 — where is the weight-decay floor? (n=5 oracle fixed)",
             fontsize=12.5)
pc.save(fig, FIG / "grok_summary_vs_wd.png", cap=(
    "(a) Fraction of 4 seeds reaching grok and (b) median epochs-to-grok "
    "(grokked seeds only) vs weight decay. Together they locate the wd floor "
    "below which the fixed n=5 oracle can no longer produce grokking within the "
    f"budget. {WD_NOTE}"))

print(f"wrote {len(list(FIG.glob('*.png')))} figures -> {FIG}")
