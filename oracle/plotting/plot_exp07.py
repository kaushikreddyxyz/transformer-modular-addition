# %% [markdown]
# # Exp 07 figures — frozen-W_E n-sweep (does injected structure carry grokking?)
# Same p=113 regime as exp01, but W_E (the token embedding) is held FIXED at
# random init for the whole run, so the model CANNOT build its own Fourier
# embedding of the inputs. Any generalization must ride on the injected oracle
# frequencies in the residual stream. n in {0,3,6,8} injected pairs, 4 seeds
# (n=0 = frozen W_E + no oracle, a floor control expected to (mostly) fail).
#
#   headline:  grok_dynamics_by_n.png   does frozen-W_E + oracle grok, how fast?
#   the ask:   we_spectrum_by_n.png     final W_E spectrum — the FROZEN control
#              no_uptake_by_n.png        W_E power on injected freqs stays flat
#   causal:    dependence_vs_n.png       final acc oracle ON vs OFF (live dep.)
#
# This is a BESPOKE plotter (not plot_suite.build()): with a frozen embedding
# the suite's "uptake / power piles up on injected sites" framing is physically
# wrong, and its post-grok ablation x-window (>=1800 ep) clips exp07's fast
# (~400 ep) grokking entirely. We reframe uptake/spectrum as null controls and
# read dependence from the final snapshot (no fragile time axis).

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
EXP = "exp07"
FIG = RES / "figures" / EXP
FIG.mkdir(parents=True, exist_ok=True)

pc.set_style()

runs = pc.load_exp(RES, EXP)
NS = pc.axis_values(runs, "n")                         # [0,3,6,8]
INJ_NS = [n for n in NS if n]                          # oracle-injected n
SEEDS = pc.axis_values(runs, "seed")
COL = pc.color_map(NS, baseline=0)                     # n=0 gray, rest viridis
L = len(pc.final_snap(runs[0])["we_freq_power_full"])  # p//2 bins
P = 2 * L + 1
CENSOR = max(r["num_epochs"] for r in runs)

# injected freq set per n (differs across n)
INJ = {n: (pc.final_snap(pc.select(runs, n=n)[0]).get("injected_freqs")
           or pc.select(runs, n=n)[0]["_axes"].get("freqs", []))
       for n in INJ_NS}

n_grok = sum(pc.groked(r) for r in runs)
print(f"exp07: {len(runs)} runs ({len(NS)} n x {len(SEEDS)} seeds), "
      f"{n_grok} grokked; p={P}, frozen W_E.")

FZ = (f"p={P}, frac_train=0.3, wd=1.0, trainable-W_E baseline = exp01; here "
      "W_E is FROZEN at random init so the embedding cannot adapt — any "
      "grokking must ride the injected oracle in the residual stream.")


def n_tag(sel):
    gk = sum(pc.groked(r) for r in sel)
    return "" if gk == len(sel) else f"  ({gk}/{len(sel)} grok)"


# %% ----------------------------------------------------------------- #
# FIG 1 (headline) — grokking dynamics by n: all seeds, dashed = never grok
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, len(NS), figsize=(3.3 * len(NS), 5.0),
                         sharex=True, sharey=True, squeeze=False)
for ax, n in zip(axes.flat, NS):
    sel = pc.select(runs, n=n)
    for r in sel:
        ep, acc = pc.hist_series(r, "test_acc")
        m = ep > 0
        ax.plot(ep[m], acc[m], color=COL[n],
                alpha=0.85 if pc.groked(r) else 0.5,
                lw=1.4 if pc.groked(r) else 0.9,
                ls="-" if pc.groked(r) else (0, (2, 1.5)))
    gk = [pc.grok_epoch(r) for r in sel if pc.groked(r)]
    ax.axhline(1.0 / P, color="0.6", ls=":", lw=0.8)
    txt = (f"{len(gk)}/{len(sel)} grok\n~{int(np.median(gk))} ep" if gk
           else f"0/{len(sel)}\nnever groks")
    ax.text(0.95, 0.06, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.25")
    ttl = f"n = {n}" + ("  (no oracle)" if n == 0 else "")
    ax.set_title(ttl, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(1, CENSOR)
    ax.set_ylim(-0.03, 1.05)
axes.flat[-1].legend(handles=[
    Line2D([], [], color="0.4", lw=1.6, ls="-", label="grokked"),
    Line2D([], [], color="0.4", lw=1.0, ls=(0, (2, 1.5)), label="never grokked")],
    fontsize=8, loc="center right", framealpha=0.9)
fig.supxlabel("epoch (log)")
fig.supylabel("test accuracy")
fig.suptitle("Exp07 (frozen W_E) — grokking dynamics by n injected pairs "
             "(all 4 seeds; dotted = chance)")
pc.save(fig, FIG / "grok_dynamics_by_n.png", cap=(
    f"Test accuracy vs epoch, one panel per n, all {len(SEEDS)} seeds (dashed = "
    f"never grokked, dotted = chance 1/{P}). With the embedding frozen, the "
    "n=0 control mostly fails, but injecting >=3 oracle pairs restores fast, "
    f"reliable grokking — and as a SMOOTH ramp, not a delayed transition. {FZ}"))


# %% ----------------------------------------------------------------- #
# FIG 2 (the ask) — final W_E power spectrum per n: the FROZEN control
# --------------------------------------------------------------------- #
ncol = min(len(INJ_NS), 3)
nrow = int(np.ceil(len(INJ_NS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow),
                         squeeze=False)
freqs = np.arange(1, L + 1)
for ax, n in zip(axes.flat, INJ_NS):
    sel = pc.select(runs, n=n)
    specs = [np.asarray(pc.final_snap(r)["we_freq_power_full"], float)
             for r in sel if pc.final_snap(r).get("we_freq_power_full")]
    if not specs:
        continue
    for j, f in enumerate(INJ[n]):
        ax.axvline(f, color="green", ls="--", lw=1.0, alpha=0.6,
                   label="injected" if j == 0 else None)
    for sp in specs:
        ax.plot(freqs, sp, color=COL[n], alpha=0.3, lw=0.9)
    ax.plot(freqs, np.mean(specs, 0), color=COL[n], lw=1.8)
    ax.set_title(f"n = {n}{n_tag(sel)}", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True)
for ax in axes.flat[len(INJ_NS):]:
    ax.set_visible(False)
fig.supxlabel(f"Fourier frequency index (1..{L})")
fig.supylabel("final W_E power")
fig.suptitle(f"Exp07 (frozen W_E) — the embedding spectrum stays at random "
             f"init: NO uptake (p={P})", fontsize=12.5)
pc.save(fig, FIG / "we_spectrum_by_n.png", cap=(
    f"Final W_E Fourier power spectrum ({L} freqs) per n; 4 seeds faint + mean "
    "bold, green dashed = injected frequencies. The spectrum is FLAT at its "
    "random-init level with NO peaks on the injected sites — the frozen "
    "embedding cannot adopt them, unlike the trainable-W_E exp01/exp08 where "
    f"power concentrates on exactly these lines. Grokking still happens. {FZ}"))


# %% ----------------------------------------------------------------- #
# FIG 3 — W_E power fraction on injected freqs vs epoch: stays flat (no uptake)
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, len(INJ_NS), figsize=(4.2 * len(INJ_NS), 4.2),
                         sharex=True, sharey=True, squeeze=False)
for ax, n in zip(axes.flat, INJ_NS):
    sel = pc.select(runs, n=n)
    drawn = pc.seed_family(
        ax, sel,
        x_fn=lambda r: np.clip(pc.snap_series(r, pc.frac_we_power_injected)[0], 1, None),
        y_fn=lambda r: pc.snap_series(r, pc.frac_we_power_injected)[1],
        color=COL[n], alpha_seed=0.3)
    # reference: random-init expectation = n freqs out of L bins
    ax.axhline(n / L, color="0.5", ls="--", lw=1.0)
    ax.set_title(f"n = {n}{n_tag(sel)}", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(1, CENSOR)
    ax.set_ylim(-0.03, 1.05)
axes.flat[-1].legend(handles=[
    Line2D([], [], color="0.5", lw=1.0, ls="--", label="random-init level n/(p//2)")],
    fontsize=8, loc="upper left", framealpha=0.9)
fig.supxlabel("epoch (log)")
fig.supylabel("fraction of W_E Fourier power on injected freqs")
fig.suptitle("Exp07 (frozen W_E) — no embedding uptake: the injected-freq power "
             "fraction never rises", fontsize=12)
pc.save(fig, FIG / "no_uptake_by_n.png", cap=(
    "Fraction of W_E Fourier power sitting on the injected freqs vs epoch, per "
    "n (4 seeds + mean). It pins to the random-init level (dashed = n/(p//2)) "
    "for the whole run — the frozen embedding never concentrates power on the "
    "oracle, the direct contrast to the trainable-W_E uptake in exp01/exp08. "
    f"{FZ}"))


# %% ----------------------------------------------------------------- #
# FIG 4 — dependence on the live oracle: final acc ON vs OFF per n (grokked)
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 5))
rng = np.random.default_rng(0)
ON_C, OFF_C = "#1f77b4", "#d62728"
for fn, color in [(lambda s: pc.ablation(s, "acc_on"), ON_C),
                  (lambda s: pc.ablation(s, "acc_off"), OFF_C)]:
    means, mx = [], []
    for n in INJ_NS:
        sub = [r for r in pc.select(runs, n=n) if pc.groked(r)]
        vals = [fn(pc.final_snap(r)) for r in sub]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        ax.scatter(n + rng.uniform(-0.18, 0.18, len(vals)), vals, color=color,
                   alpha=0.5, s=34, zorder=2)
        means.append(np.mean(vals))
        mx.append(n)
    if means:
        ax.plot(mx, means, color=color, lw=1.9, marker="o", ms=6, zorder=3)
ax.set(xlabel="n injected pairs", ylabel="final test accuracy",
       title="Exp07 (frozen W_E) — dependence on the live oracle (grokked only)",
       ylim=(-0.03, 1.05))
ax.set_xticks(INJ_NS)
ax.legend(handles=[Line2D([], [], color=ON_C, lw=2, marker="o", label="oracle ON"),
                   Line2D([], [], color=OFF_C, lw=2, marker="o", label="oracle OFF")],
          fontsize=9, framealpha=0.9, loc="center right")
pc.save(fig, FIG / "dependence_vs_n.png", cap=(
    "Final test accuracy with the live oracle ON (blue) vs switched OFF at "
    "inference (red) vs n, grokked seeds (dots = seeds, line = mean). With a "
    "frozen embedding there is no learned Fourier structure in W_E to fall back "
    "on, so the model leans heavily on the live oracle: switching it off "
    f"collapses accuracy (large ON-OFF gap = high dependence). {FZ}"))

print(f"wrote {len(list(FIG.glob('*.png')))} figures -> {FIG}")
