# %% [markdown]
# # Exp 02.1 — Delayed-injection figure suite
# Oracle is OFF before epoch T, ON at/after T. Hypothesis under test: the
# oracle DRIVES grokking, so the model should adopt injected freqs shortly
# AFTER T. These figures test that against the alternative that the model
# groks on its own ~absolute-time schedule and never depends on the oracle.
#
# Usage:  .venv/bin/python modular_addition/oracle/experiments/plot_exp02_1.py [results_dir]
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import plot_common as pc  # noqa: E402

EXP = "exp02_1"
DEFAULT_RES = str(_HERE.parent / "results" / "run_20260612_200000")
RES = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RES
OUT = Path(RES) / "figures" / EXP
DELAYS = [4000, 8000]


def main():
    pc.set_style()
    runs = pc.load_exp(RES, EXP)
    n_vals = pc.axis_values(runs, "n")
    cmap = pc.color_map(n_vals, baseline=None)        # color by n
    gr = pc.grok_rate(runs)
    print(f"loaded {len(runs)} runs from {RES}/{EXP}  grok_rate={gr:.2f}")

    n_handles = [Line2D([0], [0], color=cmap[n], lw=2, label=f"n={n}")
                 for n in n_vals]

    # ----------------------------------------------------------------- #
    # FIG 1 — test_acc vs epoch, 1x2 by T, color by n, vline at T
    # ----------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, T in zip(axes, DELAYS):
        for n in n_vals:
            sub = pc.select(runs, delay=T, n=n)
            pc.seed_family(ax, sub,
                           x_fn=lambda r: pc.hist_series(r, "test_acc")[0],
                           y_fn=lambda r: pc.hist_series(r, "test_acc")[1],
                           color=cmap[n], label=f"n={n}")
        ax.axvline(T, ls="--", color="crimson", lw=1.4, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(left=180)
        ax.set_xlabel("epoch (log)")
        ax.set_title(f"T = {T}   (red dashed = oracle ON)")
    axes[0].set_ylabel("test accuracy")
    axes[1].legend(handles=n_handles, title="# injected pairs",
                   loc="lower right", ncol=2, framealpha=0.9)
    fig.suptitle("Delayed injection — test accuracy vs epoch", y=1.0)
    pc.save(fig, OUT / "1_acc_vs_epoch_by_T.png",
            cap=("Test acc; red dashed = oracle turns ON at T. Grok onsets sit "
                 "near the SAME absolute epoch in both panels (not ~4000 later "
                 "for T=8000) — grokking does NOT follow injection at T."))

    # ----------------------------------------------------------------- #
    # FIG 2 — frac_we_power_injected vs epoch, 1x2 by T, vline at T
    # ----------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, T in zip(axes, DELAYS):
        for n in n_vals:
            sub = pc.select(runs, delay=T, n=n)
            pc.seed_family(ax, sub,
                           x_fn=lambda r: pc.snap_series(r, pc.frac_we_power_injected)[0],
                           y_fn=lambda r: pc.snap_series(r, pc.frac_we_power_injected)[1],
                           color=cmap[n], label=f"n={n}")
        ax.axvline(T, ls="--", color="crimson", lw=1.4, zorder=4)
        ax.set_xlabel("epoch")
        ax.set_title(f"T = {T}   (red dashed = oracle ON)")
    axes[0].set_ylabel(r"frac. of $W_E$ Fourier power on injected freqs")
    axes[1].legend(handles=n_handles, title="# injected pairs",
                   loc="upper right", ncol=2, framealpha=0.9)
    fig.suptitle(r"Uptake of injected freqs into $W_E$ vs epoch", y=1.0)
    pc.save(fig, OUT / "2_uptake_vs_epoch_by_T.png",
            cap=("Fraction of W_E power on injected freqs. Red dashed = T. It "
                 "DROPS through grok (model concentrates power on its OWN key "
                 "freqs) and shows no jump at T — embedding does not adopt the "
                 "oracle. Higher n starts higher (more freqs counted)."))

    # ----------------------------------------------------------------- #
    # FIG 3 — event-aligned uptake: x = epoch - T, overlay both T,
    #          subset n in {3,6}, T by linestyle, color by n, vline x=0
    # ----------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    subset_n = [m for m in (3, 6) if m in n_vals]
    ls_for = {4000: "-", 8000: "--"}
    for n in subset_n:
        for T in DELAYS:
            sub = pc.select(runs, delay=T, n=n)
            # event-align x by subtracting T inside the x_fn
            def xf(r, _T=T):
                e, _ = pc.snap_series(r, pc.frac_we_power_injected)
                return e - _T
            def yf(r):
                _, v = pc.snap_series(r, pc.frac_we_power_injected)
                return v
            # custom draw (linestyle encodes T); reuse seed_family logic inline
            use = [(xf(r), yf(r)) for r in sub]
            use = [(x, y) for x, y in use if len(x) and len(x) == len(y)]
            for x, y in use:
                ax.plot(x, y, color=cmap[n], alpha=0.22, lw=1.0, ls=ls_for[T])
            grid = max((s[0] for s in use), key=len)
            M = np.vstack([np.interp(grid, x, y) for x, y in use])
            ax.plot(grid, M.mean(0), color=cmap[n], lw=2.0, ls=ls_for[T],
                    zorder=3)
    ax.axvline(0, ls=":", color="crimson", lw=1.6, zorder=4)
    ax.text(0, ax.get_ylim()[1], "  injection moment", color="crimson",
            fontsize=8.5, va="top", ha="left")
    ax.set_xlabel("epoch − T  (0 = oracle turns ON)")
    ax.set_ylabel(r"frac. of $W_E$ power on injected freqs")
    ax.set_title("Event-aligned uptake (overlay T=4000 & T=8000)")
    handles = [Line2D([0], [0], color=cmap[n], lw=2, label=f"n={n}")
               for n in subset_n]
    handles += [Line2D([0], [0], color="0.3", lw=2, ls="-", label="T=4000"),
                Line2D([0], [0], color="0.3", lw=2, ls="--", label="T=8000")]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    pc.save(fig, OUT / "3_event_aligned_uptake.png",
            cap=("Uptake re-zeroed to the injection moment (x=epoch−T). The two "
                 "T curves do NOT collapse onto an injection-locked trajectory "
                 "and there is no post-x=0 jump — adoption is not driven by the "
                 "T switch (sanity check: the x-axis meaning differs by T)."))

    # ----------------------------------------------------------------- #
    # FIG 4 — ablation acc_off vs epoch, 1x2 by T, vline at T
    # ----------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, T in zip(axes, DELAYS):
        for n in n_vals:
            sub = pc.select(runs, delay=T, n=n)
            pc.seed_family(ax, sub,
                           x_fn=lambda r: pc.snap_series(r, lambda s: pc.ablation(s, "acc_off"))[0],
                           y_fn=lambda r: pc.snap_series(r, lambda s: pc.ablation(s, "acc_off"))[1],
                           color=cmap[n], label=f"n={n}")
        ax.axvline(T, ls="--", color="crimson", lw=1.4, zorder=4)
        ax.set_xlabel("epoch")
        ax.set_title(f"T = {T}   (red dashed = oracle ON)")
    axes[0].set_ylabel("ablation test acc with oracle OFF (independence)")
    axes[1].legend(handles=n_handles, title="# injected pairs",
                   loc="lower right", ncol=2, framealpha=0.9)
    fig.suptitle("Independence from the live oracle (acc with oracle OFF)", y=1.0)
    pc.save(fig, OUT / "4_ablation_accoff_vs_epoch_by_T.png",
            cap=("Test acc with the live oracle ABLATED. Red dashed = T. It "
                 "rises to ~1.0 at grok in both panels — the model performs "
                 "fully WITHOUT the oracle (independent), not dependent on it. "
                 "Rise tracks grok epoch, not T."))

    # ----------------------------------------------------------------- #
    # FIG 5 — grok lag (grok_epoch - T) vs n, T by marker/color
    # ----------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    T_color = {4000: "#1f77b4", 8000: "#d62728"}
    T_marker = {4000: "o", 8000: "s"}
    rng = np.random.default_rng(0)
    xpos = {n: i for i, n in enumerate(n_vals)}
    for T in DELAYS:
        dx = -0.12 if T == 4000 else 0.12
        means = []
        for n in n_vals:
            sub = pc.select(runs, delay=T, n=n)
            lags = [pc.grok_epoch(r) - T for r in sub if pc.groked(r)]
            jx = xpos[n] + dx + rng.uniform(-0.04, 0.04, size=len(lags))
            ax.scatter(jx, lags, color=T_color[T], marker=T_marker[T],
                       s=26, alpha=0.45, edgecolors="none")
            means.append(np.mean(lags))
        ax.plot([xpos[n] + dx for n in n_vals], means, color=T_color[T],
                marker=T_marker[T], ms=9, lw=1.8, zorder=4,
                label=f"T={T} (mean)")
    ax.axhline(0, ls="--", color="0.4", lw=1.2)
    ax.text(len(n_vals) - 1, 0, "grok AT injection ", color="0.4",
            fontsize=8.5, va="bottom", ha="right")
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([str(n) for n in n_vals])
    ax.set_xlabel("# injected freq pairs (n)")
    ax.set_ylabel("grok lag  =  grok_epoch − T   (epochs)")
    ax.set_title("Grok lag after injection vs n")
    ax.legend(loc="upper right", framealpha=0.9)
    pc.save(fig, OUT / "5_grok_lag_vs_T.png",
            cap=("grok_epoch − T per seed (jitter) + mean. If injection drove "
                 "grok, both T would share ~one lag; instead T=8000 lags are "
                 "FAR smaller than T=4000 (grok epoch ~T-invariant). The seed "
                 "stuck at lag=−3400 for T=8000 (groks @4600 before T) is the "
                 "self-grokking control."))

    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
