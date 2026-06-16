"""Exp02_1 (delayed injection) figures — minimalistic, subplot-only.

The oracle is held OFF until epoch T (delay), then switched ON. Question: does a
late-arriving oracle get adopted / drive grokking? Two delays T in {4000, 8000}
x n in {1,2,3,5,6,8} x 4 seeds.

Redo of the earlier busy n-overlaid panels: now ONE panel per n (2x3 grid), with
the two injection times shown as two colours (blue=4000, orange=8000) and their
injection moments as vertical lines. Includes an ON-vs-OFF ablation figure in
the same style as the other experiments. Run:
    .venv/bin/python modular_addition/oracle/experiments/plot_exp02_1.py [results_dir]
"""
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc

EXP = "exp02_1"
CHANCE = 1.0 / 113
T_COL = {4000: "#1f77b4", 8000: "#e8820c"}      # injection time -> colour
ON_C, OFF_C = "#1f77b4", "#d62728"
DEFAULT_RES = (Path(__file__).resolve().parents[1] / "results" /
               "run_20260612_200000")


def _grid(ns, ncol=3, panel=(4.3, 2.9)):
    nrow = math.ceil(len(ns) / ncol)
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True,
                             figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False)
    return fig, axes


def _t_lines(ax):
    for t, c in T_COL.items():
        ax.axvline(t, color=c, ls=":", lw=1.0, alpha=0.5)


def _snap(fn):
    return (lambda r: np.clip(pc.snap_series(r, fn)[0], 1, None),
            lambda r: pc.snap_series(r, fn)[1])


def by_delay_panels(runs, ns, outdir, x_fn, y_fn, ylabel, title, name, cap,
                    chance=False):
    """One panel per n; the two delays drawn as two colours (seeds faint + mean)."""
    fig, axes = _grid(ns)
    for ax, n in zip(axes.flat, ns):
        for t, c in T_COL.items():
            pc.seed_family(ax, pc.select(runs, n=n, delay=t),
                           x_fn=x_fn, y_fn=y_fn, color=c,
                           alpha_seed=0.10, lw_mean=2.0)
        _t_lines(ax)
        if chance:
            ax.axhline(CHANCE, color="0.6", ls=":", lw=0.8)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(left=120)
        ax.set_ylim(-0.03, 1.05)
    for ax in axes.flat[len(ns):]:
        ax.set_visible(False)
    fig.legend(handles=[Line2D([], [], color=T_COL[4000], lw=2, label="inject @ T=4000"),
                        Line2D([], [], color=T_COL[8000], lw=2, label="inject @ T=8000")],
               loc="upper right", bbox_to_anchor=(0.995, 0.99), frameon=True)
    fig.supxlabel("epoch (log)")
    fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=12.5)
    pc.save(fig, outdir / name, cap=cap)


def fig_ablation(runs, ns, outdir):
    """ON vs OFF accuracy per n (both injection times pooled)."""
    ax_on_x, ax_on_y = _snap(lambda s: (s.get("ablation_test") or {}).get("acc_on"))
    ax_off_x, ax_off_y = _snap(lambda s: (s.get("ablation_test") or {}).get("acc_off"))
    fig, axes = _grid(ns)
    for ax, n in zip(axes.flat, ns):
        sel = pc.select(runs, n=n)
        pc.seed_family(ax, sel, x_fn=ax_on_x, y_fn=ax_on_y,
                       color=ON_C, alpha_seed=0.10, lw_mean=2.0)
        pc.seed_family(ax, sel, x_fn=ax_off_x, y_fn=ax_off_y,
                       color=OFF_C, alpha_seed=0.10, lw_mean=2.0)
        _t_lines(ax)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(left=120)
        ax.set_ylim(-0.03, 1.05)
    for ax in axes.flat[len(ns):]:
        ax.set_visible(False)
    fig.legend(handles=[Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
                        Line2D([], [], color=OFF_C, lw=2, label="oracle OFF")],
               loc="upper right", bbox_to_anchor=(0.995, 0.99), frameon=True)
    fig.supxlabel("epoch (log)")
    fig.supylabel("test accuracy")
    fig.suptitle("Exp02_1 — causal use: accuracy with the live oracle ON vs OFF "
                 "(per n; both T pooled)", fontsize=12.5)
    pc.save(fig, outdir / "ablation.png", cap=(
        "Accuracy with the live oracle ON (blue) vs switched OFF at inference "
        "(red), per n; runs from both injection times pooled (vertical lines = "
        "the two T). After grok acc_off ≈ acc_on ≈ 1 — the model is independent "
        "of the live oracle, so a delayed oracle is not used as a crutch."))


def fig_grok_vs_n(runs, ns, outdir):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rng = np.random.default_rng(0)
    for i, n in enumerate(ns):
        for t, c in T_COL.items():
            g = [pc.grok_epoch(r) for r in pc.select(runs, n=n, delay=t)]
            g = [v for v in g if v is not None]
            off = -0.12 if t == 4000 else 0.12
            if g:
                ax.scatter(i + off + rng.uniform(-0.05, 0.05, len(g)), g,
                           color=c, s=34, alpha=0.8, zorder=3)
                ax.scatter([i + off], [np.mean(g)], color=c, marker="_",
                           s=340, linewidths=2.4, zorder=4)
    for t, c in T_COL.items():
        ax.axhline(t, color=c, ls=":", lw=1.0, alpha=0.6)
    ax.set(xticks=range(len(ns)), xticklabels=ns, xlabel="n injected pairs",
           ylabel="grok epoch", yscale="log",
           title="Exp02_1 — grok epoch vs n, by injection time T")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=T_COL[4000], label="T=4000"),
                       Line2D([], [], marker="o", ls="", color=T_COL[8000], label="T=8000")],
              fontsize=9, loc="upper right")
    pc.save(fig, outdir / "grok_vs_n.png", cap=(
        "Grok epoch vs n for the two injection times (dots = seeds, bar = mean; "
        "dotted lines mark T). Both T grok at ~the same absolute epoch (far above "
        "their injection lines) regardless of when the oracle arrives — grokking "
        "is on the model's own schedule, not injection-driven."))


def main(res):
    outdir = Path(res) / "figures" / EXP
    outdir.mkdir(parents=True, exist_ok=True)
    pc.set_style()
    runs = pc.load_exp(res, EXP)
    ns = pc.axis_values(runs, "n")

    by_delay_panels(
        runs, ns, outdir,
        x_fn=lambda r: np.clip(pc.hist_series(r, "test_acc")[0], 1, None),
        y_fn=lambda r: pc.hist_series(r, "test_acc")[1],
        ylabel="test accuracy",
        title="Exp02_1 — grokking vs injection time T (per n)",
        name="grok_dynamics.png", chance=True,
        cap=("Test accuracy vs epoch, one panel per n; the oracle is injected at "
             "T (vertical lines: blue=4000, orange=8000). 4 seeds faint + "
             "seed-mean bold per T. Grokking happens at ~the same absolute epoch "
             "for both T — well after injection — so a late oracle does not "
             "trigger early grokking."))

    by_delay_panels(
        runs, ns, outdir,
        x_fn=lambda r: np.clip(pc.snap_series(r, pc.frac_we_power_injected)[0], 1, None),
        y_fn=lambda r: pc.snap_series(r, pc.frac_we_power_injected)[1],
        ylabel="fraction of W_E power on injected freqs",
        title="Exp02_1 — embedding uptake vs injection time T (per n)",
        name="uptake.png",
        cap=("Fraction of W_E Fourier power on the injected freqs vs epoch, per "
             "n, by injection time T (vertical lines). The fraction does not jump "
             "at T — the trainable embedding does not adopt the injected freqs "
             "when they arrive late."))

    fig_ablation(runs, ns, outdir)
    fig_grok_vs_n(runs, ns, outdir)

    print(f"wrote {len(list(outdir.glob('*.png')))} figures to {outdir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_RES))
