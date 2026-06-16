"""Exp02_1 (delayed injection) figures — minimalistic, subplot-only.

The oracle is held OFF until epoch T (delay), then switched ON. Question: does a
late-arriving oracle get adopted / drive grokking? Two delays T in {4000, 8000}
x n in {1,2,3,5,6,8} x 4 seeds. exp02_1 has no n=0 of its own (nothing to inject
at epoch 0), so the no-oracle reference is taken from exp01's n=0 — the same
model (p=113, frac_train=0.3, 30k) with the oracle never injected — and drawn
translucently in every figure.

One panel per n (2x3). Injection time shown as two colours (blue=4000,
orange=8000) with vertical lines at T; the ablation is split into one figure per
T (pooling the two T is meaningless — 'before T' differs between them). Run:
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
BASE_C = "0.45"                                 # n=0 baseline (no oracle)
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


def _mean_curve(runs, x_fn, y_fn):
    """Seed-mean (x, y) over a run list, interpolated onto the densest grid."""
    series = [(x_fn(r), y_fn(r)) for r in runs]
    series = [(x, y) for x, y in series if len(x) and len(x) == len(y)]
    if not series:
        return None
    grid = max((s[0] for s in series), key=len)
    return grid, np.vstack([np.interp(grid, x, y) for x, y in series]).mean(0)


def _baseline_acc_drawer(base):
    """Returns f(ax, n) drawing the no-oracle baseline test-acc curve."""
    curve = _mean_curve(
        base,
        lambda r: np.clip(pc.hist_series(r, "test_acc")[0], 1, None),
        lambda r: pc.hist_series(r, "test_acc")[1]) if base else None

    def draw(ax, n):
        if curve:
            ax.plot(*curve, color=BASE_C, alpha=0.5, lw=1.7, zorder=1)
    return draw


def _baseline_uptake_drawer(base, runs):
    """Returns f(ax, n) drawing the no-oracle W_E power fraction on n's injected
    freqs — i.e. what a model that never saw the oracle naturally puts there."""
    def draw(ax, n):
        if not base:
            return
        sel = pc.select(runs, n=n)
        inj = sel[0]["_axes"].get("freqs", []) if sel else []
        if not inj:
            return
        fn = lambda s: (sum(s["we_freq_power_full"][f - 1] for f in inj) /
                        sum(s["we_freq_power_full"])
                        if s.get("we_freq_power_full") else None)
        curve = _mean_curve(base,
                            lambda r: np.clip(pc.snap_series(r, fn)[0], 1, None),
                            lambda r: pc.snap_series(r, fn)[1])
        if curve:
            ax.plot(*curve, color=BASE_C, alpha=0.5, lw=1.7, zorder=1)
    return draw


def by_delay_panels(runs, ns, outdir, x_fn, y_fn, ylabel, title, name, cap,
                    baseline_draw, chance=False):
    """One panel per n; the two delays as two colours (seeds faint + mean),
    plus the translucent n=0 baseline."""
    fig, axes = _grid(ns)
    for ax, n in zip(axes.flat, ns):
        baseline_draw(ax, n)
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
                        Line2D([], [], color=T_COL[8000], lw=2, label="inject @ T=8000"),
                        Line2D([], [], color=BASE_C, lw=2, alpha=0.6, label="n=0 baseline (no oracle)")],
               loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
    fig.supxlabel("epoch (log)")
    fig.supylabel(ylabel)
    fig.suptitle(title, fontsize=12.5)
    pc.save(fig, outdir / name, cap=cap)


def fig_ablation_by_T(runs, ns, outdir, baseline_draw):
    """ON vs OFF accuracy per n — a SEPARATE figure per injection time T, so the
    before-T (ON≈OFF) vs after-T (gap opens) structure stays coherent."""
    on_x, on_y = _snap(lambda s: (s.get("ablation_test") or {}).get("acc_on"))
    off_x, off_y = _snap(lambda s: (s.get("ablation_test") or {}).get("acc_off"))
    for T in T_COL:
        fig, axes = _grid(ns)
        for ax, n in zip(axes.flat, ns):
            baseline_draw(ax, n)
            sel = pc.select(runs, n=n, delay=T)
            pc.seed_family(ax, sel, x_fn=on_x, y_fn=on_y,
                           color=ON_C, alpha_seed=0.12, lw_mean=2.0)
            pc.seed_family(ax, sel, x_fn=off_x, y_fn=off_y,
                           color=OFF_C, alpha_seed=0.12, lw_mean=2.0)
            ax.axvline(T, color="0.25", ls="--", lw=1.1)
            ax.set_title(f"n = {n}", fontsize=10)
            ax.set_xscale("log")
            ax.set_xlim(left=120)
            ax.set_ylim(-0.03, 1.05)
        for ax in axes.flat[len(ns):]:
            ax.set_visible(False)
        fig.legend(handles=[Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
                            Line2D([], [], color=OFF_C, lw=2, label="oracle OFF"),
                            Line2D([], [], color="0.25", ls="--", lw=1.1, label=f"inject @ {T}"),
                            Line2D([], [], color=BASE_C, lw=2, alpha=0.6, label="n=0 baseline (no oracle)")],
                   loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True)
        fig.supxlabel("epoch (log)")
        fig.supylabel("test accuracy")
        fig.suptitle(f"Exp02_1 — ablation (oracle ON vs OFF), injection @ T={T} "
                     "(per n)", fontsize=12.5)
        pc.save(fig, outdir / f"ablation_T{T}.png", cap=(
            f"Accuracy with the live oracle ON (blue) vs OFF (red) at inference; "
            f"the oracle is injected at epoch {T} (dashed line), per n (4 seeds + "
            "mean; gray = no-oracle n=0 baseline). Before T the two curves nearly "
            "coincide (ON marginally lower — an injected signal the model wasn't "
            "trained on mildly perturbs it); the ON>OFF gap opens only AFTER T, as "
            "the model starts to use the oracle. OFF tracks the n=0 baseline — the "
            "model ends up nearly independent of the live signal."))


def fig_grok_vs_n(runs, ns, outdir, base):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rng = np.random.default_rng(0)
    bg = [pc.grok_epoch(r) for r in base] if base else []
    bg = [v for v in bg if v is not None]
    if bg:
        ax.axhspan(min(bg), max(bg), color=BASE_C, alpha=0.13, zorder=0)
        ax.axhline(np.mean(bg), color=BASE_C, ls="--", lw=1.3, alpha=0.7, zorder=1)
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
                       Line2D([], [], marker="o", ls="", color=T_COL[8000], label="T=8000"),
                       Line2D([], [], color=BASE_C, ls="--", lw=1.3, alpha=0.7, label="n=0 baseline grok (band)")],
              fontsize=9, loc="upper right")
    pc.save(fig, outdir / "grok_vs_n.png", cap=(
        "Grok epoch vs n for the two injection times (dots = seeds, bar = mean; "
        "dotted lines mark T; gray band = no-oracle n=0 baseline grok range). "
        "Both T grok within the no-oracle baseline band, far above their "
        "injection lines — grokking is on the model's own schedule, not "
        "injection-driven."))


def main(res):
    outdir = Path(res) / "figures" / EXP
    outdir.mkdir(parents=True, exist_ok=True)
    pc.set_style()
    runs = pc.load_exp(res, EXP)
    ns = pc.axis_values(runs, "n")
    try:
        base = pc.select(pc.load_exp(res, "exp01"), n=0)    # no-oracle reference
    except SystemExit:
        base = []
    base_acc = _baseline_acc_drawer(base)
    base_uptake = _baseline_uptake_drawer(base, runs)

    by_delay_panels(
        runs, ns, outdir,
        x_fn=lambda r: np.clip(pc.hist_series(r, "test_acc")[0], 1, None),
        y_fn=lambda r: pc.hist_series(r, "test_acc")[1],
        ylabel="test accuracy",
        title="Exp02_1 — grokking vs injection time T (per n)",
        name="grok_dynamics.png", chance=True, baseline_draw=base_acc,
        cap=("Test accuracy vs epoch, one panel per n; the oracle is injected at "
             "T (vertical lines: blue=4000, orange=8000). 4 seeds faint + "
             "seed-mean bold per T; gray = no-oracle n=0 baseline (exp01). Both T "
             "track the baseline and grok at ~the same absolute epoch — well "
             "after injection — so a late oracle does not trigger early grokking."))

    by_delay_panels(
        runs, ns, outdir,
        x_fn=lambda r: np.clip(pc.snap_series(r, pc.frac_we_power_injected)[0], 1, None),
        y_fn=lambda r: pc.snap_series(r, pc.frac_we_power_injected)[1],
        ylabel="fraction of W_E power on injected freqs",
        title="Exp02_1 — embedding uptake vs injection time T (per n)",
        name="uptake.png", baseline_draw=base_uptake,
        cap=("Fraction of W_E Fourier power on the injected freqs vs epoch, per "
             "n, by injection time T (vertical lines); gray = the same fraction "
             "for the no-oracle n=0 baseline (its natural power on those freqs). "
             "The injected runs sit at ~the baseline level with no jump at T — "
             "the embedding does not adopt the injected freqs when they arrive "
             "late."))

    fig_ablation_by_T(runs, ns, outdir, base_acc)
    fig_grok_vs_n(runs, ns, outdir, base)

    print(f"wrote {len(list(outdir.glob('*.png')))} figures to {outdir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_RES))
