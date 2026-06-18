# %% [markdown]
# # Exp 05 figures — weakly-informative answer hint
#
# Inject a low-information hint about the answer c=(i+j) mod p at the "=" readout
# position: c % 10 ("mod") or c // 10 ("div"), encoded "onehot" (strong/sparse)
# or "fourier" (weak/distributed), vs a no-hint "baseline".
#
# CRITICAL: the hint LEAKS the label, so high final accuracy is trivial and is
# NOT the result. We read accuracy only for TIMING (acceleration). The real
# signals are: acceleration vs baseline, solution simplification / laziness
# (fewer key freqs, lower W_E norm, higher Gini), and dependence on the live
# hint (ablation acc_off). The "hint" axis is categorical; baseline has no
# ablation (ablation_test=None) and is excluded from the dependence figure.
#
# injected_freqs is EMPTY for these runs -> all injected_* metrics are skipped.

# %% imports + path bootstrap
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
import plot_common as pc  # noqa: E402

EXP = "exp05"
DEFAULT_RES = ("/Users/kaushikreddy/Projects/oracle-encoding-project/"
               "oracle-encodings/modular_addition/oracle/results/run_20260612_200000")

# fixed categorical color per config (consistent across every figure)
CONFIGS = ["baseline", "hint_mod10_onehot", "hint_div10_onehot",
           "hint_mod10_fourier"]
COLOR = {
    "baseline":           "0.45",       # gray
    "hint_mod10_onehot":  "#d62728",    # red
    "hint_div10_onehot":  "#1f77b4",    # blue
    "hint_mod10_fourier": "#2ca02c",    # green
}
PRETTY = {
    "baseline":           "baseline (no hint)",
    "hint_mod10_onehot":  "mod10 · onehot",
    "hint_div10_onehot":  "div10 · onehot",
    "hint_mod10_fourier": "mod10 · fourier",
}


def present_configs(runs):
    """CONFIGS that actually have runs, in canonical order."""
    have = set(pc.axis_values(runs, "hint"))
    return [c for c in CONFIGS if c in have]


def _jitter(n, width=0.13, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) - 0.5) * 2 * width


def strip_by_config(ax, runs, value_fn, configs, *, log=False,
                    skip_none_configs=()):
    """Jittered per-seed strip + mean marker, one column per config.

    value_fn(run)->float or None. Configs in skip_none_configs are still
    plotted as an empty (excluded) column tick so the reader sees the gap.
    Returns dict config->list of values actually plotted.
    """
    out = {}
    for xi, cfg in enumerate(configs):
        rs = pc.select(runs, hint=cfg)
        vals = [v for v in (value_fn(r) for r in rs) if v is not None]
        out[cfg] = vals
        if not vals:
            # excluded / no-data column: mark it explicitly
            ax.annotate("n/a", (xi, ax.get_ylim()[0]), ha="center",
                        va="bottom", fontsize=8, color="0.6")
            continue
        x = xi + _jitter(len(vals), seed=xi)
        ax.scatter(x, vals, s=46, color=COLOR[cfg], alpha=0.6,
                   edgecolor="white", linewidth=0.6, zorder=3)
        m = float(np.mean(vals))
        ax.scatter([xi], [m], marker="D", s=95, color=COLOR[cfg],
                   edgecolor="black", linewidth=1.0, zorder=5)
        ax.hlines(m, xi - 0.22, xi + 0.22, color="black", lw=1.0, zorder=4)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([PRETTY[c] for c in configs], rotation=18, ha="right")
    if log:
        ax.set_yscale("log")
    ax.margins(x=0.12)
    return out


def legend_handles(configs):
    return [Line2D([0], [0], color=COLOR[c], lw=2.4, label=PRETTY[c])
            for c in configs]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_acceleration(runs, configs, outdir):
    """test_acc vs epoch (log-x) + grok_epoch strip by config."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    for cfg in configs:
        rs = pc.select(runs, hint=cfg)
        n = pc.seed_family(
            axL, rs,
            x_fn=lambda r: pc.hist_series(r, "epoch")[1],
            y_fn=lambda r: pc.hist_series(r, "test_acc")[1],
            color=COLOR[cfg], label=f"{PRETTY[cfg]} (n={len(rs)})")
    axL.set_xscale("log")
    axL.set_xlabel("epoch (log)")
    axL.set_ylabel("test accuracy")
    axL.set_title("Learning curves — timing only (label is leaked)")
    axL.axhline(1.0, color="0.8", lw=0.8, ls=":")
    axL.legend(loc="lower right", fontsize=8)

    # right: grok-epoch strip (log-y)
    strip_by_config(axR, runs, pc.grok_epoch, configs, log=True)
    axR.set_ylabel("grok epoch (log)")
    axR.set_title("Grok epoch by config (seeds + mean)")

    cap = ("Accuracy used ONLY for timing: the answer hint LEAKS the label, so "
           "near-perfect final accuracy is trivial and is NOT the result. "
           "Read whether a config groks SOONER than baseline (gray).")
    pc.save(fig, Path(outdir) / "acceleration.png", cap=cap)

    # also write the grok-epoch strip as a standalone file
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))
    strip_by_config(ax2, runs, pc.grok_epoch, configs, log=True)
    ax2.set_ylabel("grok epoch (log)")
    ax2.set_title("exp05 — grok epoch by config")
    pc.save(fig2, Path(outdir) / "grok_epoch_by_config.png",
            cap=("Lower = groks sooner. Hint accelerates only if a config sits "
                 "BELOW baseline (gray); the hint leaks the label so final "
                 "accuracy is not the result."))


def fig_laziness_nkeyfreqs(runs, configs, outdir):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    strip_by_config(ax, runs, lambda r: pc.n_key_freqs(pc.final_snap(r)),
                    configs)
    ax.set_ylabel("# key frequencies (final)")
    ax.set_title("exp05 — solution complexity: # working frequencies")
    cap = ("Fewer key frequencies than baseline (gray) => the hint SIMPLIFIES "
           "the learned solution. A label-leaking hint can let the model solve "
           "with fewer frequencies (laziness).")
    pc.save(fig, Path(outdir) / "laziness_nkeyfreqs.png", cap=cap)


def fig_laziness_we(runs, configs, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    strip_by_config(a1, runs, lambda r: pc.final_snap(r).get("we_total_norm"),
                    configs)
    a1.set_ylabel("W_E total norm (final)")
    a1.set_title("Embedding norm")
    strip_by_config(a2, runs, lambda r: pc.final_snap(r).get("we_gini"),
                    configs)
    a2.set_ylabel("W_E Gini (final)")
    a2.set_title("Embedding concentration (Gini)")
    cap = ("Lower W_E norm and/or higher Gini than baseline (gray) => a lazier, "
           "more concentrated embedding (the model leans on the hint instead of "
           "building a full embedding).")
    pc.save(fig, Path(outdir) / "laziness_we.png", cap=cap)


def fig_dependence_accoff(runs, configs, outdir):
    """Final ablation acc_off by config, EXCLUDING baseline (no ablation)."""
    hint_cfgs = [c for c in configs if c != "baseline"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    strip_by_config(ax, runs,
                    lambda r: pc.ablation(pc.final_snap(r), "acc_off"),
                    hint_cfgs)
    ax.set_ylabel("accuracy with hint ablated (acc_off)")
    ax.set_title("exp05 — dependence on the live hint")
    ax.axhline(1.0, color="0.8", lw=0.8, ls=":")
    cap = ("Baseline EXCLUDED (no hint -> no ablation). LOW acc_off => the model "
           "DEPENDS on the live hint as a crutch. Onehot vs fourier are not "
           "comparable on signal strength; compare dependence, not accuracy.")
    pc.save(fig, Path(outdir) / "dependence_accoff.png", cap=cap)


def fig_spectrum_compare(runs, configs, outdir):
    """Mean final W_E Fourier-power spectrum overlaid per config (56 freqs)."""
    fig, (axS, axH) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    # left: mean normalized W_E freq-power spectrum, one line per config
    nfreq = None
    for cfg in configs:
        rs = pc.select(runs, hint=cfg)
        specs = []
        for r in rs:
            p = pc.final_snap(r).get("we_freq_power_full")
            if not p:
                continue
            p = np.asarray(p, float)
            tot = p.sum()
            specs.append(p / tot if tot > 0 else p)
        if not specs:
            continue
        M = np.vstack(specs)
        nfreq = M.shape[1]
        freqs = np.arange(1, nfreq + 1)
        mean = M.mean(0)
        sd = M.std(0)
        axS.plot(freqs, mean, color=COLOR[cfg], lw=1.9, label=PRETTY[cfg])
        # power is non-negative: clip the lower band at 0
        axS.fill_between(freqs, np.clip(mean - sd, 0, None), mean + sd,
                         color=COLOR[cfg], alpha=0.12)
    axS.set_xlabel("frequency index (1..%d)" % (nfreq or 56))
    axS.set_ylabel("mean normalized W_E power")
    axS.set_title("Learned frequency spectrum (shape) by config")
    axS.legend(fontsize=8)

    # right: distribution of #key-freqs by config (echoes fig2, as a check on
    # whether the hint changes the solution shape and not only its speed)
    strip_by_config(axH, runs, lambda r: pc.n_key_freqs(pc.final_snap(r)),
                    configs)
    axH.set_ylabel("# key frequencies (final)")
    axH.set_title("# working frequencies by config")

    cap = ("Does the hint change the SHAPE of the solution (which/how many "
           "frequencies) or only its speed? A flatter / sparser spectrum vs "
           "baseline (gray) indicates a structurally different, simpler basis.")
    pc.save(fig, Path(outdir) / "spectrum_compare.png", cap=cap)


def fig_we_spectrum(runs, configs, outdir):
    """Final W_E Fourier spectrum per config, one panel each (4 seeds + mean).

    The answer hint is injected at the readout, NOT into W_E, so there are no
    'injected' frequencies to mark — this asks whether the hint reshapes the
    LEARNED embedding basis relative to baseline.
    """
    fig, axes = plt.subplots(1, len(configs), figsize=(3.5 * len(configs), 3.7),
                             sharex=True, sharey=True, squeeze=False)
    for ax, cfg in zip(axes[0], configs):
        rs = pc.select(runs, hint=cfg)
        specs = [np.asarray(pc.final_snap(r)["we_freq_power_full"], float)
                 for r in rs if pc.final_snap(r).get("we_freq_power_full")]
        if specs:
            freqs = np.arange(1, len(specs[0]) + 1)
            for sp in specs:
                ax.plot(freqs, sp, color=COLOR[cfg], alpha=0.30, lw=0.8)
            ax.plot(freqs, np.mean(specs, 0), color=COLOR[cfg], lw=1.8)
        gk = sum(pc.groked(r) for r in rs)
        ax.set_title(f"{PRETTY[cfg]}  ({gk}/{len(rs)} grok)", fontsize=9.5)
        ax.set_xlabel("freq index (1..56)")
    axes[0][0].set_ylabel("final W_E Fourier power")
    fig.suptitle("exp05 — final W_E spectrum by hint config (4 seeds + mean)",
                 fontsize=12.5)
    cap = ("Final W_E Fourier power spectrum per config (4 seeds faint + mean "
           "bold). The hint injects NOTHING into W_E, so the question is whether "
           "it reshapes the LEARNED basis vs baseline (gray): a sparser / lower "
           "comb = a simpler, lazier embedding. Companion to spectrum_compare.")
    pc.save(fig, Path(outdir) / "we_spectrum.png", cap=cap)


def fig_ablation(runs, configs, outdir):
    """ON-vs-ablated accuracy trajectory per hint config (baseline excluded)."""
    ON_C, OFF_C = "#1f77b4", "#d62728"
    hint_cfgs = [c for c in configs if c != "baseline"]

    def snap_y(key):
        return lambda r: pc.snap_series(
            r, lambda s: (s.get("ablation_test") or {}).get(key))[1]

    def snap_x(key):
        return lambda r: np.clip(pc.snap_series(
            r, lambda s: (s.get("ablation_test") or {}).get(key))[0], 1, None)

    fig, axes = plt.subplots(1, len(hint_cfgs),
                             figsize=(4.3 * len(hint_cfgs), 4.3),
                             sharex=True, sharey=True, squeeze=False)
    for ax, cfg in zip(axes[0], hint_cfgs):
        rs = pc.select(runs, hint=cfg)
        pc.seed_family(ax, rs, x_fn=snap_x("acc_on"), y_fn=snap_y("acc_on"),
                       color=ON_C, alpha_seed=0.18, lw_mean=1.9)
        pc.seed_family(ax, rs, x_fn=snap_x("acc_off"), y_fn=snap_y("acc_off"),
                       color=OFF_C, alpha_seed=0.18, lw_mean=1.9)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(PRETTY[cfg], fontsize=10)
        ax.set_xlabel("epoch (log)")
    axes[0][0].set_ylabel("test accuracy")
    fig.legend(handles=[
        Line2D([], [], color=ON_C, lw=2, label="hint live (status-quo)"),
        Line2D([], [], color=OFF_C, lw=2, label="hint ablated (forced off)")],
        loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=True,
        fontsize=8)
    fig.suptitle("exp05 — dependence on the live hint: accuracy live vs ablated",
                 fontsize=12.5)
    cap = ("Accuracy over training with the hint live (blue) vs ablated at "
           "inference (red), per hint config; baseline has no hint to ablate. "
           "The hint LEAKS the label so blue rises to ~1 trivially — the "
           "blue-red GAP is the model's dependence on the live hint (large gap "
           "= crutch). Trajectory companion to dependence_accoff.")
    pc.save(fig, Path(outdir) / "ablation.png", cap=cap)


# --------------------------------------------------------------------------- #
def main():
    pc.set_style()
    res = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RES
    runs = pc.load_exp(res, EXP)
    configs = present_configs(runs)
    outdir = Path(res) / "figures" / EXP
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[exp05] {len(runs)} runs; configs={configs}")
    for cfg in configs:
        rs = pc.select(runs, hint=cfg)
        gr = [pc.grok_epoch(r) for r in rs]
        print(f"  {cfg:<20} n={len(rs)} grok={gr}")

    fig_acceleration(runs, configs, outdir)
    fig_laziness_nkeyfreqs(runs, configs, outdir)
    fig_laziness_we(runs, configs, outdir)
    fig_dependence_accoff(runs, configs, outdir)
    fig_spectrum_compare(runs, configs, outdir)
    fig_we_spectrum(runs, configs, outdir)
    fig_ablation(runs, configs, outdir)

    print(f"[exp05] wrote figures to {outdir}")
    for p in sorted(outdir.glob("*.png")):
        print("   ", p.name)


if __name__ == "__main__":
    main()
