# %% [markdown]
# # Exp06 figure suite — exp01's analyses rebuilt on richer data
# exp06 strictly dominates exp01 as a data source: a finer, contiguous n-sweep
# (n=0..11 vs exp01's [0,1,2,3,5,6,8]), a larger model (p=211 -> 105 Fourier
# freqs; d_model=256), and a low-data regime (frac_train=0.075, 75k ep) where
# the n=0 baseline CANNOT grok — so "completeness" becomes a clean rescue test.
#
# We test exactly what exp01 tested, as a CURATED set of subplot-only figures:
#   1 grok_dynamics  — completeness/rescue threshold (test acc per n)
#   2 uptake         — W_E adopts the injected freqs (power fraction per n)
#   3 spectrum       — exact uptake at injected sites (final W_E spectrum per n)
#   4 ablation       — causal use: accuracy with the live oracle ON vs OFF per n
#   5 sufficiency    — do the injected freqs alone explain the model? (trig loss)
#   6 summary_vs_n   — mechanistic dashboard across n (incl. the kept-set ceiling)
#
# NOTE: `excluded_loss_injected` (exp01's "necessity" metric) is 100% NaN at
# p=211 (numerical blow-up in calculate_excluded_loss at the larger prime), so
# necessity is carried by the ablation figure (causal removal of the live
# signal) instead. Run:
#   .venv/bin/python modular_addition/oracle/experiments/plot_exp06.py [results_dir]
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc

EXP = "exp06"
CHANCE = 1.0 / 211
CENSOR = 75_000
P113_CEILING = 5            # working-set ceiling observed at the smaller p=113 model
ON_C, OFF_C = "#1f77b4", "#d62728"
DEFAULT_RES = ("/Users/kaushikreddy/Projects/oracle-encoding-project/"
               "oracle-encodings/modular_addition/oracle/results/run_20260612_200000")


def jitter(x, n, w=0.13):
    return np.full(n, float(x)) if n <= 1 else x + np.linspace(-w, w, n)


def _grid(ns, ncol=4, panel=(3.6, 2.7), sharey=True):
    nrow = math.ceil(len(ns) / ncol)
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=sharey,
                             figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False)
    return fig, axes


def _hide_extra(axes, ns):
    for ax in axes.flat[len(ns):]:
        ax.set_visible(False)


def _snap_y(fn):
    """y_fn for seed_family from a snapshot->scalar fn (epoch clipped >=1)."""
    return (lambda r: pc.snap_series(r, fn)[1])


def _snap_x(fn):
    return (lambda r: np.clip(pc.snap_series(r, fn)[0], 1, None))


def _finite(xs):
    return [v for v in xs if v is not None and v == v]   # drop None + NaN


# --------------------------------------------------------------------------- #
# 1. grokking dynamics / completeness threshold
# --------------------------------------------------------------------------- #
def fig_grok_dynamics(runs, outdir):
    ns = list(range(12))
    cmap = pc.color_map(ns, baseline=0)
    fig, axes = _grid(ns)
    for ax, n in zip(axes.flat, ns):
        sel = pc.select(runs, n=n)
        pc.seed_family(ax, sel,
                       x_fn=lambda r: np.clip(pc.hist_series(r, "test_acc")[0], 1, None),
                       y_fn=lambda r: pc.hist_series(r, "test_acc")[1],
                       color=cmap[n], alpha_seed=0.25)
        ax.axhline(CHANCE, color="0.6", ls=":", lw=0.8)
        gk = _finite([pc.grok_epoch(r) for r in sel])
        txt = (f"{len(gk)}/4 grok\n~{int(np.median(gk))} ep" if gk
               else "0/4\nnever groks")
        ax.text(0.95, 0.14, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=("0.25" if gk else "#b00020"))
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(1, CENSOR)
        ax.set_ylim(-0.03, 1.05)
    _hide_extra(axes, ns)
    fig.supxlabel("epoch (log)")
    fig.supylabel("test accuracy")
    fig.suptitle("Exp06 — completeness threshold: injected pairs RESCUE grokking "
                 "(p=211, frac_train=0.075, 4 seeds)", fontsize=12.5)
    pc.save(fig, outdir / "grok_dynamics.png", cap=(
        "Test accuracy vs epoch, one panel per n (4 seeds faint + seed-mean "
        "bold; dotted = chance, 1/211). In this low-data regime n=0,1,2 NEVER "
        "grok (a 1-2 pair basis is insufficient); generalization is rescued "
        "sharply at n=3 (all 4 seeds, ~1.1k ep). n=4 is the unstable transition "
        "(one seed crawls to ~55k); n>=5 grok fast and tight."))


# --------------------------------------------------------------------------- #
# 2. uptake into the trainable embedding
# --------------------------------------------------------------------------- #
def fig_uptake(runs, outdir):
    ns = list(range(1, 12))                 # n=0 has no injected freqs
    cmap = pc.color_map(list(range(12)), baseline=0)
    fig, axes = _grid(ns)
    for ax, n in zip(axes.flat, ns):
        sel = pc.select(runs, n=n)
        pc.seed_family(ax, sel,
                       x_fn=_snap_x(pc.frac_we_power_injected),
                       y_fn=_snap_y(pc.frac_we_power_injected),
                       color=cmap[n], alpha_seed=0.25)
        ax.set_title(f"n = {n}" + ("  (never groks)" if n in (1, 2) else ""),
                     fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(1, CENSOR)
        ax.set_ylim(-0.03, 1.05)
    _hide_extra(axes, ns)
    fig.supxlabel("epoch (log)")
    fig.supylabel("fraction of W_E Fourier power on injected freqs")
    fig.suptitle("Exp06 — the embedding ADOPTS the injected frequencies",
                 fontsize=12.5)
    pc.save(fig, outdir / "uptake.png", cap=(
        "Fraction of W_E Fourier power sitting on the injected freqs vs epoch, "
        "per n (4 seeds + mean). Grokking runs (n>=3) rapidly concentrate "
        "70-95% of embedding power onto the oracle's frequencies — direct "
        "uptake into trainable weights. n=1,2 (never grok) show weak/partial "
        "uptake: the embedding starts to adopt the freqs but the basis is too "
        "small to solve the task."))


# --------------------------------------------------------------------------- #
# 3. final W_E Fourier spectrum — exact uptake at injected sites
# --------------------------------------------------------------------------- #
def fig_spectrum(runs, outdir):
    ns = [2, 3, 5, 7, 9, 11]
    cmap = pc.color_map(list(range(12)), baseline=0)
    fig, axes = _grid(ns, ncol=3, panel=(4.6, 2.7), sharey=False)
    for ax, n in zip(axes.flat, ns):
        sel = pc.select(runs, n=n)
        specs = [np.asarray(pc.final_snap(r)["we_freq_power_full"], float)
                 for r in sel if pc.final_snap(r).get("we_freq_power_full")]
        if not specs:
            continue
        freqs = np.arange(1, len(specs[0]) + 1)
        inj = pc.final_snap(sel[0]).get("injected_freqs") or \
            sel[0]["_axes"].get("freqs", [])
        for j, f in enumerate(inj):
            ax.axvline(f, color="green", ls="--", lw=1.0, alpha=0.6,
                       label="injected" if j == 0 else None)
        for sp in specs:
            ax.plot(freqs, sp, color=cmap[n], alpha=0.3, lw=0.9)
        ax.plot(freqs, np.mean(specs, 0), color=cmap[n], lw=1.7)
        grk = sum(pc.groked(r) for r in sel)
        ax.set_title(f"n = {n}  ({grk}/4 grok)", fontsize=10)
        ax.legend(loc="upper right", fontsize=7.5, frameon=True)
    _hide_extra(axes, ns)
    fig.supxlabel("Fourier frequency index (1..105)")
    fig.supylabel("final W_E power")
    fig.suptitle("Exp06 — power piles up exactly on the injected sites "
                 "(final W_E spectrum, p=211)", fontsize=12.5)
    pc.save(fig, outdir / "spectrum.png", cap=(
        "Final W_E Fourier power spectrum (105 freqs) per n; 4 seeds faint + "
        "mean bold, green dashed = injected frequencies. For grokking n the "
        "peaks land almost exactly on the injected sites with little leakage. "
        "n=2 (never groks) has a diffuse spectrum — no clean structure forms. "
        "As n grows the model lights up more injected sites (but not all carry "
        "equal power — see the kept-set in fig 6)."))


# --------------------------------------------------------------------------- #
# 4. ablation: causal use of the live oracle (accuracy ON vs OFF)
# --------------------------------------------------------------------------- #
def fig_ablation(runs, outdir):
    ns = list(range(1, 12))
    fig, axes = _grid(ns)
    for ax, n in zip(axes.flat, ns):
        sel = pc.select(runs, n=n)
        pc.seed_family(ax, sel,
                       x_fn=_snap_x(lambda s: (s.get("ablation_test") or {}).get("acc_on")),
                       y_fn=_snap_y(lambda s: (s.get("ablation_test") or {}).get("acc_on")),
                       color=ON_C, alpha_seed=0.15, lw_mean=1.7)
        pc.seed_family(ax, sel,
                       x_fn=_snap_x(lambda s: (s.get("ablation_test") or {}).get("acc_off")),
                       y_fn=_snap_y(lambda s: (s.get("ablation_test") or {}).get("acc_off")),
                       color=OFF_C, alpha_seed=0.15, lw_mean=1.7)
        ax.set_title(f"n = {n}" + ("  (never groks)" if n in (1, 2) else ""),
                     fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(1800, CENSOR)
        ax.set_ylim(-0.03, 1.05)
    _hide_extra(axes, ns)
    fig.legend(handles=[Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
                        Line2D([], [], color=OFF_C, lw=2, label="oracle OFF")],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=True)
    fig.supxlabel("epoch (log)")
    fig.supylabel("test accuracy")
    fig.suptitle("Exp06 — causal use: accuracy with the live oracle ON vs OFF",
                 fontsize=12.5)
    pc.save(fig, outdir / "ablation.png", cap=(
        "Accuracy with the live oracle ON (blue) vs switched OFF at inference "
        "(red), per n (4 seeds + mean). The ON-OFF gap = how much the model "
        "leans on the live signal. Striking: the gap WIDENS with n — at n>=8 "
        "the model depends on the oracle MORE (acc_off ~0.7) than at n=3 "
        "(acc_off ~0.9), the OPPOSITE of the p=113 trend where larger bases "
        "became ablation-proof. n=1,2 sit at chance for both (nothing learned)."))


# --------------------------------------------------------------------------- #
# 5. sufficiency: do the injected freqs alone explain the model?
# --------------------------------------------------------------------------- #
def fig_sufficiency(runs, outdir):
    ns = list(range(1, 12))
    fig, axes = _grid(ns)
    for ax, n in zip(axes.flat, ns):
        sel = pc.select(runs, n=n)
        # full test loss (reference), then trig-loss restricted to injected
        # freqs (sufficiency) and to the model's own key freqs.
        pc.seed_family(ax, sel,
                       x_fn=lambda r: np.clip(pc.hist_series(r, "test_loss")[0], 1, None),
                       y_fn=lambda r: pc.hist_series(r, "test_loss")[1],
                       color="0.6", alpha_seed=0.12, lw_mean=1.3)
        pc.seed_family(ax, sel,
                       x_fn=_snap_x(lambda s: s.get("trig_loss_keyfreqs")),
                       y_fn=_snap_y(lambda s: s.get("trig_loss_keyfreqs")),
                       color="#2ca02c", alpha_seed=0.14, lw_mean=1.6)
        pc.seed_family(ax, sel,
                       x_fn=_snap_x(lambda s: s.get("trig_loss_injected")),
                       y_fn=_snap_y(lambda s: s.get("trig_loss_injected")),
                       color="#9467bd", alpha_seed=0.14, lw_mean=1.7)
        ax.set_title(f"n = {n}" + ("  (never groks)" if n in (1, 2) else ""),
                     fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, CENSOR)
        ax.set_ylim(1e-4, 60)
    _hide_extra(axes, ns)
    fig.legend(handles=[
        Line2D([], [], color="0.6", lw=2, label="full test loss"),
        Line2D([], [], color="#9467bd", lw=2, label="trig loss | injected freqs (sufficiency)"),
        Line2D([], [], color="#2ca02c", lw=2, label="trig loss | model key freqs")],
        loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=True, fontsize=8)
    fig.supxlabel("epoch (log)")
    fig.supylabel("loss (log)")
    fig.suptitle("Exp06 — sufficiency: do the injected freqs alone reproduce "
                 "the model?", fontsize=12.5)
    pc.save(fig, outdir / "sufficiency.png", cap=(
        "Loss using ONLY the injected freqs' logit components (purple, "
        "sufficiency) vs only the model's key freqs (green) vs the full test "
        "loss (gray), per n. If purple tracks gray, the injected freqs alone "
        "explain the model. For n>=3 the injected-only loss collapses to near "
        "the full loss (sufficient); the gap that remains is what the model "
        "carries beyond the injected set. (Necessity is shown causally in the "
        "ablation figure — exp01's excluded-loss metric is NaN at p=211.)"))


# --------------------------------------------------------------------------- #
# 6. mechanistic summary across n (incl. the decisive kept-set ceiling test)
# --------------------------------------------------------------------------- #
def fig_summary(runs, outdir):
    ns = pc.axis_values(runs, "n")
    grok_ns = [n for n in ns if any(pc.groked(r) for r in pc.select(runs, n=n))]
    cmap = pc.color_map(ns, baseline=0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    aKS, aRET, aABL, aDEL = axes.flat

    def strip(ax, fn, color_fn, only_grok=True):
        for n in grok_ns:
            sel = [r for r in pc.select(runs, n=n) if (pc.groked(r) or not only_grok)]
            vals = _finite([fn(r) for r in sel])
            if not vals:
                continue
            ax.scatter(jitter(n, len(vals)), vals, color=color_fn(n), s=34,
                       alpha=0.85, zorder=3)
            ax.scatter([n], [np.mean(vals)], color=color_fn(n), marker="_",
                       s=520, linewidths=2.4, zorder=4)
        ax.set_xticks(grok_ns)

    # (a) kept-set ceiling — the decisive capacity test
    strip(aKS, lambda r: pc.n_key_freqs(pc.final_snap(r)), lambda n: cmap[n])
    xs = np.array(grok_ns)
    aKS.plot(xs, xs, color="0.7", ls="-.", lw=1.0, label="y = n (keep all offered)")
    aKS.axhline(P113_CEILING, color="crimson", ls="--", lw=1.6)
    aKS.text(grok_ns[-1], P113_CEILING + 0.25, f"p=113 ceiling (~{P113_CEILING})",
             ha="right", va="bottom", color="crimson", fontsize=9, fontweight="bold")
    aKS.set(xlabel="n injected pairs", ylabel="final n key freqs (working set)",
            title="(a) kept-set ceiling MOVES with model scale")
    aKS.legend(loc="upper left", fontsize=8, frameon=True)

    # (b) retention of injected freqs as key freqs
    strip(aRET, lambda r: pc.frac_injected_in_key(pc.final_snap(r)), lambda n: cmap[n])
    aRET.set(xlabel="n injected pairs", ylabel="frac injected retained as key",
             title="(b) retained fraction of injected freqs", ylim=(0, 1.08))

    # (c) final accuracy oracle ON vs OFF (independence)
    for n in grok_ns:
        sel = [r for r in pc.select(runs, n=n) if pc.groked(r)]
        on = _finite([pc.ablation(pc.final_snap(r), "acc_on") for r in sel])
        off = _finite([pc.ablation(pc.final_snap(r), "acc_off") for r in sel])
        if on:
            aABL.scatter(jitter(n, len(on)), on, color=ON_C, s=30, alpha=0.8, zorder=3)
            aABL.scatter([n], [np.mean(on)], color=ON_C, marker="_", s=460, lw=2.2, zorder=4)
        if off:
            aABL.scatter(jitter(n, len(off)), off, color=OFF_C, s=30, alpha=0.8, zorder=3)
            aABL.scatter([n], [np.mean(off)], color=OFF_C, marker="_", s=460, lw=2.2, zorder=4)
    aABL.set(xlabel="n injected pairs", ylabel="final test accuracy",
             title="(c) dependence: oracle ON vs OFF", ylim=(0, 1.08))
    aABL.set_xticks(grok_ns)
    aABL.legend(handles=[Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
                         Line2D([], [], color=OFF_C, lw=2, label="oracle OFF")],
                loc="lower left", fontsize=8, frameon=True)

    # (d) final ablation delta (CE_off - CE_on)
    strip(aDEL, lambda r: pc.ablation(pc.final_snap(r), "delta"), lambda n: cmap[n])
    aDEL.set(xlabel="n injected pairs", ylabel="CE_off - CE_on",
             title="(d) causal dependence delta (grows with n)")

    fig.suptitle("Exp06 — mechanistic summary across n (groked runs only)",
                 fontsize=13)
    pc.save(fig, outdir / "summary_vs_n.png", cap=(
        "Final-state metrics vs n, groked runs only (jittered seeds + mean). "
        "(a) DECISIVE: the working-set size grows past the p=113 ~5 ceiling to "
        "~9-10 by n=11 — the ceiling is capacity-bound, not architectural "
        "(parameter-cost hypothesis). It stays below y=n: a curated subset is "
        "kept. (b) retained fraction of injected freqs falls as more are "
        "offered. (c,d) dependence on the live oracle (ON-OFF gap and CE delta) "
        "GROWS with n — larger bases lean on the oracle more, opposite to p=113."))


# --------------------------------------------------------------------------- #
def main():
    res = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RES
    outdir = Path(res) / "figures" / EXP
    outdir.mkdir(parents=True, exist_ok=True)
    pc.set_style()
    runs = pc.load_exp(res, EXP)

    fig_grok_dynamics(runs, outdir)
    fig_uptake(runs, outdir)
    fig_spectrum(runs, outdir)
    fig_ablation(runs, outdir)
    fig_sufficiency(runs, outdir)
    fig_summary(runs, outdir)

    print(f"wrote {len(list(outdir.glob('*.png')))} figures to {outdir}")
    for p in sorted(outdir.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
