"""Curated figure suite for the oracle-injection experiments.

The clean, subplot-only suite first built for exp06, generalized so the SAME
code renders any experiment — p, n-range, chance line, grok censor cap and the
spectrum panel selection are all derived from the data. Each experiment gets a
thin wrapper (plot_exp01.py, plot_exp06.py) that just calls build().

Six figures (all subplot grids / a vs-n dashboard, no overlays):
  1 grok_dynamics — completeness threshold (test acc per n)
  2 uptake        — W_E adopts the injected freqs (power fraction per n)
  3 spectrum      — exact uptake at injected sites (final W_E spectrum per n)
  4 ablation      — causal use: accuracy with the live oracle ON vs OFF per n
  5 sufficiency   — do the injected freqs alone explain the model? (trig loss)
  6 summary_vs_n  — mechanistic dashboard across n (working set, retention,
                    ON/OFF dependence, CE delta)

Captions are descriptive and parameterised by the model (p / frac_train /
epochs read from data); experiment-specific interpretation lives in the
figures/README.md, not baked into the captions, so the same generator stays
honest across models.
"""
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_common as pc

ON_C, OFF_C = "#1f77b4", "#d62728"
REF_CEILING = 5            # working-set size seen at the small p=113 model


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
    return (lambda r: pc.snap_series(r, fn)[1])


def _snap_x(fn):
    return (lambda r: np.clip(pc.snap_series(r, fn)[0], 1, None))


def _finite(xs):
    return [v for v in xs if v is not None and v == v]


def _tag(sel):
    """Per-panel grok-rate suffix, '' when every seed groks."""
    gk = sum(pc.groked(r) for r in sel)
    return "" if gk == len(sel) else f"  ({gk}/{len(sel)} grok)"


def _derive(runs):
    ns = pc.axis_values(runs, "n")
    inj = [n for n in ns if n]
    L = len(pc.final_snap(runs[0])["we_freq_power_full"])     # p//2 Fourier bins
    p = 2 * L + 1
    censor = max(r["num_epochs"] for r in runs)
    ax = runs[0]["_axes"]
    cfg = (runs[0].get("spec", {}) or {}).get("config", {}) or {}
    frac = ax.get("frac_train", cfg.get("frac_train", 0.3))
    if len(inj) <= 6:
        spec = inj
    else:
        idx = np.linspace(0, len(inj) - 1, 6).round().astype(int)
        spec = sorted({inj[i] for i in idx})
    return dict(ns=ns, inj=inj, p=p, L=L, chance=1.0 / p, censor=censor,
                frac=frac, spec=spec,
                tag=f"p={p}, frac_train={frac:g}, {censor // 1000}k ep")


# --------------------------------------------------------------------------- #
def fig_grok_dynamics(runs, outdir, d, label):
    ns = d["ns"]
    cmap = pc.color_map(ns, baseline=0)
    fig, axes = _grid(ns)
    for ax, n in zip(axes.flat, ns):
        sel = pc.select(runs, n=n)
        pc.seed_family(ax, sel,
                       x_fn=lambda r: np.clip(pc.hist_series(r, "test_acc")[0], 1, None),
                       y_fn=lambda r: pc.hist_series(r, "test_acc")[1],
                       color=cmap[n], alpha_seed=0.25)
        ax.axhline(d["chance"], color="0.6", ls=":", lw=0.8)
        gk = _finite([pc.grok_epoch(r) for r in sel])
        tot = len(sel)
        txt = (f"{len(gk)}/{tot} grok\n~{int(np.median(gk))} ep" if gk
               else f"0/{tot}\nnever groks")
        col = "0.25" if len(gk) == tot else ("#b00020" if not gk else "#9a6a00")
        ax.text(0.95, 0.14, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=col)
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(1, d["censor"])
        ax.set_ylim(-0.03, 1.05)
    _hide_extra(axes, ns)
    fig.supxlabel("epoch (log)")
    fig.supylabel("test accuracy")
    fig.suptitle(f"{label} — grokking dynamics by n injected pairs ({d['tag']})",
                 fontsize=12.5)
    pc.save(fig, outdir / "grok_dynamics.png", cap=(
        f"Test accuracy vs epoch, one panel per n (4 seeds faint + seed-mean "
        f"bold; dotted = chance, 1/{d['p']}). Each panel is annotated with the "
        "grok rate and median grok epoch. The completeness threshold is the "
        "smallest n at which the model reliably groks."))


def fig_uptake(runs, outdir, d, label):
    cmap = pc.color_map(d["ns"], baseline=0)
    fig, axes = _grid(d["inj"])
    for ax, n in zip(axes.flat, d["inj"]):
        sel = pc.select(runs, n=n)
        pc.seed_family(ax, sel,
                       x_fn=_snap_x(pc.frac_we_power_injected),
                       y_fn=_snap_y(pc.frac_we_power_injected),
                       color=cmap[n], alpha_seed=0.25)
        ax.set_title(f"n = {n}{_tag(sel)}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(1, d["censor"])
        ax.set_ylim(-0.03, 1.05)
    _hide_extra(axes, d["inj"])
    fig.supxlabel("epoch (log)")
    fig.supylabel("fraction of W_E Fourier power on injected freqs")
    fig.suptitle(f"{label} — the embedding adopts the injected frequencies",
                 fontsize=12.5)
    pc.save(fig, outdir / "uptake.png", cap=(
        "Fraction of W_E Fourier power on the injected freqs vs epoch, per n "
        "(4 seeds + mean). Grokking runs concentrate most of the embedding's "
        "Fourier power on the oracle's frequencies — direct uptake into the "
        "trainable embedding."))


def fig_spectrum(runs, outdir, d, label):
    cmap = pc.color_map(d["ns"], baseline=0)
    fig, axes = _grid(d["spec"], ncol=3, panel=(4.6, 2.7), sharey=False)
    for ax, n in zip(axes.flat, d["spec"]):
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
        ax.set_title(f"n = {n}{_tag(sel)}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7.5, frameon=True)
    _hide_extra(axes, d["spec"])
    fig.supxlabel(f"Fourier frequency index (1..{d['L']})")
    fig.supylabel("final W_E power")
    fig.suptitle(f"{label} — power piles up on the injected sites "
                 f"(final W_E spectrum, p={d['p']})", fontsize=12.5)
    pc.save(fig, outdir / "spectrum.png", cap=(
        f"Final W_E Fourier power spectrum ({d['L']} freqs) per n; 4 seeds "
        "faint + mean bold, green dashed = injected frequencies. For grokking "
        "n the peaks land on the injected sites with little leakage."))


def fig_ablation(runs, outdir, d, label):
    fig, axes = _grid(d["inj"])
    for ax, n in zip(axes.flat, d["inj"]):
        sel = pc.select(runs, n=n)
        pc.seed_family(ax, sel,
                       x_fn=_snap_x(lambda s: (s.get("ablation_test") or {}).get("acc_on")),
                       y_fn=_snap_y(lambda s: (s.get("ablation_test") or {}).get("acc_on")),
                       color=ON_C, alpha_seed=0.15, lw_mean=1.7)
        pc.seed_family(ax, sel,
                       x_fn=_snap_x(lambda s: (s.get("ablation_test") or {}).get("acc_off")),
                       y_fn=_snap_y(lambda s: (s.get("ablation_test") or {}).get("acc_off")),
                       color=OFF_C, alpha_seed=0.15, lw_mean=1.7)
        ax.set_title(f"n = {n}{_tag(sel)}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(min(1800, d["censor"] // 10), d["censor"])
        ax.set_ylim(-0.03, 1.05)
    _hide_extra(axes, d["inj"])
    fig.legend(handles=[Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
                        Line2D([], [], color=OFF_C, lw=2, label="oracle OFF")],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=True)
    fig.supxlabel("epoch (log)")
    fig.supylabel("test accuracy")
    fig.suptitle(f"{label} — causal use: accuracy with the live oracle ON vs OFF",
                 fontsize=12.5)
    pc.save(fig, outdir / "ablation.png", cap=(
        "Accuracy with the live oracle ON (blue) vs switched OFF at inference "
        "(red), per n (4 seeds + mean). The ON-OFF gap measures how much the "
        "model depends on the live signal versus having internalised it into "
        "weights (small gap = independent / internalised)."))


def fig_sufficiency(runs, outdir, d, label):
    fig, axes = _grid(d["inj"])
    for ax, n in zip(axes.flat, d["inj"]):
        sel = pc.select(runs, n=n)
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
        ax.set_title(f"n = {n}{_tag(sel)}", fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, d["censor"])
        ax.set_ylim(1e-4, 60)
    _hide_extra(axes, d["inj"])
    fig.legend(handles=[
        Line2D([], [], color="0.6", lw=2, label="full test loss"),
        Line2D([], [], color="#9467bd", lw=2, label="trig loss | injected freqs (sufficiency)"),
        Line2D([], [], color="#2ca02c", lw=2, label="trig loss | model key freqs")],
        loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=True, fontsize=8)
    fig.supxlabel("epoch (log)")
    fig.supylabel("loss (log)")
    fig.suptitle(f"{label} — sufficiency: do the injected freqs alone reproduce "
                 "the model?", fontsize=12.5)
    pc.save(fig, outdir / "sufficiency.png", cap=(
        "Loss using ONLY the injected freqs' logit components (purple, "
        "sufficiency) vs only the model's key freqs (green) vs the full test "
        "loss (gray), per n. Purple tracking gray ⇒ the injected freqs alone "
        "reproduce the model; the residual gap is what the model carries "
        "beyond the injected set."))


def fig_summary(runs, outdir, d, label):
    ns = d["ns"]
    grok_ns = [n for n in ns if any(pc.groked(r) for r in pc.select(runs, n=n))]
    cmap = pc.color_map(ns, baseline=0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    aKS, aRET, aABL, aDEL = axes.flat

    def strip(ax, fn):
        for n in grok_ns:
            sel = [r for r in pc.select(runs, n=n) if pc.groked(r)]
            vals = _finite([fn(r) for r in sel])
            if not vals:
                continue
            ax.scatter(jitter(n, len(vals)), vals, color=cmap[n], s=34,
                       alpha=0.85, zorder=3)
            ax.scatter([n], [np.mean(vals)], color=cmap[n], marker="_",
                       s=520, linewidths=2.4, zorder=4)
        ax.set_xticks(grok_ns)

    strip(aKS, lambda r: pc.n_key_freqs(pc.final_snap(r)))
    xs = np.array(grok_ns)
    aKS.plot(xs, xs, color="0.7", ls="-.", lw=1.0, label="y = n (keep all offered)")
    aKS.axhline(REF_CEILING, color="crimson", ls="--", lw=1.4)
    aKS.text(grok_ns[-1], REF_CEILING + 0.25, f"p=113 working-set (~{REF_CEILING})",
             ha="right", va="bottom", color="crimson", fontsize=9, fontweight="bold")
    aKS.set(xlabel="n injected pairs", ylabel="final n key freqs (working set)",
            title="(a) working set vs n")
    aKS.legend(loc="upper left", fontsize=8, frameon=True)

    strip(aRET, lambda r: pc.frac_injected_in_key(pc.final_snap(r)))
    aRET.set(xlabel="n injected pairs", ylabel="frac injected retained as key",
             title="(b) retained fraction of injected freqs", ylim=(0, 1.08))

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

    strip(aDEL, lambda r: pc.ablation(pc.final_snap(r), "delta"))
    aDEL.set(xlabel="n injected pairs", ylabel="CE_off - CE_on",
             title="(d) causal dependence delta")

    fig.suptitle(f"{label} — mechanistic summary across n (groked runs only)",
                 fontsize=13)
    pc.save(fig, outdir / "summary_vs_n.png", cap=(
        "Final-state metrics vs n, groked runs only (jittered seeds + mean). "
        f"(a) working-set size with y=n and a ~{REF_CEILING}-freq reference "
        "(the p=113 working-set ceiling); (b) fraction of injected freqs kept "
        "as key freqs; (c) final accuracy oracle ON vs OFF (gap = dependence); "
        "(d) causal dependence delta, CE_off - CE_on."))


# --------------------------------------------------------------------------- #
def build(res, exp, label):
    outdir = Path(res) / "figures" / exp
    outdir.mkdir(parents=True, exist_ok=True)
    pc.set_style()
    runs = pc.load_exp(res, exp)
    d = _derive(runs)
    fig_grok_dynamics(runs, outdir, d, label)
    fig_uptake(runs, outdir, d, label)
    fig_spectrum(runs, outdir, d, label)
    fig_ablation(runs, outdir, d, label)
    fig_sufficiency(runs, outdir, d, label)
    fig_summary(runs, outdir, d, label)
    print(f"wrote {len(list(outdir.glob('*.png')))} figures to {outdir}")
    return d
