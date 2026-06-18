# %% [markdown]
# # Exp 02.2 — Amplitude sweep figure suite (W_E laziness vs. oracle loudness)
#
# Hypotheses under test:
#   (a) LAZINESS — as oracle AMPLITUDE rises, the trainable W_E offloads work
#       onto the live oracle signal: lower embedding norm, lower own-power at
#       the injected freqs, and *more dependence* on the oracle being on.
#   (b) DESTABILISATION — too-loud an oracle can stop the model grokking,
#       clustered at high amp & low n (few injected freqs).
#
# CONFOUND GUARD: raw ablation ΔCE (= ce_off − ce_on) scales ~linearly with
# amplitude, so it is NOT a clean dependence metric. We use ablation `acc_off`
# (bounded 0..1, amplitude-robust) as the dependence metric, and show raw ΔCE
# only in an explicit "caveat" figure (fig 5).
#
# GUARD: every aggregated line/marker figure is ONLY-GROKED — never-grokked
# seeds are excluded from means and from scatter. The two 0%-grok cells
# (amp=2,n=1 and amp=2,n=2) therefore vanish from the line figures entirely;
# the grok heatmap (fig 1) is the one place failures are shown.
#
# Usage:  .venv/bin/python modular_addition/oracle/experiments/plot_exp02_2.py [RESULTS_DIR]

# %% imports + path bootstrap
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import plot_common as pc

EXP = "exp02_2"
DEFAULT_RES = (_HERE.parents[0] / "results" / "run_20260612_200000")
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RES
OUT = RES / "figures" / EXP


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _final_scalar(r, fn):
    """fn(final_snapshot) -> float|None."""
    s = pc.final_snap(r)
    return fn(s) if s else None


def _final_test_acc(r):
    h = r.get("history")
    return float(h[-1]["test_acc"]) if h else None


def _jitter(x, k, span=0.10):
    """Deterministic symmetric jitter for k points centred on x (log-x safe:
    multiplicative)."""
    if k <= 1:
        return [x]
    offs = np.linspace(-span, span, k)
    return [x * (1.0 + o) for o in offs]


def _seed_scatter_vs_amp(ax, runs, y_fn, amps, ns, cmap, only_groked=True):
    """For each (amp,n): jittered per-seed markers + a bold mean marker.
    Returns dict (amp,n)->n_seeds_drawn for caption bookkeeping."""
    counts = {}
    for n in ns:
        xs_mean, ys_mean = [], []
        col = cmap[n]
        for amp in amps:
            cell = pc.select(runs, amp=amp, n=n)
            if only_groked:
                cell = [r for r in cell if pc.groked(r)]
            ys = [y_fn(r) for r in cell]
            ys = [v for v in ys if v is not None and np.isfinite(v)]
            counts[(amp, n)] = len(ys)
            if not ys:
                continue
            for xj, yv in zip(_jitter(amp, len(ys)), ys):
                ax.scatter(xj, yv, color=col, s=22, alpha=0.45,
                           edgecolors="none", zorder=2)
            m = float(np.mean(ys))
            xs_mean.append(amp)
            ys_mean.append(m)
            ax.scatter(amp, m, color=col, s=70, marker="D",
                       edgecolors="black", linewidths=0.6, zorder=4)
        if xs_mean:
            ax.plot(xs_mean, ys_mean, color=col, lw=1.8, alpha=0.95,
                    zorder=3, label=f"n={n}")
    return counts


def _amp_axis(ax, amps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(amps)
    ax.set_xticklabels([f"{a:g}" for a in amps])
    ax.set_xlim(amps[0] * 0.8, amps[-1] * 1.25)
    ax.set_xlabel("oracle amplitude")


# --------------------------------------------------------------------------- #
# amp(rows) x n(cols) panel grids: standard W_E spectrum + ablation, matching
# the exp04 reliability grids (same idiom: sweep1 down rows, n across cols).
# --------------------------------------------------------------------------- #
ON_C, OFF_C = "#1f77b4", "#d62728"


def _amp_n_grid(amps, ns, panel=(2.55, 1.95), sharey=False):
    fig, axes = plt.subplots(len(amps), len(ns),
                             figsize=(panel[0] * len(ns), panel[1] * len(amps)),
                             sharex=True, sharey=sharey, squeeze=False)
    return fig, axes


def _grok_badge(ax, cell):
    gk = sum(pc.groked(r) for r in cell)
    col = "0.3" if gk == len(cell) else ("#b00020" if gk == 0 else "#9a6a00")
    ax.text(0.96, 0.92, f"{gk}/{len(cell)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.2, color=col)


def fig_we_spectrum(runs, amps, ns, cmap, out):
    """Final W_E spectrum, amp(rows) x n(cols), injected freqs marked green.
    Raw power, per-panel y-scale so the comb stays visible across magnitudes."""
    fig, axes = _amp_n_grid(amps, ns, sharey=False)
    for i, amp in enumerate(amps):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            cell = pc.select(runs, amp=amp, n=n)
            specs = [np.asarray(pc.final_snap(r)["we_freq_power_full"], float)
                     for r in cell if pc.final_snap(r).get("we_freq_power_full")]
            if specs:
                freqs = np.arange(1, len(specs[0]) + 1)
                for f in (pc.final_snap(cell[0]).get("injected_freqs") or []):
                    ax.axvline(f, color="green", ls="--", lw=0.7, alpha=0.5)
                for sp in specs:
                    ax.plot(freqs, sp, color=cmap[n], alpha=0.22, lw=0.7)
                ax.plot(freqs, np.mean(specs, 0), color=cmap[n], lw=1.4)
            _grok_badge(ax, cell)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"amp = {amp:g}", fontsize=9.5)
            ax.tick_params(labelsize=6.5)
    fig.legend(handles=[Line2D([], [], color="green", ls="--", lw=1.2,
                               label="injected freqs")],
               loc="upper right", bbox_to_anchor=(0.998, 0.998), fontsize=8)
    fig.supxlabel("Fourier frequency index (1..56)")
    fig.suptitle("Exp02_2 — final W_E Fourier spectrum  (rows: oracle "
                 "amplitude, cols: n injected pairs)", fontsize=12.5)
    pc.save(fig, out / "we_spectrum.png", cap=(
        "Final W_E power spectrum per (amp, n) cell (4 seeds faint + mean; green "
        "dashed = injected freqs; corner badge = grok count). The comb lands on "
        "the injected freqs wherever the model groks; as amplitude rises the "
        "embedding offloads onto the louder oracle (lazier, lower own-power), "
        "and the amp=2, n in {1,2} cells (0/4 grok) show diffuse power, no comb."))


def _has_wl(runs):
    return any(pc.final_snap(r).get("wl_freq_power_full") for r in runs)


def fig_wl_spectrum(runs, amps, ns, cmap, out):
    """Final neuron-logit-map spectrum W_L = W_out^T W_U (readout side), amp(rows)
    x n(cols), injected freqs green. The output-weight mirror of we_spectrum:
    Nanda et al. 2023's 'neuron-logit map', whose DFT on the answer-token axis is
    sparse on the key freqs. Requires the wl_* snapshot fields (re-run needed)."""
    if not _has_wl(runs):
        print("  [fig_wl_spectrum] no wl_freq_power_full in data -- skipping "
              "(re-run exp02_2 with the updated analysis.py to populate it).")
        return
    fig, axes = _amp_n_grid(amps, ns, sharey=False)
    for i, amp in enumerate(amps):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            cell = pc.select(runs, amp=amp, n=n)
            specs = [np.asarray(pc.final_snap(r)["wl_freq_power_full"], float)
                     for r in cell if pc.final_snap(r).get("wl_freq_power_full")]
            if specs:
                freqs = np.arange(1, len(specs[0]) + 1)
                for f in (pc.final_snap(cell[0]).get("injected_freqs") or []):
                    ax.axvline(f, color="green", ls="--", lw=0.7, alpha=0.5)
                for sp in specs:
                    ax.plot(freqs, sp, color=cmap[n], alpha=0.22, lw=0.7)
                ax.plot(freqs, np.mean(specs, 0), color=cmap[n], lw=1.4)
            _grok_badge(ax, cell)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"amp = {amp:g}", fontsize=9.5)
            ax.tick_params(labelsize=6.5)
    fig.legend(handles=[Line2D([], [], color="green", ls="--", lw=1.2,
                               label="injected freqs")],
               loc="upper right", bbox_to_anchor=(0.998, 0.998), fontsize=8)
    fig.supxlabel("Fourier frequency index (1..56)")
    fig.suptitle("Exp02_2 — final neuron-logit-map spectrum W_L=W_out^T W_U  "
                 "(readout side; rows: amplitude, cols: n)", fontsize=12.5)
    pc.save(fig, out / "wl_spectrum.png", cap=(
        "Final neuron-logit-map W_L = W_out^T W_U Fourier power per (amp, n) cell, "
        "decomposed over the OUTPUT (answer) tokens -- the readout-side mirror of "
        "we_spectrum (Nanda et al. 2023). Green dashed = injected freqs; badge = "
        "grok count. Unlike W_E, the readout stays sharply tuned to (most of) the "
        "injected freqs even at high amplitude: the model's output machinery uses "
        "them; only the input embedding offloads."))


def fig_ablation(runs, amps, ns, out):
    """Accuracy with the live oracle ON vs forced OFF over training, amp(rows)
    x n(cols). The ON-OFF gap is the behavioural dependence."""
    def snap(key):
        return (lambda r: np.clip(pc.snap_series(
                    r, lambda s: (s.get("ablation_test") or {}).get(key))[0], 1, None),
                lambda r: pc.snap_series(
                    r, lambda s: (s.get("ablation_test") or {}).get(key))[1])
    on_x, on_y = snap("acc_on")
    off_x, off_y = snap("acc_off")
    fig, axes = _amp_n_grid(amps, ns, sharey=True)
    for i, amp in enumerate(amps):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            cell = pc.select(runs, amp=amp, n=n)
            pc.seed_family(ax, cell, x_fn=on_x, y_fn=on_y,
                           color=ON_C, alpha_seed=0.13, lw_mean=1.5)
            pc.seed_family(ax, cell, x_fn=off_x, y_fn=off_y,
                           color=OFF_C, alpha_seed=0.13, lw_mean=1.5)
            ax.set_xscale("log")
            ax.set_xlim(180, 30000)
            ax.set_ylim(-0.03, 1.05)
            _grok_badge(ax, cell)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"amp = {amp:g}", fontsize=9.5)
            ax.tick_params(labelsize=6.5)
    fig.legend(handles=[Line2D([], [], color=ON_C, lw=2, label="oracle live (ON)"),
                        Line2D([], [], color=OFF_C, lw=2, label="oracle forced OFF")],
               loc="upper right", bbox_to_anchor=(0.998, 0.998), fontsize=8)
    fig.supxlabel("epoch (log)")
    fig.supylabel("test accuracy")
    fig.suptitle("Exp02_2 — causal use of the live oracle, ON vs OFF  (rows: "
                 "oracle amplitude, cols: n)", fontsize=12.5)
    pc.save(fig, out / "ablation.png", cap=(
        "Accuracy with the live oracle ON (blue) vs forced OFF at inference "
        "(red), per (amp, n) cell (4 seeds + mean; badge = grok count). The "
        "ON-OFF gap is the behavioural dependence: it WIDENS down the rows (a "
        "louder oracle => the model leans on the live signal more, acc_off "
        "falls), strongest at small n. The amplitude-robust scalar version is "
        "dependence_accoff_vs_amp (final snapshot)."))


def fig_ce_ablation(runs, amps, ns, out):
    """Cross-entropy with the live oracle ON vs forced OFF over training,
    amp(rows) x n(cols). CE companion to the accuracy ablation. The OFF curve
    inflates mechanically with amplitude (the amp_scaling confound)."""
    def snap(key):
        return (lambda r: np.clip(pc.snap_series(
                    r, lambda s: (s.get("ablation_test") or {}).get(key))[0], 1, None),
                lambda r: pc.snap_series(
                    r, lambda s: (s.get("ablation_test") or {}).get(key))[1])
    on_x, on_y = snap("ce_on")
    off_x, off_y = snap("ce_off")
    chance = float(np.log(2 * len(pc.final_snap(runs[0])["we_freq_power_full"]) + 1))
    fig, axes = _amp_n_grid(amps, ns, sharey=True)
    for i, amp in enumerate(amps):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            cell = pc.select(runs, amp=amp, n=n)
            ax.axhline(chance, color="0.6", ls=":", lw=0.8)
            pc.seed_family(ax, cell, x_fn=on_x, y_fn=on_y,
                           color=ON_C, alpha_seed=0.13, lw_mean=1.5)
            pc.seed_family(ax, cell, x_fn=off_x, y_fn=off_y,
                           color=OFF_C, alpha_seed=0.13, lw_mean=1.5)
            ax.set_xscale("log")
            ax.set_xlim(180, 30000)
            ax.set_ylim(-0.3, 17)
            _grok_badge(ax, cell)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"amp = {amp:g}", fontsize=9.5)
            ax.tick_params(labelsize=6.5)
    fig.legend(handles=[
        Line2D([], [], color=ON_C, lw=2, label="oracle live (ON)"),
        Line2D([], [], color=OFF_C, lw=2, label="oracle forced OFF"),
        Line2D([], [], color="0.6", ls=":", lw=1.2, label="chance CE = ln p")],
        loc="upper right", bbox_to_anchor=(0.998, 0.998), fontsize=8)
    fig.supxlabel("epoch (log)")
    fig.supylabel("test cross-entropy")
    fig.suptitle("Exp02_2 — test cross-entropy with the live oracle ON vs OFF  "
                 "(rows: oracle amplitude, cols: n)", fontsize=12.5)
    pc.save(fig, out / "ce_ablation.png", cap=(
        "Test cross-entropy with the live oracle ON (blue) vs forced OFF at "
        "inference (red), per (amp, n) cell (4 seeds + mean; badge = grok count; "
        "dotted = chance, ln p). CAVEAT: the ON-OFF CE gap grows down the rows "
        "partly as a MEASUREMENT artefact - removing a louder oracle mechanically "
        "inflates CE (ce_off ~4.8 at amp=0.5 -> ~16 at amp=4) regardless of true "
        "reliance. For an amplitude-robust read use the accuracy version "
        "(ablation.png) / dependence_accoff_vs_amp."))


IN_C, OUT_C, WL_C = "#1f77b4", "#ff7f0e", "#2ca02c"   # W_E / logits / W_L


def _neff_injected(power, inj):
    """Effective # of injected freqs in use = 1 / Σ share²  over the injected
    set (Hill number / inverse participation ratio). 1 => all weight on one
    freq; n => spread evenly across all n injected freqs."""
    pw = np.asarray(power, float)[[f - 1 for f in inj]]
    s = pw.sum()
    if s <= 0:
        return np.nan
    sh = pw / s
    return float(1.0 / np.sum(sh ** 2))


def _neff_we(snap):
    inj = snap.get("injected_freqs") or []
    return _neff_injected(snap["we_freq_power_full"], inj) if inj else np.nan


def _neff_logit(snap):
    inj = snap.get("injected_freqs") or []
    if not inj:
        return np.nan
    return _neff_injected(np.asarray(snap["logit_coeff_full"], float) ** 2, inj)


def _neff_wl(snap):
    inj = snap.get("injected_freqs") or []
    fp = snap.get("wl_freq_power_full")
    return _neff_injected(fp, inj) if (inj and fp) else np.nan


def _inj_shares(cell, side):
    """Mean (over grokked seeds) share-of-injected-power per injected freq.
    side: 'we' (W_E power) or 'lg' (logit coeff²). Returns {freq: mean_share}."""
    acc = {}
    for r in cell:
        sn = pc.final_snap(r)
        inj = sn.get("injected_freqs") or []
        pw = (np.asarray(sn["we_freq_power_full"], float) if side == "we"
              else np.asarray(sn["logit_coeff_full"], float) ** 2)
        sub = pw[[f - 1 for f in inj]]
        tot = sub.sum()
        for f, v in zip(inj, sub):
            acc.setdefault(f, []).append(v / tot if tot > 0 else 0.0)
    return {f: float(np.mean(v)) for f, v in acc.items()}


def fig_input_vs_output_freqs(runs, amps, ns, out):
    """Effective # of injected freqs used by the INPUT (W_E) vs the OUTPUT
    (logits) vs amplitude. Output > input => the model uses freqs W_E never
    encoded => it reads them straight from the live oracle (W_E is lazy)."""
    use_ns = [n for n in ns if n >= 3]          # n=1,2 are degenerate (N_eff<=2)
    ncol = 2
    nrow = (len(use_ns) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.7 * nrow),
                             squeeze=False)
    xs = list(range(len(amps)))
    has_wl = _has_wl(runs)
    for ax, n in zip(axes.flat, use_ns):
        we_m, lg_m, wl_m = [], [], []
        for xi, amp in zip(xs, amps):
            cell = [r for r in pc.select(runs, amp=amp, n=n) if pc.groked(r)]
            we = [v for v in (_neff_we(pc.final_snap(r)) for r in cell) if v == v]
            lg = [v for v in (_neff_logit(pc.final_snap(r)) for r in cell) if v == v]
            wl = [v for v in (_neff_wl(pc.final_snap(r)) for r in cell) if v == v]
            for v in we:
                ax.scatter(xi - 0.08, v, color=IN_C, s=16, alpha=0.4, zorder=2)
            for v in lg:
                ax.scatter(xi + 0.08, v, color=OUT_C, s=16, alpha=0.4, zorder=2)
            for v in wl:
                ax.scatter(xi, v, color=WL_C, s=16, alpha=0.4, zorder=2)
            we_m.append(np.mean(we) if we else np.nan)
            lg_m.append(np.mean(lg) if lg else np.nan)
            wl_m.append(np.mean(wl) if wl else np.nan)
        ax.plot(xs, we_m, color=IN_C, lw=2.0, marker="o", ms=6, zorder=3)
        ax.plot(xs, lg_m, color=OUT_C, lw=2.0, marker="s", ms=6, zorder=3)
        if has_wl:
            ax.plot(xs, wl_m, color=WL_C, lw=2.0, marker="^", ms=6, zorder=3)
        ax.axhline(n, color="0.6", ls=":", lw=1.0)
        ax.text(xs[-1], n, f" n={n} offered", va="bottom", ha="right",
                color="0.5", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{a:g}" for a in amps])
        ax.set_title(f"n = {n} injected pairs", fontsize=10)
        ax.set_ylim(0.8, n + 0.5)
    for ax in axes.flat[len(use_ns):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=IN_C, lw=2, marker="o",
                      label="W_E  (input / embedding weights)"),
               Line2D([], [], color=OUT_C, lw=2, marker="s",
                      label="logits  (realized output)")]
    if has_wl:
        handles.append(Line2D([], [], color=WL_C, lw=2, marker="^",
                              label="W_L  (readout weights, W_out·W_U)"))
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.998, 0.998), fontsize=9)
    fig.supxlabel("oracle amplitude")
    fig.supylabel("effective # of injected freqs used  (1 / Σ share²)")
    fig.suptitle("Exp02_2 — does the WHOLE MODEL use the freqs W_E drops?  "
                 "input (W_E) vs readout (W_L) vs output (logits)", fontsize=12.5)
    pc.save(fig, out / "input_vs_output_freqs.png", cap=(
        "Effective number of injected frequencies in use = participation ratio "
        "1/Σ(share²) over the injected set, grokked seeds only, for: the INPUT "
        "embedding W_E (blue), the readout WEIGHTS W_L=W_out·W_U (green), and the "
        "realized OUTPUT logits (orange = coeff on cos(w(x+y-z)), oracle live). As "
        "amplitude rises only the W_E curve FALLS (the embedding concentrates on "
        "fewer freqs = lazy) while the readout W_L and the logits stay high / rise "
        "and overtake it: the model's output machinery keeps using MORE distinct "
        "injected freqs than W_E encodes, reading the surplus straight from the "
        "live oracle. => the laziness is a property of W_E, not the whole model."))


def fig_freq_usage_detail(runs, amps, out, n=8):
    """Per-injected-frequency share of W_E power vs logit power, one panel per
    amplitude (the per-freq evidence behind input_vs_output_freqs)."""
    cells = {amp: [r for r in pc.select(runs, amp=amp, n=n) if pc.groked(r)]
             for amp in amps}
    base = cells[amps[0]]
    we_base = _inj_shares(base, "we")
    order = sorted(we_base, key=lambda f: -we_base[f])     # fixed across panels
    x = np.arange(len(order))
    w = 0.4
    fig, axes = plt.subplots(1, len(amps), figsize=(3.6 * len(amps), 4.1),
                             sharey=True, squeeze=False)
    for ax, amp in zip(axes[0], amps):
        we_s = _inj_shares(cells[amp], "we")
        lg_s = _inj_shares(cells[amp], "lg")
        ax.bar(x - w / 2, [we_s.get(f, 0) for f in order], w, color=IN_C,
               label="W_E (input)")
        ax.bar(x + w / 2, [lg_s.get(f, 0) for f in order], w, color=OUT_C,
               label="logits (output)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"f{f}" for f in order], rotation=45, fontsize=7)
        ax.set_title(f"amp = {amp:g}", fontsize=10)
    axes[0][0].set_ylabel("share of injected-set power")
    axes[0][0].legend(fontsize=8.5, frameon=True)
    fig.supxlabel(f"injected frequencies (n={n}), ordered by W_E share at "
                  f"amp={amps[0]:g}")
    fig.suptitle(f"Exp02_2 — per-frequency input vs output share "
                 f"(n={n}, grokked-seed mean)", fontsize=12)
    pc.save(fig, out / f"freq_usage_n{n}_detail.png", cap=(
        "Share of the injected-set power on each injected frequency, in W_E "
        "(blue, input) vs the logits (orange, output), grokked-seed mean. At low "
        "amp the two roughly match (the model builds in W_E what it outputs). As "
        "amp rises W_E's share concentrates onto 2-3 freqs while the output keeps "
        "spreading across ~6: frequencies where orange >> blue are used by the "
        "model yet barely encoded by W_E -- supplied by the live oracle."))


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    pc.set_style()
    runs = pc.load_exp(RES, EXP)
    amps = pc.axis_values(runs, "amp")
    ns = pc.axis_values(runs, "n")
    cmap = pc.color_map(ns, baseline=None)
    print(f"loaded {len(runs)} runs | amps={amps} | ns={ns}")

    # grok bookkeeping --------------------------------------------------------
    n_total = len(runs)
    n_grok = sum(pc.groked(r) for r in runs)
    print(f"grokked {n_grok}/{n_total} ({n_total - n_grok} failures)")

    # ----------------------------------------------------------------------- #
    # FIG 1a — grok-success heatmap (amp rows x n cols)
    # ----------------------------------------------------------------------- #
    grok = np.full((len(amps), len(ns)), np.nan)
    nseed = np.zeros((len(amps), len(ns)), dtype=int)
    for i, amp in enumerate(amps):
        for j, n in enumerate(ns):
            cell = pc.select(runs, amp=amp, n=n)
            nseed[i, j] = len(cell)
            grok[i, j] = pc.grok_rate(cell)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(grok, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_yticks(range(len(amps)))
    ax.set_yticklabels([f"{a:g}" for a in amps])
    ax.set_xlabel("n  (# injected freq pairs)")
    ax.set_ylabel("oracle amplitude")
    ax.set_title("Grok success rate  (fraction of 4 seeds that grokked)")
    ax.grid(False)
    for i in range(len(amps)):
        for j in range(len(ns)):
            v = grok[i, j]
            k = int(round(v * nseed[i, j]))
            txt = f"{v:.2f}\n{k}/{nseed[i, j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="black" if 0.25 < v < 0.85 else "white",
                    fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("grok rate")
    pc.save(fig, OUT / "grok_success_heatmap.png", cap=(
        "Fig 1a. Grok-success rate per (amp, n) cell; annotation = rate and "
        "grokked/total seeds. FAILURE CLUSTER at low n (1-2): amp=2 fails "
        "ENTIRELY at n=1 and n=2 (0/4 both), and amp in {1,4} grok partially "
        "there. n>=3 always groks. This destabilisation is the headline AND the "
        "confound guard: those cells are excluded from all line figures below."))

    # ----------------------------------------------------------------------- #
    # FIG 1b — median grok_epoch over grokked seeds per cell
    # ----------------------------------------------------------------------- #
    gep = np.full((len(amps), len(ns)), np.nan)
    for i, amp in enumerate(amps):
        for j, n in enumerate(ns):
            cell = [pc.grok_epoch(r) for r in pc.select(runs, amp=amp, n=n)
                    if pc.groked(r)]
            if cell:
                gep[i, j] = float(np.median(cell))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(gep, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_yticks(range(len(amps)))
    ax.set_yticklabels([f"{a:g}" for a in amps])
    ax.set_xlabel("n  (# injected freq pairs)")
    ax.set_ylabel("oracle amplitude")
    ax.set_title("Median grok epoch  (over grokked seeds)")
    ax.grid(False)
    finite = gep[np.isfinite(gep)]
    mid = (finite.min() + finite.max()) / 2 if finite.size else 0
    for i in range(len(amps)):
        for j in range(len(ns)):
            v = gep[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if v > mid else "black", fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11,
                        color="0.7")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("median grok epoch")
    pc.save(fig, OUT / "grok_epoch_heatmap.png", cap=(
        "Fig 1b. Median grok epoch over the grokked seeds in each cell "
        "(em-dash = no seed grokked, i.e. amp=2 at n=1,2). Companion to Fig 1a: "
        "where the model does grok, low n tends to grok later; amplitude has a "
        "weaker, less monotonic effect on timing than on success."))

    # ----------------------------------------------------------------------- #
    # FIG 2 — laziness: final |W_E| vs amp (only-grokked)
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    c2 = _seed_scatter_vs_amp(
        ax, runs, lambda r: _final_scalar(r, lambda s: s.get("we_total_norm")),
        amps, ns, cmap)
    _amp_axis(ax, amps)
    ax.set_ylabel("final |W_E|  (we_total_norm)")
    ax.set_title("Laziness — embedding norm vs oracle amplitude (grokked only)")
    ax.legend(title="freq pairs", ncol=2, fontsize=8, framealpha=0.9)
    pc.save(fig, OUT / "laziness_we_norm_vs_amp.png", cap=(
        "Fig 2. Final trainable embedding norm |W_E| vs amplitude; diamonds=seed "
        "mean, dots=individual grokked seeds (jittered), colour=n. Only grokked "
        "seeds shown (amp=2,n in{1,2} therefore absent). If laziness holds, |W_E| "
        "should DROP as amp rises - the embedding offloads work onto the louder "
        "oracle."))

    # ----------------------------------------------------------------------- #
    # FIG 3 — laziness: final frac W_E power on injected freqs vs amp
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _seed_scatter_vs_amp(
        ax, runs, lambda r: _final_scalar(r, pc.frac_we_power_injected),
        amps, ns, cmap)
    _amp_axis(ax, amps)
    ax.set_ylabel("final frac W_E power on injected freqs")
    ax.set_title("Laziness — W_E uptake of injected freqs vs amplitude (grokked only)")
    ax.legend(title="freq pairs", ncol=2, fontsize=8, framealpha=0.9)
    pc.save(fig, OUT / "laziness_uptake_vs_amp.png", cap=(
        "Fig 3. Fraction of W_E Fourier power sitting ON the injected freqs vs "
        "amplitude (grokked only, colour=n). Interpretation is subtle: a DROP "
        "means W_E stops representing the injected freqs itself (leans on the "
        "oracle = lazy); a RISE could mean W_E is being entrained toward them. "
        "Read alongside Fig 4 (acc_off), the behavioural dependence metric."))

    # ----------------------------------------------------------------------- #
    # FIG 4 — dependence: final ablation acc_off vs amp (THE metric)
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _seed_scatter_vs_amp(
        ax, runs, lambda r: _final_scalar(r, lambda s: pc.ablation(s, "acc_off")),
        amps, ns, cmap)
    _amp_axis(ax, amps)
    ax.axhline(1.0, color="0.6", ls=":", lw=1.0)
    ax.set_ylabel("final ablation acc_off  (accuracy w/ oracle OFF)")
    ax.set_title("Dependence — accuracy with the oracle ablated (grokked only)")
    ax.legend(title="freq pairs", ncol=2, fontsize=8, framealpha=0.9)
    pc.save(fig, OUT / "dependence_accoff_vs_amp.png", cap=(
        "Fig 4. THE dependence metric: test accuracy with the oracle turned OFF "
        "(bounded 0..1, amplitude-robust), final snapshot, grokked seeds only, "
        "colour=n. LOWER acc_off = the model leans more on the live oracle signal. "
        "A downward trend with amp is the clean evidence for laziness/dependence; "
        "1.0 (dotted) = fully self-sufficient. Use this, NOT raw ΔCE (Fig 5)."))

    # ----------------------------------------------------------------------- #
    # FIG 5 — caveat: raw ablation delta vs amp (CONFOUND)
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _seed_scatter_vs_amp(
        ax, runs, lambda r: _final_scalar(r, lambda s: pc.ablation(s, "delta")),
        amps, ns, cmap)
    _amp_axis(ax, amps)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.axhline(0.0, color="0.6", ls=":", lw=1.0)
    ax.set_ylabel("final ablation ΔCE = ce_off − ce_on   (symlog)")
    ax.set_title("CAVEAT — raw ablation ΔCE vs amplitude (measurement confound)")
    ax.legend(title="freq pairs", ncol=2, fontsize=8, framealpha=0.9)
    pc.save(fig, OUT / "amp_scaling_caveat.png", cap=(
        "Fig 5. CAVEAT FIGURE. Raw ablation ΔCE (ce_off−ce_on) scales with "
        "amplitude - a MEASUREMENT CONFOUND, not a behavioural effect: a louder "
        "oracle mechanically blows up CE when removed regardless of true reliance. "
        "Shown for honesty/sanity only. See dependence_accoff_vs_amp.png (Fig 4) "
        "for the meaningful, amplitude-robust version. (grokked only, symlog y.)"))

    # ----------------------------------------------------------------------- #
    # FIG 6 — performance: final test_acc vs amp (only-grokked)
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _seed_scatter_vs_amp(ax, runs, _final_test_acc, amps, ns, cmap)
    _amp_axis(ax, amps)
    ax.set_ylabel("final test accuracy")
    ax.set_title("Performance — final test accuracy vs amplitude (grokked only)")
    ax.legend(title="freq pairs", ncol=2, fontsize=8, framealpha=0.9,
              loc="lower left")
    pc.save(fig, OUT / "perf_vs_amp.png", cap=(
        "Fig 6. Final test accuracy vs amplitude, grokked seeds only (colour=n). "
        "Asks: even when a run DOES grok, does a very loud oracle cost final "
        "performance? Conditioning on grok removes the destabilisation effect "
        "(Fig 1); any residual droop here is a performance cost on top of it."))

    # ----------------------------------------------------------------------- #
    # FIG 7 & 8 — standard W_E spectrum + ablation grids (amp x n)
    # ----------------------------------------------------------------------- #
    fig_we_spectrum(runs, amps, ns, cmap, OUT)
    fig_wl_spectrum(runs, amps, ns, cmap, OUT)      # readout-side mirror (needs wl_* fields)
    fig_ablation(runs, amps, ns, OUT)
    fig_ce_ablation(runs, amps, ns, OUT)

    # ----------------------------------------------------------------------- #
    # FIG 9 & 10 — input (W_E) vs output (logits) frequency usage: is the
    # high-amp laziness a W_E property or a whole-model property?
    # ----------------------------------------------------------------------- #
    fig_input_vs_output_freqs(runs, amps, ns, OUT)
    fig_freq_usage_detail(runs, amps, OUT, n=8)

    # ----------------------------------------------------------------------- #
    print(f"\nwrote figures to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
