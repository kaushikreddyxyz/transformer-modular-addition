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
# per-cell grok counts are shown as badges on the spectrum/ablation grids.
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

pc.SHOW_CAPTIONS = False          # minimalist: no burned-in captions


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
    ax.set_xlabel("amplitude")


# --------------------------------------------------------------------------- #
# amp(rows) x n(cols) panel grids: standard W_E spectrum + ablation, matching
# the exp04 reliability grids (same idiom: sweep1 down rows, n across cols).
# --------------------------------------------------------------------------- #
ON_C, OFF_C = "#1f77b4", "#d62728"
# Shared per-frequency AMPLITUDE axis (= circle radius = sqrt(power/p)). For W_E
# this is residual-stream units (= oracle amp); for W_L it is logit-space units.
AMP_YLIM = (0.0, 2.5)                   # linear, shared across panels

# Injection-site markers: a thin equal-width shaded band centred on each
# injected freq (replaces dashed verticals -- subtler, reads as a "zone").
INJ_BAND_HW = 0.6                       # half-width in freq units (band = 1.2 wide)
                                        # wide enough to stay visible either side of
                                        # a peak (bands sit under the traces)


def _mark_injected(ax, freqs):
    """Shade an equal-width green band at each injected frequency."""
    for f in (freqs or []):
        ax.axvspan(f - INJ_BAND_HW, f + INJ_BAND_HW,
                   color="green", alpha=0.13, lw=0, zorder=0)


def _amp_spectrum(power_full, p):
    """Per-frequency amplitude = circle radius = sqrt(power / p), in the matrix's
    native units — the one quantity directly comparable to the oracle amp."""
    return pc.amp_spectrum(power_full, p)          # shared convention helper


def _p_of(runs):
    return 2 * len(pc.final_snap(runs[0])["we_freq_power_full"]) + 1


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
    """W_E per-frequency amplitude sqrt(power/p) (residual units, = oracle amp).
    Shared linear y-axis; green band = injected freqs."""
    p = _p_of(runs)
    fig, axes = _amp_n_grid(amps, ns, sharey=True)
    for i, amp in enumerate(amps):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            cell = pc.select(runs, amp=amp, n=n)
            specs = [_amp_spectrum(pc.final_snap(r)["we_freq_power_full"], p)
                     for r in cell if pc.final_snap(r).get("we_freq_power_full")]
            _mark_injected(ax, pc.final_snap(cell[0]).get("injected_freqs"))
            if specs:
                freqs = np.arange(1, len(specs[0]) + 1)
                for sp in specs:
                    ax.plot(freqs, sp, color=cmap[n], alpha=0.22, lw=0.7)
                ax.plot(freqs, np.mean(specs, 0), color=cmap[n], lw=1.4)
            ax.set_ylim(*AMP_YLIM)
            _grok_badge(ax, cell)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"amp = {amp:g}", fontsize=9.5)
            ax.tick_params(labelsize=6.5)
    fig.supxlabel("frequency")
    fig.supylabel("amplitude")
    fig.suptitle("W_E spectrum", fontsize=12.5)
    pc.save(fig, out / "we_spectrum.png")


def _has_wl(runs):
    return any(pc.final_snap(r).get("wl_freq_power_full") for r in runs)


def fig_wl_spectrum(runs, amps, ns, cmap, out):
    """Final neuron-logit-map spectrum W_L = W_out^T W_U (readout side), amp(rows)
    x n(cols), injected freqs green band. The output-weight mirror of we_spectrum:
    Nanda et al. 2023's 'neuron-logit map', whose DFT on the answer-token axis is
    sparse on the key freqs. Requires the wl_* snapshot fields (re-run needed)."""
    if not _has_wl(runs):
        print("  [fig_wl_spectrum] no wl_freq_power_full in data -- skipping "
              "(re-run exp02_2 with the updated analysis.py to populate it).")
        return
    p = _p_of(runs)
    fig, axes = _amp_n_grid(amps, ns, sharey=True)
    for i, amp in enumerate(amps):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            cell = pc.select(runs, amp=amp, n=n)
            specs = [_amp_spectrum(pc.final_snap(r)["wl_freq_power_full"], p)
                     for r in cell if pc.final_snap(r).get("wl_freq_power_full")]
            _mark_injected(ax, pc.final_snap(cell[0]).get("injected_freqs"))
            if specs:
                freqs = np.arange(1, len(specs[0]) + 1)
                for sp in specs:
                    ax.plot(freqs, sp, color=cmap[n], alpha=0.22, lw=0.7)
                ax.plot(freqs, np.mean(specs, 0), color=cmap[n], lw=1.4)
            ax.set_ylim(*AMP_YLIM)
            _grok_badge(ax, cell)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"amp = {amp:g}", fontsize=9.5)
            ax.tick_params(labelsize=6.5)
    fig.supxlabel("frequency")
    fig.supylabel("amplitude (logit units)")
    fig.suptitle("W_L spectrum", fontsize=12.5)
    pc.save(fig, out / "wl_spectrum.png")


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
    fig.legend(handles=[Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
                        Line2D([], [], color=OFF_C, lw=2, label="oracle OFF")],
               loc="upper right", bbox_to_anchor=(0.998, 0.998), fontsize=8)
    fig.supxlabel("epoch")
    fig.supylabel("accuracy")
    fig.suptitle("ablation: oracle ON vs OFF", fontsize=12.5)
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
        Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
        Line2D([], [], color=OFF_C, lw=2, label="oracle OFF"),
        Line2D([], [], color="0.6", ls=":", lw=1.2, label="chance")],
        loc="upper right", bbox_to_anchor=(0.998, 0.998), fontsize=8)
    fig.supxlabel("epoch")
    fig.supylabel("cross-entropy")
    fig.suptitle("ablation (CE): oracle ON vs OFF", fontsize=12.5)
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
        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_ylim(0.8, n + 0.5)
    for ax in axes.flat[len(use_ns):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=IN_C, lw=2, marker="o", label="W_E"),
               Line2D([], [], color=OUT_C, lw=2, marker="s", label="logits")]
    if has_wl:
        handles.append(Line2D([], [], color=WL_C, lw=2, marker="^", label="W_L"))
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.998, 0.998), fontsize=9)
    fig.supxlabel("amplitude")
    fig.supylabel("freqs used (N_eff)")
    fig.suptitle("frequencies used: W_E vs W_L vs logits", fontsize=12.5)
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


def fig_contribution_decomp(runs, amps, out, n=8):
    """BUILD vs GROUND TRUTH (residual units), n injected pairs, columns = amp.
    Green ceiling at y=amp is the oracle's KNOWN per-freq contribution; blue stems
    are what W_E rebuilds at each injected freq; the gap is oracle-supplied. Metric
    s = <W_E amp at injected>/amp = self-supply fraction (1 => W_E rebuilds the whole
    oracle freq; 0 => pure free-riding). s FALLS as amp rises => the live oracle
    supplies the rest ('the other contribution').

    W_E-only on purpose: W_L's absolute amplitude is gauge-free (ReLU lets W_in/c,
    W_out*c leave the model identical while W_L scales by c), so it has no fixed
    ruler and can NOT be put on the same residual axis as the oracle. Only W_E is
    anchored (read against the frozen oracle in the residual stream)."""
    p = _p_of(runs)
    L = (p - 1) // 2

    def mean_snap(cell, fn):
        return np.mean([fn(pc.final_snap(r)) for r in cell], 0)

    fig, axes = plt.subplots(1, len(amps), figsize=(3.2 * len(amps), 2.9),
                             sharex=True, sharey=False, squeeze=False)
    for j, amp in enumerate(amps):
        cell = [r for r in pc.select(runs, amp=amp, n=n) if pc.groked(r)]
        ax = axes[0][j]
        ax.set_title(f"amp = {amp:g}", fontsize=10)
        if not cell:
            continue
        inj = pc.final_snap(cell[0]).get("injected_freqs") or []
        ii = [f - 1 for f in inj]                       # injected_freqs are 1-indexed
        we_amp = mean_snap(cell, lambda s: _amp_spectrum(s["we_freq_power_full"], p))
        ax.hlines(amp, 0.5, L + 0.5, color="green", lw=1.1, alpha=0.6)
        ax.vlines(inj, we_amp[ii], amp, color="green", lw=2.0, alpha=0.22)   # gap
        ax.vlines(inj, 0, we_amp[ii], color=IN_C, lw=2.2)                    # W_E built
        ax.scatter(inj, we_amp[ii], color=IN_C, s=14, zorder=3)
        ax.set_ylim(0, amp * 1.12)                      # green ceiling near top
        s_frac = float(np.mean(we_amp[ii]) / amp)
        ax.text(0.95, 0.93, f"s={s_frac:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=IN_C, weight="bold")
    axes[0][0].set_ylabel("amplitude (residual)")
    axes[0][0].legend(handles=[
        Line2D([], [], color=IN_C, lw=2, label="W_E built"),
        Line2D([], [], color="green", lw=2, label="oracle = amp")],
        loc="upper left", fontsize=7, framealpha=0.9)
    fig.supxlabel("frequency")
    fig.suptitle(f"the other contribution (n = {n})", fontsize=12.5)
    pc.save(fig, out / f"contribution_decomp_n{n}.png")


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
    # FIG 2 — laziness: final |W_E| vs amp (only-grokked)
    # ----------------------------------------------------------------------- #
    sp = np.sqrt(_p_of(runs))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _seed_scatter_vs_amp(
        ax, runs,
        lambda r: (lambda v: v / sp if v is not None else None)(
            _final_scalar(r, lambda s: s.get("we_total_norm"))),
        amps, ns, cmap)
    _amp_axis(ax, amps)
    # oracle total per-token contribution amp*sqrt(n) -- SAME residual units
    amp_grid = np.array(amps, float)
    for n in ns:
        ax.plot(amp_grid, amp_grid * np.sqrt(n), color=cmap[n], ls=":", lw=1.2,
                alpha=0.7)
    ax.plot([], [], color="0.4", ls=":", lw=1.2, label="oracle total = amp·√n")
    ax.set_yscale("log")
    ax.set_ylabel("‖W_E‖/√p")
    ax.set_title("W_E norm vs amplitude")
    ax.legend(title="freq pairs", ncol=2, fontsize=8, framealpha=0.9)
    pc.save(fig, OUT / "laziness_we_norm_vs_amp.png", cap=(
        "Fig 2. Final embedding per-token norm ‖W_E‖/√p (residual-stream units — "
        "the aggregate √-sum of the per-frequency amplitudes) vs amplitude; "
        "diamonds=seed mean, dots=grokked seeds, colour=n. Dotted = the oracle's "
        "total per-token contribution amp·√n in the SAME units. ‖W_E‖ DROPS as amp "
        "rises while the oracle total climbs steeply: the embedding offloads onto "
        "the louder oracle. (amp=2, n in {1,2} absent — never grok.)"))

    # ----------------------------------------------------------------------- #
    # FIG 3 — laziness: final frac W_E power on injected freqs vs amp
    # ----------------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _seed_scatter_vs_amp(
        ax, runs, lambda r: _final_scalar(r, pc.frac_we_power_injected),
        amps, ns, cmap)
    _amp_axis(ax, amps)
    ax.set_ylabel("uptake")
    ax.set_title("uptake vs amplitude")
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
    ax.set_ylabel("acc (oracle off)")
    ax.set_title("dependence vs amplitude")
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
    ax.set_ylabel("ΔCE (ce_off − ce_on)")
    ax.set_title("ΔCE vs amplitude (confound)")
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
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy vs amplitude")
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
    fig_contribution_decomp(runs, amps, OUT, n=8)

    # ----------------------------------------------------------------------- #
    print(f"\nwrote figures to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
