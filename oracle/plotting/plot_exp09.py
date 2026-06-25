# %% [markdown]
# # Exp 09 — Organic-oracle figure suite
# Standard readouts for the organic-oracle runs (4 donor baselines × 3 training
# seeds). The grouping axis is the DONOR (which baseline's learned W_E was injected
# verbatim), one panel per donor; the 3 training seeds are faint + a bold seed-mean.
#
# Figures:
#   grok_dynamics — train/test accuracy vs epoch (does it grok, and how fast)
#   ablation      — accuracy with the live oracle ON vs forced OFF (causal use)
#   ce_ablation   — cross-entropy ON vs OFF (CE companion; OFF inflates w/ strength)
#   we_spectrum   — final W_E per-freq amplitude; green band = the donor's freqs
#   wl_spectrum   — final W_L = W_out^T W_U per-freq amplitude (readout side)
#
# The injected freqs are each donor's OWN dominant Fourier modes, so the green band
# differs per panel — the question is whether the student reuses exactly those.
#
# Usage:  .venv/bin/python modular_addition/oracle/plotting/plot_exp09.py [RESULTS_DIR]

# %% imports + path bootstrap
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import plot_common as pc  # noqa: E402

EXP = "exp09"
DEFAULT_RES = _HERE.parents[0] / "results" / "run_20260619_144122"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RES
OUT = RES / "figures" / EXP

ON_C, OFF_C = "#1f77b4", "#d62728"        # oracle ON / OFF (suite convention)
INJ_BAND_HW = 0.6                         # injected-freq shaded band half-width


# --------------------------------------------------------------------------- #
# small helpers (mirror plot_suite / plot_exp02_2 idioms, axis = donor)
# --------------------------------------------------------------------------- #
def _p_of(runs):
    return 2 * len(pc.final_snap(runs[0])["we_freq_power_full"]) + 1


def _amp_spectrum(power_full, p):
    """Per-frequency amplitude sqrt(power/p), in the matrix's native units —
    the quantity directly comparable to the synthetic oracle's amp."""
    return pc.amp_spectrum(power_full, p)          # shared convention helper


def _mark_injected(ax, freqs):
    """Shade an equal-width green band at each injected (donor) frequency."""
    for f in (freqs or []):
        ax.axvspan(f - INJ_BAND_HW, f + INJ_BAND_HW,
                   color="green", alpha=0.13, lw=0, zorder=0)


def _snap_x(fn):
    return lambda r: np.clip(pc.snap_series(r, fn)[0], 1, None)


def _snap_y(fn):
    return lambda r: pc.snap_series(r, fn)[1]


def _abl(key):
    return lambda s: (s.get("ablation_test") or {}).get(key)


def _grok_badge(ax, cell):
    gk = sum(pc.groked(r) for r in cell)
    col = "0.3" if gk == len(cell) else ("#b00020" if gk == 0 else "#9a6a00")
    ax.text(0.96, 0.06, f"{gk}/{len(cell)} grok", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=col)


def _donor_grid(donors, panel=(4.5, 3.2), sharey=True):
    """2-col grid, one panel per donor."""
    ncol = 2
    nrow = (len(donors) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=sharey,
                             figsize=(panel[0] * ncol, panel[1] * nrow),
                             squeeze=False)
    return fig, axes.flat


def _title(ax, donor, cell):
    inj = pc.final_snap(cell[0]).get("injected_freqs") or []
    ax.set_title(f"donor {donor}   freqs={inj}", fontsize=9.5)


# --------------------------------------------------------------------------- #
# 1) grokking dynamics — train + test accuracy vs epoch
# --------------------------------------------------------------------------- #
def fig_grok_dynamics(runs, donors, cmap, censor, p, out):
    fig, axes = _donor_grid(donors)
    for ax, donor in zip(axes, donors):
        cell = pc.select(runs, donor=donor)
        pc.seed_family(ax, cell,
                       x_fn=lambda r: np.clip(pc.hist_series(r, "train_acc")[0], 1, None),
                       y_fn=lambda r: pc.hist_series(r, "train_acc")[1],
                       color="0.6", alpha_seed=0.12, lw_mean=1.2)
        pc.seed_family(ax, cell,
                       x_fn=lambda r: np.clip(pc.hist_series(r, "test_acc")[0], 1, None),
                       y_fn=lambda r: pc.hist_series(r, "test_acc")[1],
                       color=cmap[donor], alpha_seed=0.25, lw_mean=1.9)
        ax.axhline(1.0 / p, color="0.6", ls=":", lw=0.8)
        gk = [pc.grok_epoch(r) for r in cell if pc.grok_epoch(r) is not None]
        if gk:
            ax.text(0.96, 0.18, f"~{int(np.median(gk))} ep", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=cmap[donor])
        _grok_badge(ax, cell)
        _title(ax, donor, cell)
        ax.set_xscale("log")
        ax.set_xlim(1, censor)
        ax.set_ylim(-0.03, 1.05)
    fig.legend(handles=[Line2D([], [], color="0.6", lw=2, label="train acc"),
                        Line2D([], [], color="0.25", lw=2, label="test acc")],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=8)
    fig.supxlabel("epoch (log)")
    fig.supylabel("accuracy")
    fig.suptitle(f"Exp09 — grokking dynamics with an organic oracle (p={p})",
                 fontsize=12.5)
    pc.save(fig, out / "grok_dynamics.png", cap=(
        "Train (gray) and test (colour) accuracy vs epoch, one panel per donor "
        "baseline whose learned W_E was injected verbatim (3 seeds faint + "
        f"seed-mean bold; dotted = chance 1/{p}). The organic oracle is live from "
        "epoch 0, so the grok epoch (annotated) is the fast onset of test "
        "accuracy."))


# --------------------------------------------------------------------------- #
# 2) accuracy ablation — oracle ON vs OFF over training
# --------------------------------------------------------------------------- #
def fig_ablation(runs, donors, censor, out):
    fig, axes = _donor_grid(donors)
    for ax, donor in zip(axes, donors):
        cell = pc.select(runs, donor=donor)
        pc.seed_family(ax, cell, x_fn=_snap_x(_abl("acc_on")),
                       y_fn=_snap_y(_abl("acc_on")), color=ON_C,
                       alpha_seed=0.15, lw_mean=1.8)
        pc.seed_family(ax, cell, x_fn=_snap_x(_abl("acc_off")),
                       y_fn=_snap_y(_abl("acc_off")), color=OFF_C,
                       alpha_seed=0.15, lw_mean=1.8)
        _grok_badge(ax, cell)
        _title(ax, donor, cell)
        ax.set_xscale("log")
        ax.set_xlim(min(180, censor // 10), censor)
        ax.set_ylim(-0.03, 1.05)
    fig.legend(handles=[Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
                        Line2D([], [], color=OFF_C, lw=2, label="oracle OFF")],
               loc="lower right", bbox_to_anchor=(0.995, 0.06), fontsize=8)
    fig.supxlabel("epoch (log)")
    fig.supylabel("test accuracy")
    fig.suptitle("Exp09 — causal use: accuracy with the organic oracle ON vs OFF",
                 fontsize=12.5)
    pc.save(fig, out / "ablation.png", cap=(
        "Test accuracy with the live oracle ON (blue) vs forced OFF at inference "
        "(red), per donor (3 seeds + mean). The ON-OFF gap is the behavioural "
        "dependence: small gap ⇒ the model internalised the feature into weights; "
        "large gap ⇒ it leans on the live organic signal."))


# --------------------------------------------------------------------------- #
# 3) CE ablation — oracle ON vs OFF over training
# --------------------------------------------------------------------------- #
def fig_ce_ablation(runs, donors, censor, p, out):
    chance = float(np.log(p))
    fig, axes = _donor_grid(donors)
    for ax, donor in zip(axes, donors):
        cell = pc.select(runs, donor=donor)
        ax.axhline(chance, color="0.6", ls=":", lw=0.8)
        pc.seed_family(ax, cell, x_fn=_snap_x(_abl("ce_on")),
                       y_fn=_snap_y(_abl("ce_on")), color=ON_C,
                       alpha_seed=0.15, lw_mean=1.8)
        pc.seed_family(ax, cell, x_fn=_snap_x(_abl("ce_off")),
                       y_fn=_snap_y(_abl("ce_off")), color=OFF_C,
                       alpha_seed=0.15, lw_mean=1.8)
        _grok_badge(ax, cell)
        _title(ax, donor, cell)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min(180, censor // 10), censor)
    fig.legend(handles=[
        Line2D([], [], color=ON_C, lw=2, label="oracle ON"),
        Line2D([], [], color=OFF_C, lw=2, label="oracle OFF"),
        Line2D([], [], color="0.6", ls=":", lw=1.2, label="chance (ln p)")],
        loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=8)
    fig.supxlabel("epoch (log)")
    fig.supylabel("cross-entropy (log)")
    fig.suptitle("Exp09 — ablation (CE): organic oracle ON vs OFF", fontsize=12.5)
    pc.save(fig, out / "ce_ablation.png", cap=(
        "Test cross-entropy with the live oracle ON (blue) vs forced OFF (red), "
        "per donor (3 seeds + mean; dotted = chance, ln p). CAVEAT: the ON-OFF CE "
        "gap is not amplitude-robust — removing the feature mechanically inflates "
        "CE; read the accuracy version (ablation.png) for behavioural dependence."))


# --------------------------------------------------------------------------- #
# 4 + 5) final W_E / W_L per-frequency amplitude spectra
# --------------------------------------------------------------------------- #
def _spectrum_fig(runs, donors, cmap, p, key, unit, title, fname, out):
    fig, axes = _donor_grid(donors, sharey=True)
    for ax, donor in zip(axes, donors):
        cell = pc.select(runs, donor=donor)
        specs = [_amp_spectrum(pc.final_snap(r)[key], p)
                 for r in cell if pc.final_snap(r).get(key)]
        _mark_injected(ax, pc.final_snap(cell[0]).get("injected_freqs"))
        if specs:
            freqs = np.arange(1, len(specs[0]) + 1)
            for sp in specs:
                ax.plot(freqs, sp, color=cmap[donor], alpha=0.30, lw=0.8)
            ax.plot(freqs, np.mean(specs, 0), color=cmap[donor], lw=1.6)
        ax.set_ylim(bottom=0)
        _title(ax, donor, cell)
        ax.margins(x=0.01)
    fig.supxlabel(f"Fourier frequency index (1..{(p - 1) // 2})")
    fig.supylabel(f"amplitude  √(power/p)  ({unit})")
    fig.suptitle(title, fontsize=12.5)
    pc.save(fig, out / fname, cap=(
        f"Final per-frequency amplitude √(power/p) of {title.split('—')[1].strip()}, "
        "one panel per donor (3 seeds faint + mean bold). Green band = the donor's "
        "own dominant frequencies (the organic oracle's injected sites); peaks "
        "landing on the band mean the student reuses exactly the donor's freqs."))


def fig_we_spectrum(runs, donors, cmap, p, out):
    _spectrum_fig(runs, donors, cmap, p, "we_freq_power_full", "residual units",
                  "Exp09 — final W_E spectrum (organic oracle)",
                  "we_spectrum.png", out)


def fig_wl_spectrum(runs, donors, cmap, p, out):
    if not any(pc.final_snap(r).get("wl_freq_power_full") for r in runs):
        print("  [wl_spectrum] no wl_freq_power_full in data — skipping.")
        return
    _spectrum_fig(runs, donors, cmap, p, "wl_freq_power_full", "logit units",
                  "Exp09 — final W_L spectrum (neuron-logit map)",
                  "wl_spectrum.png", out)


# =========================================================================== #
def main():
    pc.set_style()
    runs = pc.load_exp(RES, EXP)
    donors = pc.axis_values(runs, "donor")
    cmap = pc.color_map(donors, baseline=None)
    p = _p_of(runs)
    censor = max(r["num_epochs"] for r in runs)
    n_grok = sum(pc.groked(r) for r in runs)
    print(f"loaded {len(runs)} runs | donors={donors} | p={p} | "
          f"grokked {n_grok}/{len(runs)}")

    fig_grok_dynamics(runs, donors, cmap, censor, p, OUT)
    fig_ablation(runs, donors, censor, OUT)
    fig_ce_ablation(runs, donors, censor, p, OUT)
    fig_we_spectrum(runs, donors, cmap, p, OUT)
    fig_wl_spectrum(runs, donors, cmap, p, OUT)

    print(f"\nwrote figures to {OUT}")
    for f in sorted(OUT.glob("*.png")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
