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
    print(f"\nwrote figures to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
