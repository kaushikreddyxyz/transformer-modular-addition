"""Shared plotting conventions for the oracle-injection experiment figures.

Every per-experiment figure script (plot_exp0*.py) imports from here so the
suite is visually and methodologically consistent:

  * 4 seeds drawn as faint lines + the seed-mean drawn bold (seed_family)
  * a continuous sweep axis (n / amp / rel) -> viridis; baseline -> gray
  * never-grokked runs are NEVER silently averaged into means: filter with
    `groked` and report counts, or show them explicitly (censored markers /
    success-rate heatmaps)
  * a one-line caption burned into the bottom of every PNG (save(..., cap=...))

Data model: each run is the parsed `<label>.result.json`. We attach `_axes`
(= spec.axes) and `_label`. Snapshot metrics live in r["snapshots"] (every
2000 epochs; exp02_1 every 1000), scalars in r["history"] (every 200).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
        "savefig.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 11, "legend.fontsize": 9,
    })


# --------------------------------------------------------------------------- #
# loading + indexing
# --------------------------------------------------------------------------- #
def load_exp(results_dir, exp):
    """List of result dicts for one experiment, each with _axes and _label."""
    runs = []
    for f in sorted((Path(results_dir) / exp).glob("*.result.json")):
        r = json.load(open(f))
        r["_axes"] = r.get("spec", {}).get("axes", {})
        r["_label"] = r.get("label", f.stem.replace(".result", ""))
        runs.append(r)
    if not runs:
        raise SystemExit(f"no result.json found in {Path(results_dir)/exp}")
    return runs


def axis_values(runs, key):
    """Sorted distinct values of an axis present across runs."""
    return sorted({r["_axes"][key] for r in runs if key in r["_axes"]})


def select(runs, **filt):
    """Runs whose _axes match all key=value filters."""
    return [r for r in runs
            if all(r["_axes"].get(k) == v for k, v in filt.items())]


# --------------------------------------------------------------------------- #
# series extraction
# --------------------------------------------------------------------------- #
def hist_series(r, key):
    h = r["history"]
    return (np.array([x["epoch"] for x in h]),
            np.array([x[key] for x in h], dtype=float))


def snap_series(r, fn):
    """(epochs, values) over snapshots; fn(snapshot)->float or None (skipped)."""
    pts = [(s["epoch"], fn(s)) for s in r["snapshots"]]
    pts = [(e, v) for e, v in pts if v is not None]
    if not pts:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts], float)


def final_snap(r):
    return r["snapshots"][-1] if r["snapshots"] else {}


# common snapshot-derived scalars (return None when undefined for the run)
def frac_we_power_injected(s):
    """fraction of W_E Fourier power sitting on the injected freqs."""
    inj, full = s.get("we_freq_power_injected"), s.get("we_freq_power_full")
    if not inj or not full:
        return None
    tot = float(np.sum(full))
    return float(np.sum(inj)) / tot if tot > 0 else None


def frac_injected_in_key(s):
    inj = s.get("injected_freqs")
    if not inj:
        return None
    return len(s.get("injected_in_key_freqs", [])) / len(inj)


def n_key_freqs(s):
    kf = s.get("key_freqs")
    return len(kf) if kf is not None else None


def ablation(s, sub):
    a = s.get("ablation_test")
    return a.get(sub) if isinstance(a, dict) else None


# --------------------------------------------------------------------------- #
# grok helpers
# --------------------------------------------------------------------------- #
def groked(r):
    return r.get("grok_epoch") is not None


def grok_epoch(r):
    return r.get("grok_epoch")


def grok_rate(runs):
    """fraction of runs that grokked (0..1); 0 runs -> nan."""
    return np.mean([groked(r) for r in runs]) if runs else np.nan


# --------------------------------------------------------------------------- #
# color
# --------------------------------------------------------------------------- #
def color_map(values, cmap="viridis", baseline=None, baseline_color="0.4"):
    """Map sweep values -> colors. `baseline` (e.g. n=0) gets gray."""
    vals = list(values)
    body = [v for v in vals if v != baseline]
    cm = plt.get_cmap(cmap)
    out = {v: cm(0.08 + 0.84 * i / max(1, len(body) - 1))
           for i, v in enumerate(sorted(body))}
    if baseline in vals:
        out[baseline] = baseline_color
    return out


# --------------------------------------------------------------------------- #
# the workhorse: plot a family of same-config seeds (faint) + their mean (bold)
# --------------------------------------------------------------------------- #
def seed_family(ax, runs, x_fn, y_fn, color, label=None,
                alpha_seed=0.28, lw_mean=1.9, only_groked=False):
    """Draw each run's (x_fn, y_fn) faint; overlay the interpolated mean bold.

    Returns the number of runs actually drawn (0 if none had data). Set
    only_groked=True to exclude never-grokked runs from BOTH lines and mean.
    """
    use = [r for r in runs if (groked(r) or not only_groked)]
    series = [(x_fn(r), y_fn(r)) for r in use]
    series = [(x, y) for x, y in series if len(x) and len(y) and len(x) == len(y)]
    if not series:
        return 0
    for x, y in series:
        ax.plot(x, y, color=color, alpha=alpha_seed, lw=1.0)
    grid = max((s[0] for s in series), key=len)        # densest seed's x-grid
    M = np.vstack([np.interp(grid, x, y) for x, y in series])
    ax.plot(grid, M.mean(0), color=color, alpha=1.0, lw=lw_mean,
            label=label, zorder=3)
    return len(series)


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def caption(fig, text):
    """Burn a one-line (wrapped) caption into the bottom margin of the PNG."""
    fig.text(0.5, -0.015, text, ha="center", va="top", fontsize=8.5,
             color="0.30", wrap=True)


# Set False to suppress the burned-in captions (minimalist figures).
SHOW_CAPTIONS = True


def save(fig, path, cap=None, tight=True):
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    if cap and SHOW_CAPTIONS:
        caption(fig, cap)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
