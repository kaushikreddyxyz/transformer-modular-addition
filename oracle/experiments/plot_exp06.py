# %% [markdown]
# # Exp 06 figures — exp01's EXACT plot suite, applied to a different model
# Faithful reproduction of plot_exp01.py: same metrics, same plot types, same
# styling, same filenames. The ONLY difference is the model/data — exp06 is
# p=211 (105 Fourier freqs), d_model=256, frac_train=0.075, 75k epochs, with a
# contiguous n=0..11 sweep. Three things are derived from the data rather than
# hard-coded, so the same code renders both models correctly:
#   * the per-n subplot grids auto-size to however many n exist (11 here vs 6),
#   * the grok "never-grokked" censor cap = the model's num_epochs (75k vs 30k),
#   * titles/captions name the model and stay descriptive (exp01's interpretive
#     claims do NOT all hold here — e.g. dependence GROWS with n at p=211).
# NOTE: `excluded_loss_injected` is 100% NaN at p=211 (calculate_excluded_loss
# blows up at the larger prime), so the "necessity" panels render blank; the
# trig/sufficiency and Gini panels are valid.
#   .venv/bin/python modular_addition/oracle/experiments/plot_exp06.py [results_dir]

# %% imports + data
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import caption  # noqa: E402

EXP_NAME = "exp06"
LABEL = "Exp06"
RES = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[1] / "results" / "latest")
EXP = RES / EXP_NAME
FIG = RES / "figures" / EXP_NAME
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})

runs = {}                               # (n, seed) -> result dict
for f in sorted(EXP.glob("*.result.json")):
    r = json.load(open(f))
    runs[(r["spec"]["axes"]["n"], r["spec"]["seed"])] = r

NS = sorted({n for n, _ in runs})       # [0, 1, ..., 11]
SEEDS = sorted({s for _, s in runs})
CAP = max(r["num_epochs"] for r in runs.values())   # 75000 (was 30000 in exp01)
cm = plt.get_cmap("viridis")
COL = {n: ("0.25" if n == 0 else cm(0.05 + 0.85 * i / max(1, len(NS) - 2)))
       for i, n in enumerate(n for n in NS if n)} | {0: "0.25"}

A_SEED, A_MEAN = 0.30, 1.0              # opacities: individual seed / mean


def hist(n, seed, key):
    h = runs[(n, seed)]["history"]
    return np.array([r["epoch"] for r in h]), np.array([r[key] for r in h])


def snap(n, seed, fn):
    """[(epoch, fn(snapshot))] for one run; fn may return None to skip."""
    out = [(s["epoch"], fn(s)) for s in runs[(n, seed)]["snapshots"]]
    keep = [(e, v) for e, v in out if v is not None]
    return np.array(keep).T if keep else np.empty((2, 0))


def seed_lines(ax, n, xy_of, lw=1.0):
    """Plot all seeds of one n (alpha A_SEED) + their mean (alpha A_MEAN)."""
    xs, ys = zip(*(xy_of(n, s) for s in SEEDS))
    for x, y in zip(xs, ys):
        ax.plot(x, y, color=COL[n], alpha=A_SEED, lw=lw)
    grid = max(xs, key=len)             # densest seed grid (handles empties)
    if len(grid) == 0:
        return
    mean = np.mean([np.interp(grid, x, y) for x, y in zip(xs, ys) if len(x)],
                   axis=0)
    ax.plot(grid, mean, color=COL[n], alpha=A_MEAN, lw=1.8,
            label=f"n={n}", zorder=3)


def _grid_dims(k, ncol=3):
    return math.ceil(k / ncol), ncol


def overlay(xy_of, ylabel, name, logx=True, ylim=None, title="", cap=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in NS:
        seed_lines(ax, n, xy_of)
    ax.set(xlabel="epoch", ylabel=ylabel, title=title)
    if logx:
        ax.set_xscale("log")
        ax.set_xlim(left=180)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=2, fontsize=9, title="injected pairs",
              title_fontsize=9, framealpha=0.9)
    if cap:
        caption(fig, cap)
    fig.savefig(FIG / f"{name}_overlay.png", bbox_inches="tight")
    plt.close(fig)


def subplots(xy_of, ylabel, name, logx=True, ylim=None, title="",
             baseline_ref=True, ns=None, cap=None):
    ns = ns if ns is not None else [n for n in NS if n]
    nrow, ncol = _grid_dims(len(ns))           # auto-size (2x3 for exp01's 6)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.5 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for ax, n in zip(axes.flat, ns):
        if baseline_ref and 0 in NS:           # n=0 mean as reference per panel
            x0, _ = xy_of(0, SEEDS[0])
            m0 = np.mean([np.interp(x0, *xy_of(0, s)) for s in SEEDS], axis=0)
            ax.plot(x0, m0, color="0.25", ls="--", lw=1.2, label="n=0 mean")
        for s in SEEDS:
            ax.plot(*xy_of(n, s), color=COL[n], alpha=0.65, lw=1.1)
        ax.set_title(f"n = {n}", fontsize=10)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(left=180)
        if ylim:
            ax.set_ylim(*ylim)
    for ax in axes.flat[len(ns):]:
        ax.set_visible(False)
    if baseline_ref:
        axes.flat[0].legend(fontsize=8)
    fig.supxlabel("epoch")
    fig.supylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    if cap:
        caption(fig, cap)
    fig.savefig(FIG / f"{name}_subplots.png", bbox_inches="tight")
    plt.close(fig)


# %% main hypothesis — grokking dynamics: test accuracy vs epoch
CAP_ACC = (f"{LABEL} (p=211, frac_train=0.075, 75k ep). Test accuracy vs epoch "
           "by injected-pair count n (4 seeds + mean). In this low-data regime "
           "n=0,1,2 never grok; generalization is rescued at n>=3.")
acc = lambda n, s: hist(n, s, "test_acc")
overlay(acc, "test accuracy", "acc", ylim=(-0.02, 1.02),
        title=f"{LABEL} — grokking vs n injected pairs (4 seeds + mean)",
        cap=CAP_ACC)
subplots(acc, "test accuracy", "acc", ylim=(-0.02, 1.02),
         title=f"{LABEL} — test accuracy per n (4 seeds; gray dashed = n=0 mean)",
         cap=CAP_ACC)

# %% main hypothesis — time-to-grok summary
fig, ax = plt.subplots(figsize=(7, 4.6))
rng = np.random.default_rng(0)
for i, n in enumerate(NS):
    g = [runs[(n, s)]["grok_epoch"] for s in SEEDS]
    xj = i + rng.uniform(-0.08, 0.08, len(g))
    grokked = [(x, v) for x, v in zip(xj, g) if v is not None]
    failed = [x for x, v in zip(xj, g) if v is None]
    if grokked:
        ax.scatter(*zip(*grokked), color=COL[n], alpha=0.55, s=38, zorder=3)
    if failed:
        ax.scatter(failed, [CAP] * len(failed), color=COL[n], marker="^",
                   facecolors="none", s=60, zorder=3,
                   label=f"never grokked (≥{CAP // 1000}k)" if not any(
                       runs[(m, s)]["grok_epoch"] is None
                       for m in NS[:i] for s in SEEDS) else None)
    ax.scatter([i], [np.mean([v if v is not None else CAP for v in g])],
               color=COL[n], marker="_", s=600, lw=2.5)
ax.set(xticks=range(len(NS)), xticklabels=NS, xlabel="n injected pairs",
       ylabel="grok epoch (test acc ≥ 0.99)", yscale="log",
       title=f"{LABEL} — time to grok (dots = seeds, dash = mean, "
             f"censored at {CAP // 1000}k)")
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(fontsize=8, loc="center right")
caption(fig, f"Epochs to grok (test acc >=0.99) vs n; dots=seeds, dash=mean, "
             f"triangles=never-grokked (censored at {CAP // 1000}k). Sharp "
             "rescue threshold at n=3 (n=0,1,2 never grok at p=211 low-data).")
fig.savefig(FIG / "grok_vs_n.png", bbox_inches="tight")
plt.close(fig)

# %% main hypothesis — uptake: W_E spectral power fraction at injected freqs
frac_inj = lambda s: (sum(s["we_freq_power_injected"]) /
                      sum(s["we_freq_power_full"])
                      if s["we_freq_power_injected"] else None)
up = lambda n, s: snap(n, s, frac_inj)
key = lambda n, s: snap(n, s, lambda sn: (len(sn["injected_in_key_freqs"]) /
                                          len(sn["injected_freqs"])
                                          if sn["injected_freqs"] else None))
NS_INJ = [n for n in NS if n]


def overlay_inj(xy_of, ylabel, name, title, ylim=None, cap=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in NS_INJ:
        seed_lines(ax, n, xy_of)
    ax.set(xlabel="epoch (snapshots every 2k)", ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=2, fontsize=9, title="injected pairs", title_fontsize=9)
    if cap:
        caption(fig, cap)
    fig.savefig(FIG / f"{name}_overlay.png", bbox_inches="tight")
    plt.close(fig)


CAP_UPTAKE = ("Fraction of W_E Fourier power on the injected freqs vs epoch. "
              "Grokking runs (n>=3) concentrate ~70-95% of embedding power on "
              "the oracle's frequencies — direct uptake into trainable weights.")
overlay_inj(up, "W_E power fraction at injected freqs", "uptake",
            f"{LABEL} — embedding uptake of injected freqs (4 seeds + mean)",
            cap=CAP_UPTAKE)
subplots(up, "W_E power fraction at injected freqs", "uptake", logx=False,
         title=f"{LABEL} — embedding uptake per n", baseline_ref=False,
         cap=CAP_UPTAKE)
overlay_inj(key, "fraction of injected freqs in key freqs", "injected_in_key",
            f"{LABEL} — injected freqs adopted as key freqs", ylim=(-0.05, 1.05),
            cap="Fraction of injected freqs retained in the model's key "
                "(working) frequencies. The retained fraction falls as more "
                "pairs are offered than the network keeps.")

# %% sanity — ablation: model.inject=False at inference (whole oracle off;
# all-or-nothing — learned weights keep their uptake, the live signal is cut)
abl_delta = lambda n, s: snap(n, s, lambda sn: sn.get("ablation_test",
                                                      {}).get("delta")
                              if sn.get("ablation_test") else None)
abl_acc = lambda n, s: snap(n, s, lambda sn: sn.get("ablation_test",
                                                    {}).get("acc_off")
                            if sn.get("ablation_test") else None)
CAP_ABL_ACC = ("Test accuracy with the live oracle switched off at inference, "
               "by n (low = the model leans on the live signal). At p=211 the "
               "ON-OFF gap GROWS with n — the reverse of the p=113 trend.")
CAP_ABL_DELTA = ("CE increase when the oracle is switched off (causal "
                 "dependence on the live signal) vs epoch, by n.")
overlay_inj(abl_delta, "CE(oracle off) − CE(oracle on)", "ablation_delta",
            f"{LABEL} sanity — CE jump with oracle injection disabled",
            cap=CAP_ABL_DELTA)
overlay_inj(abl_acc, "test accuracy, oracle injection off",
            "ablation_acc", f"{LABEL} sanity — accuracy without the live oracle "
            "signal (low = model leans on it at inference)", ylim=(-0.02, 1.02),
            cap=CAP_ABL_ACC)
subplots(abl_delta, "CE(oracle off) − CE(oracle on)", "ablation_delta",
         logx=False, baseline_ref=False, ns=NS_INJ,
         title=f"{LABEL} sanity — oracle-off CE delta per n (4 seeds)",
         cap=CAP_ABL_DELTA)
subplots(abl_acc, "test accuracy, oracle injection off",
         "ablation_acc", logx=False, ylim=(-0.02, 1.02), baseline_ref=False,
         ns=NS_INJ,
         title=f"{LABEL} sanity — accuracy with oracle off, per n (4 seeds)",
         cap=CAP_ABL_ACC)

# final-snapshot ablation summary vs n
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, n in enumerate(NS_INJ):
    on = [runs[(n, s)]["snapshots"][-1]["ablation_test"]["acc_on"]
          for s in SEEDS]
    off = [runs[(n, s)]["snapshots"][-1]["ablation_test"]["acc_off"]
           for s in SEEDS]
    x = i + rng.uniform(-0.06, 0.06, len(SEEDS))
    ax.scatter(x, on, color=COL[n], alpha=0.8, s=36,
               label="oracle on" if i == 0 else None)
    ax.scatter(x, off, color=COL[n], alpha=0.8, s=42, marker="x",
               label="oracle off" if i == 0 else None)
ax.set(xticks=range(len(NS_INJ)), xticklabels=NS_INJ,
       xlabel="n injected pairs", ylabel="final test accuracy",
       title=f"{LABEL} sanity — final accuracy, oracle on vs off (whole mechanism)")
ax.legend()
caption(fig, "Final accuracy, oracle on vs off, per n. The on-off gap = "
             "dependence on the live signal. At p=211 the gap GROWS with n "
             "(larger bases lean on the oracle MORE — opposite to p=113).")
fig.savefig(FIG / "ablation_final.png", bbox_inches="tight")
plt.close(fig)

# %% sanity — excluded loss at injected freqs + W_E spectral concentration
# NOTE: excluded_loss_injected is NaN at p=211 -> left panel renders blank.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for n in NS_INJ:
    seed_lines(axes[0], n,
               lambda n_, s_: snap(n_, s_,
                                   lambda sn: (np.mean(
                                       sn["excluded_loss_injected"])
                                       if sn["excluded_loss_injected"]
                                       else None)))
for n in NS:
    seed_lines(axes[1], n, lambda n_, s_: snap(n_, s_,
                                               lambda sn: sn["we_gini"]))
axes[0].set(xlabel="epoch", ylabel="mean excluded loss (injected freqs)",
            title="excluded loss (NaN at p=211 — blank)")
axes[0].text(0.5, 0.5, "excluded_loss_injected\nis NaN at p=211",
             transform=axes[0].transAxes, ha="center", va="center",
             color="0.5", fontsize=11)
axes[1].set(xlabel="epoch", ylabel="W_E spectral Gini",
            title="embedding spectral concentration")
axes[1].legend(ncol=2, fontsize=8, title="n", title_fontsize=8)
fig.tight_layout()
caption(fig, "Per-injected-freq excluded loss (necessity) and W_E spectral "
             "Gini (concentration), by n. Excluded loss is NaN at p=211 so the "
             "left panel is blank; necessity is shown causally by the ablation "
             "figures instead.")
fig.savefig(FIG / "sanity_excluded_gini.png", bbox_inches="tight")
plt.close(fig)

# %% mechanistic — trig (sufficiency) vs excluded (necessity) loss
# trig_loss_injected: loss from ONLY the injected freqs' logit components —
# low means they alone reproduce the model. excluded_loss_injected is NaN here.
trig = lambda n, s: snap(n, s, lambda sn: sn["trig_loss_injected"])
excl = lambda n, s: snap(n, s, lambda sn: (np.mean(sn["excluded_loss_injected"])
                                           if sn["excluded_loss_injected"]
                                           else None))

fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
for n in NS_INJ:
    seed_lines(axs[0], n, trig)
    seed_lines(axs[1], n, excl)
axs[0].set(yscale="log", xlabel="epoch", ylabel="trig loss (injected only)",
           title="sufficiency: loss from injected freqs alone")
axs[1].set(xlabel="epoch", ylabel="mean excluded loss (injected)",
           title="necessity (NaN at p=211 — blank)")
axs[1].text(0.5, 0.5, "excluded_loss_injected\nis NaN at p=211",
            transform=axs[1].transAxes, ha="center", va="center",
            color="0.5", fontsize=11)
axs[0].legend(ncol=2, fontsize=8, title="n", title_fontsize=8)
fig.suptitle(f"{LABEL} — injected freqs: sufficiency vs necessity (4 seeds + mean)")
fig.tight_layout()
CAP_TRIG = ("Sufficiency (trig loss: injected freqs alone) vs necessity "
            "(excluded loss). For n>=3 the injected-only loss falls sharply "
            "(sufficient). Necessity (excluded loss) is NaN at p=211 -> blank.")
caption(fig, CAP_TRIG)
fig.savefig(FIG / "trig_excluded_overlay.png", bbox_inches="tight")
plt.close(fig)

nrow, ncol = _grid_dims(len(NS_INJ))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.5 * nrow),
                         sharex=True, sharey=True, squeeze=False)
for ax, n in zip(axes.flat, NS_INJ):
    for s in SEEDS:
        ax.plot(*hist(n, s, "test_loss"), color="0.55", alpha=0.4, lw=0.9,
                label="full test loss" if (n == NS_INJ[0] and s == 0) else None)
        ax.plot(*trig(n, s), color=COL[n], alpha=0.7, lw=1.3,
                label="trig (sufficiency)" if (n == NS_INJ[0] and s == 0)
                else None)
        ax.plot(*excl(n, s), color="crimson", alpha=0.55, lw=1.1, ls="--",
                label="excluded (necessity)" if (n == NS_INJ[0] and s == 0)
                else None)
    ax.set_title(f"n = {n}", fontsize=10)
    ax.set_yscale("log")
for ax in axes.flat[len(NS_INJ):]:
    ax.set_visible(False)
axes.flat[0].legend(fontsize=8, loc="lower left")
fig.supxlabel("epoch")
fig.supylabel("loss (log)")
fig.suptitle(f"{LABEL} — trig vs excluded loss per n (4 seeds; excluded is NaN "
             "at p=211)")
fig.tight_layout()
caption(fig, CAP_TRIG)
fig.savefig(FIG / "trig_excluded_subplots.png", bbox_inches="tight")
plt.close(fig)

# %% mechanistic — final W_E Fourier spectrum (same regime panels as exp01)
# All 4 seeds per panel; green dashed = injected, red v = non-injected key freqs.
SPEC_NS = [n for n in (0, 2, 3, 5, 6, 8) if n in NS]
nrow, ncol = _grid_dims(len(SPEC_NS))
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.6 * nrow),
                         sharex=True, squeeze=False)
for ax, n in zip(axes.flat, SPEC_NS):
    inj = runs[(n, SEEDS[0])]["injected_freqs"]
    for f in inj:
        ax.axvline(f, color="tab:green", ls="--", lw=1.1, alpha=0.8, zorder=1)
    for s in SEEDS:
        spec = runs[(n, s)]["snapshots"][-1]["we_freq_power_full"]
        ax.plot(range(1, len(spec) + 1), spec, color=COL[n], alpha=0.55,
                lw=1.2, zorder=2)
        extra = [f for f in runs[(n, s)]["snapshots"][-1]["key_freqs"]
                 if f not in inj]
        ax.plot(extra, [max(spec) * 1.06] * len(extra), ls="none",
                marker="v", color="tab:red", ms=5, alpha=0.7, zorder=3)
    ax.set_title(f"n = {n}" + ("  (baseline)" if n == 0 else
                               f"  (injected: {inj})"), fontsize=10)
for ax in axes.flat[len(SPEC_NS):]:
    ax.set_visible(False)
fig.supxlabel(f"frequency (1..{len(runs[(NS_INJ[0], SEEDS[0])]['snapshots'][-1]['we_freq_power_full'])})")
fig.supylabel("final W_E Fourier power")
fig.suptitle(f"{LABEL} mechanistic — W_E spectrum, 4 seeds per panel "
             "(green dashed = injected, red ▼ = non-injected key freqs)")
fig.tight_layout()
caption(fig, "Final W_E Fourier spectrum per regime (4 seeds; green "
             "dashed=injected, red v=non-injected key freqs). At p=211 the "
             "spectrum spans 105 freqs; power for grokking n lands on the "
             "injected sites. n=2 (never groks) is diffuse.")
fig.savefig(FIG / "we_spectrum.png", bbox_inches="tight")
plt.close(fig)

print(f"wrote {len(list(FIG.glob('*.png')))} figures -> {FIG}")
