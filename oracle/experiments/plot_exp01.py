# %% [markdown]
# # Exp 01 figures — per-seed uptake plots
# Unlike make_figures.py (seed-mean ± std bands), every seed is drawn as its
# own low-opacity line with the seed-mean on top, per the "show all 4 seeds"
# requirement. Each curve figure is produced in two layouts — a single overlay
# axes (color = n, alpha = seed) and a per-n subplot grid — so the legible one
# can be picked per metric.
#
#   main hypothesis:  acc_*.png        grok dynamics vs n
#                     grok_vs_n.png    time-to-grok summary
#                     uptake_*.png     W_E spectral power at injected freqs
#   sanity checks:    ablation_*.png   zeroing injected freqs hurts (causal use)
#                     sanity_excluded_gini.png

# %% imports + data
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_common import caption  # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[1] / "results" / "latest")
EXP = RES / "exp01"
FIG = RES / "figures" / "exp01"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})

runs = {}                               # (n, seed) -> result dict
for f in sorted(EXP.glob("*.result.json")):
    r = json.load(open(f))
    runs[(r["spec"]["axes"]["n"], r["spec"]["seed"])] = r

NS = sorted({n for n, _ in runs})       # [0, 1, 2, 3, 5, 6, 8]
SEEDS = sorted({s for _, s in runs})
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
    return np.array([(e, v) for e, v in out if v is not None]).T


def seed_lines(ax, n, xy_of, lw=1.0):
    """Plot all seeds of one n (alpha A_SEED) + their mean (alpha A_MEAN)."""
    xs, ys = zip(*(xy_of(n, s) for s in SEEDS))
    for x, y in zip(xs, ys):
        ax.plot(x, y, color=COL[n], alpha=A_SEED, lw=lw)
    grid = xs[0]                        # eval epochs align across seeds
    mean = np.mean([np.interp(grid, x, y) for x, y in zip(xs, ys)], axis=0)
    ax.plot(grid, mean, color=COL[n], alpha=A_MEAN, lw=1.8,
            label=f"n={n}", zorder=3)


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
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, ns):
        if baseline_ref:                # n=0 mean as reference on every panel
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
CAP_ACC = ("Test accuracy vs epoch by injected-pair count n (4 seeds + mean). "
           "n>=3 groks ~400 ep vs ~10k for baseline; n=1-2 is a 'valley of "
           "death' where several seeds never grok.")
acc = lambda n, s: hist(n, s, "test_acc")
overlay(acc, "test accuracy", "acc", ylim=(-0.02, 1.02),
        title="Exp01 — grokking vs n injected pairs (4 seeds + mean)",
        cap=CAP_ACC)
subplots(acc, "test accuracy", "acc", ylim=(-0.02, 1.02),
         title="Exp01 — test accuracy per n (4 seeds; gray dashed = n=0 mean)",
         cap=CAP_ACC)

# %% main hypothesis — time-to-grok summary
fig, ax = plt.subplots(figsize=(7, 4.6))
rng = np.random.default_rng(0)
CAP = 30_000                            # runs that never grokked, censored
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
                   label="never grokked (≥30k)" if i == 1 else None)
    ax.scatter([i], [np.mean([v if v is not None else CAP for v in g])],
               color=COL[n], marker="_", s=600, lw=2.5)
ax.set(xticks=range(len(NS)), xticklabels=NS, xlabel="n injected pairs",
       ylabel="grok epoch (test acc ≥ 0.99)", yscale="log",
       title="Exp01 — time to grok (dots = seeds, dash = mean, censored at 30k)")
ax.legend(fontsize=8, loc="center right")
caption(fig, "Epochs to grok (test acc >=0.99) vs n; dots=seeds, dash=mean, "
             "triangles=never-grokked (censored at 30k). Sharp completeness "
             "threshold at n=3.")
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
              "Rapid uptake to ~60-95% within the first 2k epochs — the "
              "embedding adopts the oracle's frequencies.")
overlay_inj(up, "W_E power fraction at injected freqs", "uptake",
            "Exp01 — embedding uptake of injected freqs (4 seeds + mean)",
            cap=CAP_UPTAKE)
subplots(up, "W_E power fraction at injected freqs", "uptake", logx=False,
         title="Exp01 — embedding uptake per n", baseline_ref=False,
         cap=CAP_UPTAKE)
overlay_inj(key, "fraction of injected freqs in key freqs", "injected_in_key",
            "Exp01 — injected freqs adopted as key freqs", ylim=(-0.05, 1.05),
            cap="Fraction of injected freqs retained in the model's key "
                "(working) frequencies. ~100% at small n, dropping to ~62% at "
                "n=8 (over-complete basis is pruned).")

# %% sanity — ablation: model.inject=False at inference (whole oracle off;
# all-or-nothing — learned weights keep their uptake, the live signal is cut)
abl_delta = lambda n, s: snap(n, s, lambda sn: sn.get("ablation_test",
                                                      {}).get("delta"))
abl_acc = lambda n, s: snap(n, s, lambda sn: sn.get("ablation_test",
                                                    {}).get("acc_off"))
CAP_ABL_ACC = ("Test accuracy with the live oracle switched off at inference. "
               "n=3 collapses (oracle-dependent); n=8 stays high (internalized) "
               "— independence grows with n.")
CAP_ABL_DELTA = ("CE increase when the oracle is switched off (causal "
                 "dependence on the live signal) vs epoch, by n.")
overlay_inj(abl_delta, "CE(oracle off) − CE(oracle on)", "ablation_delta",
            "Exp01 sanity — CE jump with oracle injection disabled at inference",
            cap=CAP_ABL_DELTA)
overlay_inj(abl_acc, "test accuracy, oracle injection off",
            "ablation_acc", "Exp01 sanity — accuracy without the live oracle "
            "signal (low = model leans on it at inference)", ylim=(-0.02, 1.02),
            cap=CAP_ABL_ACC)
subplots(abl_delta, "CE(oracle off) − CE(oracle on)", "ablation_delta",
         logx=False, baseline_ref=False, ns=NS_INJ,
         title="Exp01 sanity — oracle-off CE delta per n (4 seeds)",
         cap=CAP_ABL_DELTA)
subplots(abl_acc, "test accuracy, oracle injection off",
         "ablation_acc", logx=False, ylim=(-0.02, 1.02), baseline_ref=False,
         ns=NS_INJ,
         title="Exp01 sanity — accuracy with oracle off, per n (4 seeds)",
         cap=CAP_ABL_ACC)

# final-snapshot ablation summary vs n
fig, ax = plt.subplots(figsize=(7, 4.6))
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
       title="Exp01 sanity — final accuracy, oracle on vs off (whole mechanism)")
ax.legend()
caption(fig, "Final accuracy, oracle on vs off, per n. The on-off gap shrinks "
             "as n grows: redundant bases (n=8) become ablation-proof; minimal "
             "bases (n=3) stay dependent.")
fig.savefig(FIG / "ablation_final.png", bbox_inches="tight")
plt.close(fig)

# %% sanity — excluded loss at injected freqs + W_E spectral concentration
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
            yscale="log",
            title="excluded loss: solution carried by injected freqs")
axes[1].set(xlabel="epoch", ylabel="W_E spectral Gini",
            title="embedding spectral concentration")
axes[1].legend(ncol=2, fontsize=8, title="n", title_fontsize=8)
fig.tight_layout()
caption(fig, "Sanity: per-injected-freq excluded loss (necessity) and W_E "
             "spectral Gini (concentration) over training, by n.")
fig.savefig(FIG / "sanity_excluded_gini.png", bbox_inches="tight")
plt.close(fig)

# %% mechanistic — trig (sufficiency) vs excluded (necessity) loss
# trig_loss_injected: loss from ONLY the injected freqs' logit components —
# low means they alone reproduce the model. excluded_loss_injected: train loss
# after deleting each injected freq (one at a time; mean plotted) — high means
# the model needs them. Low trig + high excluded = solution lives in the
# injected freqs.
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
axs[1].set(yscale="log", xlabel="epoch",
           ylabel="mean excluded loss (injected)",
           title="necessity: train loss with an injected freq deleted")
axs[1].legend(ncol=2, fontsize=8, title="n", title_fontsize=8)
fig.suptitle("Exp01 — injected freqs: sufficiency vs necessity (4 seeds + mean)")
fig.tight_layout()
CAP_TRIG = ("Sufficiency (trig loss: injected freqs alone) vs necessity "
            "(excluded loss: delete an injected freq) per n. Sufficiency rises "
            "and per-freq necessity falls as n grows.")
caption(fig, CAP_TRIG)
fig.savefig(FIG / "trig_excluded_overlay.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=True)
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
axes.flat[0].legend(fontsize=8, loc="lower left")
fig.supxlabel("epoch")
fig.supylabel("loss (log)")
fig.suptitle("Exp01 — trig vs excluded loss per n (4 seeds; "
             "low solid + high dashed = solution lives in injected freqs)")
fig.tight_layout()
caption(fig, CAP_TRIG)
fig.savefig(FIG / "trig_excluded_subplots.png", bbox_inches="tight")
plt.close(fig)

# %% mechanistic — final W_E Fourier spectrum (legacy 02_MAIN_WE_spectrum redux)
# One panel per regime: baseline / below completeness threshold / minimal
# complete / over-complete. All 4 seeds per panel; green dashed = injected.
# Below threshold (n=2) the model keeps the injected freqs but RECRUITS extra
# seed-dependent ones (the legacy plot's unexplained peak); at n>=3 the
# injected set is the whole circuit.
SPEC_NS = [n for n in (0, 2, 3, 5, 6, 8) if n in NS]
fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
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
fig.supxlabel("frequency")
fig.supylabel("final W_E Fourier power")
fig.suptitle("Exp01 mechanistic — W_E spectrum, 4 seeds per panel "
             "(green dashed = injected, red ▼ = non-injected key freqs)")
fig.tight_layout()
caption(fig, "Final W_E Fourier spectrum per regime (4 seeds; green "
             "dashed=injected, red v=recruited non-injected key freqs). Below "
             "threshold (n=2) the model recruits extra freqs; n>=3 uses exactly "
             "the injected set.")
fig.savefig(FIG / "we_spectrum.png", bbox_inches="tight")
plt.close(fig)

print(f"wrote {len(list(FIG.glob('*.png')))} figures -> {FIG}")
