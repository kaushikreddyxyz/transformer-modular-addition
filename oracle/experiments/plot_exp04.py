# %% [markdown]
# # Exp 04 figures — unreliable / corrupted oracle
# The oracle supplies the TRUE base frequency with prob `rel`, else a random
# frequency (per example, per injected pair). Grid: rel x n x 4 seeds, amp=1.0.
#
#   headline:  grok_success_heatmap.png   rel x n grok-rate (the threshold)
#              grok_epoch_heatmap.png      median epochs-to-grok (grokked-only)
#   does the model stop leaning on the oracle as rel drops?
#              dependence_vs_rel.png       final acc_off vs rel, by n (groked)
#              uptake_vs_rel.png           final W_E power on TRUE freqs (groked)
#              injected_in_key_vs_rel.png  TRUE freqs retained in key set (groked)
#   the failure structure (all runs, incl never-grok):
#              acc_curves_by_rel.png       test_acc vs epoch, small-multiples/rel
#
# CONFOUND noted in captions: corruption is seeded by the model seed, so the
# per-seed spread mixes init/data randomness with the corruption draw.

# %% imports + data
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc  # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else
           Path(__file__).resolve().parents[1] / "results" / "run_20260612_200000")
EXP = "exp04"
FIG = RES / "figures" / EXP
FIG.mkdir(parents=True, exist_ok=True)

pc.set_style()

runs = pc.load_exp(RES, EXP)
RELS = sorted(pc.axis_values(runs, "rel"))            # 0.0 .. 1.0
RELS_DESC = sorted(RELS, reverse=True)                # 1.0 .. 0.0 (intuitive)
NS = pc.axis_values(runs, "n")                        # [1,2,3,5,6,8]
SEEDS = pc.axis_values(runs, "seed")
COL = pc.color_map(NS, baseline=None)                 # color by n (no baseline)

n_total = len(runs)
n_grok = sum(pc.groked(r) for r in runs)
print(f"exp04: {n_total} runs, {n_grok} grokked, {n_total - n_grok} never grokked")

CONFOUND = ("Corruption draw is seeded by the model seed, so per-seed spread "
            "mixes init/data and corruption randomness.")


# %% ----------------------------------------------------------------- #
# FIG 1a — grok-success heatmap (rel rows desc, n cols), annotated
# --------------------------------------------------------------------- #
rate = np.full((len(RELS_DESC), len(NS)), np.nan)
cnt = np.zeros_like(rate, dtype=int)
for i, rel in enumerate(RELS_DESC):
    for j, n in enumerate(NS):
        sub = pc.select(runs, rel=rel, n=n)
        rate[i, j] = pc.grok_rate(sub)
        cnt[i, j] = sum(pc.groked(r) for r in sub)

fig, ax = plt.subplots(figsize=(7.4, 5.2))
im = ax.imshow(rate, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(NS)), NS)
ax.set_yticks(range(len(RELS_DESC)), [f"{r:g}" for r in RELS_DESC])
ax.set_xlabel("n injected pairs")
ax.set_ylabel("reliability  (rel)")
ax.set_title("Exp04 — grok success rate over 4 seeds")
for i in range(len(RELS_DESC)):
    for j in range(len(NS)):
        v = rate[i, j]
        ax.text(j, i, f"{v:.2f}\n{cnt[i, j]}/{len(SEEDS)}",
                ha="center", va="center", fontsize=8.5,
                color="white" if v < 0.55 else "black")
ax.set_xticks(np.arange(-.5, len(NS), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(RELS_DESC), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("grok rate (fraction of 4 seeds)")
pc.save(fig, FIG / "grok_success_heatmap.png",
        cap=("Fraction of 4 seeds reaching grok (cells annotate rate and "
             "count). Reliability falls top->bottom. Failures concentrate at "
             "LOW rel (rel<=0.25) and the n=2 column; high n does NOT rescue "
             f"rel=0.25 (all n fail by rel=0.25, n>=5). {CONFOUND}"))


# %% ----------------------------------------------------------------- #
# FIG 1b — companion: median grok-epoch (grokked-only) heatmap
# --------------------------------------------------------------------- #
med = np.full((len(RELS_DESC), len(NS)), np.nan)
for i, rel in enumerate(RELS_DESC):
    for j, n in enumerate(NS):
        ge = [pc.grok_epoch(r) for r in pc.select(runs, rel=rel, n=n)
              if pc.groked(r)]
        if ge:
            med[i, j] = float(np.median(ge))

fig, ax = plt.subplots(figsize=(7.4, 5.2))
masked = np.ma.masked_invalid(med)
cmap = plt.get_cmap("magma_r").copy()
cmap.set_bad("0.85")
im = ax.imshow(masked, cmap=cmap, aspect="auto")
ax.set_xticks(range(len(NS)), NS)
ax.set_yticks(range(len(RELS_DESC)), [f"{r:g}" for r in RELS_DESC])
ax.set_xlabel("n injected pairs")
ax.set_ylabel("reliability  (rel)")
ax.set_title("Exp04 — median epochs to grok (grokked seeds only)")
vmax = np.nanmax(med)
for i in range(len(RELS_DESC)):
    for j in range(len(NS)):
        if np.isnan(med[i, j]):
            ax.text(j, i, "—", ha="center", va="center", fontsize=11,
                    color="0.45")
        else:
            ax.text(j, i, f"{med[i, j]:.0f}", ha="center", va="center",
                    fontsize=8.8,
                    color="white" if med[i, j] > 0.55 * vmax else "black")
ax.set_xticks(np.arange(-.5, len(NS), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(RELS_DESC), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.2)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("median grok epoch")
pc.save(fig, FIG / "grok_epoch_heatmap.png",
        cap=("Median epochs to grok over grokked seeds (gray dash = no seed "
             "grokked, see success heatmap). Time-to-grok lengthens as rel "
             "drops; n>=3 groks fast at high rel but that speed does not "
             f"survive into rel=0.25. {CONFOUND}"))


# %% ----------------------------------------------------------------- #
# shared: a final-snapshot scalar vs rel, color by n, only-grokked,
# jittered seed dots + connected mean line.
# --------------------------------------------------------------------- #
def final_vs_rel(value_fn, ylabel, title, fname, cap, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    drawn = 0
    for n in NS:
        means, mrel = [], []
        for rel in RELS_DESC:
            sub = [r for r in pc.select(runs, rel=rel, n=n) if pc.groked(r)]
            vals = [value_fn(pc.final_snap(r)) for r in sub]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            xj = rel + rng.uniform(-0.012, 0.012, len(vals))
            ax.scatter(xj, vals, color=COL[n], alpha=0.45, s=30, zorder=2)
            means.append(np.mean(vals))
            mrel.append(rel)
            drawn += len(vals)
        if means:
            order = np.argsort(mrel)
            ax.plot(np.array(mrel)[order], np.array(means)[order],
                    color=COL[n], lw=1.9, marker="o", ms=5, zorder=3,
                    label=f"n={n}")
    ax.set(xlabel="reliability  (rel)  —  1.0 = always-true oracle",
           ylabel=ylabel, title=title)
    ax.set_xticks(RELS)
    ax.invert_xaxis()                     # 1.0 on the left (intuitive)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=2, fontsize=9, title="injected pairs", title_fontsize=9,
              framealpha=0.9)
    print(f"  {fname}: {drawn} grokked seed-points drawn")
    pc.save(fig, FIG / fname, cap=cap)


# %% FIG 2 — dependence on the live oracle (acc with oracle OFF) vs rel
final_vs_rel(
    lambda s: pc.ablation(s, "acc_off"),
    "final test acc, oracle OFF  (high = independent of live oracle)",
    "Exp04 — dependence on the live oracle vs reliability (grokked only)",
    "dependence_vs_rel.png",
    cap=("Final accuracy with the live oracle switched off at inference, vs "
         "rel (x inverted: reliable on left). High acc_off = the model does "
         "NOT lean on the live oracle. Only grokked runs; means are dots->line "
         f"per n. {CONFOUND}"),
    ylim=(-0.03, 1.03))


# %% FIG 3 — embedding uptake of the TRUE base freqs vs rel
final_vs_rel(
    pc.frac_we_power_injected,
    "final W_E power fraction on TRUE injected freqs",
    "Exp04 — embedding uptake of the TRUE freqs vs reliability (grokked only)",
    "uptake_vs_rel.png",
    cap=("Fraction of final W_E Fourier power sitting on the TRUE base freqs "
         "vs rel (x inverted). The injected signal is partly wrong at low rel, "
         "so lower uptake is expected. Only grokked runs. "
         f"{CONFOUND}"))


# %% FIG 4 — TRUE freqs retained in the key (working) set vs rel
final_vs_rel(
    pc.frac_injected_in_key,
    "fraction of TRUE freqs kept in key set",
    "Exp04 — TRUE freqs retained in the working set vs reliability (grokked)",
    "injected_in_key_vs_rel.png",
    cap=("Fraction of the TRUE base freqs retained among the model's key "
         "(working) frequencies vs rel (x inverted). Only grokked runs. "
         f"{CONFOUND}"),
    ylim=(-0.05, 1.05))


# %% ----------------------------------------------------------------- #
# FIG 5 — test_acc vs epoch, small-multiples by rel, ALL seeds (incl fails)
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, len(RELS_DESC), figsize=(3.3 * len(RELS_DESC), 5.0),
                         sharex=True, sharey=True)
for ax, rel in zip(axes, RELS_DESC):
    nfail = 0
    for n in NS:
        for r in pc.select(runs, rel=rel, n=n):
            ep, acc = pc.hist_series(r, "test_acc")
            m = ep > 0                      # drop epoch 0 for log-x
            ax.plot(ep[m], acc[m], color=COL[n],
                    alpha=0.75 if pc.groked(r) else 0.5,
                    lw=1.3 if pc.groked(r) else 0.9,
                    ls="-" if pc.groked(r) else (0, (2, 1.5)))
            nfail += int(not pc.groked(r))
    rate_here = 1 - nfail / (len(NS) * len(SEEDS))
    ax.set_title(f"rel = {rel:g}   (grok {rate_here:.0%})", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(left=180)
    ax.set_ylim(-0.02, 1.02)
# legend: n colors + the solid/dashed convention
handles = [plt.Line2D([], [], color=COL[n], lw=2, label=f"n={n}") for n in NS]
handles += [plt.Line2D([], [], color="0.4", lw=1.4, ls="-", label="grokked"),
            plt.Line2D([], [], color="0.4", lw=1.0, ls=(0, (2, 1.5)),
                       label="never grokked")]
axes[-1].legend(handles=handles, ncol=2, fontsize=7.5, loc="lower right",
                framealpha=0.9)
fig.supxlabel("epoch (log)")
fig.supylabel("test accuracy")
fig.suptitle("Exp04 — grokking dynamics by reliability (ALL seeds; "
             "dashed = never grokked)")
pc.save(fig, FIG / "acc_curves_by_rel.png",
        cap=("Test accuracy vs epoch, one panel per reliability, ALL 24 runs "
             "per panel (6 n x 4 seeds), color = n, dashed = never grokked. "
             "Failures pile up at LOW rel (rel<=0.25); within those panels the "
             "low-n curves are the ones stuck near chance. "
             f"{CONFOUND}"))

print(f"wrote {len(list(FIG.glob('*.png')))} figures -> {FIG}")
