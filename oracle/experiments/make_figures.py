# %% [markdown]
# # make_figures — per-experiment figures, aggregated over seeds
# Every experiment is a grid (sweep axes × 4 seeds). Figures show seed-mean
# curves with ±std bands and errorbar summaries per axis value. For each
# experiment: MAIN (hypothesis) + VALIDATION (sanity) + USE (was the injected
# feature actually used — ablation through loss, and mechanistically via W_E
# Fourier power / logit coeffs / key-freqs).
# Reads results/expNN/*.result.json (axes from the embedded run spec);
# writes results/figures/<exp>/<name>.png. Skips whatever isn't on disk.

# %% imports + path bootstrap
import sys
from pathlib import Path
try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modular_addition.oracle import sweep

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 11, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "legend.fontsize": 9,
                     "savefig.dpi": 130})
RES = str(sweep.RESULTS_DIR)
FIG = f"{RES}/figures"


# --------------------------------------------------------------------------- #
# Loading / grouping
# --------------------------------------------------------------------------- #
def load_results(exp):
    """{label: result} for every result.json in results/<exp>/.

    Results from the pre-sweep era (no embedded run spec → no axes) are
    skipped: the figures aggregate over spec axes and can't place them.
    """
    out = {}
    for p in sorted(Path(RES, exp).glob("*.result.json")):
        r = json.load(open(p))
        if (r.get("spec") or {}).get("axes"):
            out[r["label"]] = r
    return out


def axes_of(res):
    return (res.get("spec") or {}).get("axes") or {}


def group_by(results, *keys):
    """{(axis values...): [results]} grouped by spec axes, seed-sorted."""
    groups = {}
    for r in results:
        ax = axes_of(r)
        groups.setdefault(tuple(ax.get(k) for k in keys), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: axes_of(r).get("seed", 0))
    return dict(sorted(groups.items(), key=lambda kv: str(kv[0])))


def history_stack(runs, key):
    """(epochs, matrix[seed, epoch]) — truncated to the shortest history."""
    series = [([h["epoch"] for h in r["history"]],
               [h[key] for h in r["history"]]) for r in runs]
    m = min(len(e) for e, _ in series)
    return np.asarray(series[0][0][:m]), np.asarray([v[:m] for _, v in series])


def snap_stack(runs, key, reduce=None):
    """(epochs, matrix[seed, snap]) for a snapshot field; `reduce` maps the
    per-snapshot value (e.g. a per-freq list) to a scalar."""
    series = []
    for r in runs:
        e, v = [], []
        for s in r.get("snapshots", []):
            val = s.get(key)
            if val is None:
                continue
            e.append(s["epoch"])
            v.append(reduce(val) if reduce else val)
        series.append((e, v))
    m = min((len(e) for e, _ in series), default=0)
    if m == 0:
        return np.array([]), np.zeros((0, 0))
    return np.asarray(series[0][0][:m]), np.asarray([v[:m] for _, v in series])


def band(ax, epochs, mat, color, label, ls="-"):
    """Seed-mean curve with ±std band."""
    if len(epochs) == 0 or mat.size == 0:
        return
    mu, sd = mat.mean(0), mat.std(0)
    ax.plot(epochs, mu, ls, color=color, label=label, lw=1.6)
    ax.fill_between(epochs, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)


def errbar_by(ax, agg, key, color="tab:blue", label=None, scale=1.0):
    """Errorbar of mean±std vs the (single) group axis from sweep.mean_std."""
    xs, mus, sds = [], [], []
    for (x,), a in sorted(agg.items()):
        st = a.get(key)
        if st:
            xs.append(x); mus.append(st["mean"] * scale); sds.append(st["std"] * scale)
    if xs:
        ax.errorbar(xs, mus, yerr=sds, fmt="o-", color=color, capsize=3, label=label)


def colors_for(values, cmap="viridis"):
    vals = sorted(set(values))
    cm = plt.get_cmap(cmap)
    pts = np.linspace(0.05, 0.9, max(len(vals), 2))
    return {v: cm(p) for v, p in zip(vals, pts)}


def save(fig, group, name, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    d = f"{FIG}/{group}"
    os.makedirs(d, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{d}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  {group}/{name}.png")


def records_and_agg(results, keys, group_keys):
    recs = [sweep.final_record(r) for r in results]
    return recs, sweep.mean_std(recs, keys=keys,
                                group_keys=[f"ax_{k}" for k in group_keys])


# --------------------------------------------------------------------------- #
# Global — reproducibility + grokking-testbed sanity (exp01 baselines)
# --------------------------------------------------------------------------- #
def fig_global():
    d = load_results("exp01")
    base = [r for r in d.values() if axes_of(r).get("n") == 0]
    if not base:
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for r in base:
        e = [h["epoch"] for h in r["history"]]
        v = [h["test_acc"] for h in r["history"]]
        ax[0].plot(e, v, alpha=.7, label=f"seed {axes_of(r).get('seed')}")
    ax[0].axhline(.99, ls=":", c="k", lw=.8)
    ax[0].set(title="4 baseline seeds (n=0): grokking testbed sanity",
              xlabel="epoch", ylabel="test acc"); ax[0].legend()
    r = base[0]
    e = [h["epoch"] for h in r["history"]]
    ax[1].semilogy(e, [h["train_loss"] for h in r["history"]], label="train loss")
    ax[1].semilogy(e, [h["test_loss"] for h in r["history"]], label="test loss")
    if r["grok_epoch"]:
        ax[1].axvline(r["grok_epoch"], c="purple", ls=":",
                      label=f"grok @ {r['grok_epoch']}")
    ax[1].set(title="seed 0 baseline: train→0 fast, test generalizes late",
              xlabel="epoch", ylabel="loss (log)"); ax[1].legend()
    save(fig, "00_global", "repro_and_grokking",
         "GLOBAL sanity — baseline grokking across seeds")


# --------------------------------------------------------------------------- #
# Uptake-vs-n figure set — shared by exp01 (p=113) and exp06 (large p/d_model)
# --------------------------------------------------------------------------- #
def fig_exp01():
    _uptake_figs("exp01", "Exp01")


def fig_exp06():
    _uptake_figs("exp06", "Exp06 (large p, d_model)")


def _uptake_figs(exp, tp):
    d = load_results(exp)
    if not d:
        return
    results = list(d.values())
    byn = group_by(results, "n")
    cols = colors_for([n for (n,) in byn])

    # MAIN — training curves, seed bands per n
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for (n,), runs in byn.items():
        e, m = history_stack(runs, "test_acc")
        band(ax[0], e, m, cols[n], f"n={n}")
        e, m = history_stack(runs, "test_loss")
        band(ax[1], e, m, cols[n], f"n={n}")
    ax[0].axhline(.99, ls=":", c="k", lw=.8)
    ax[0].set(title="test accuracy (mean±std over seeds)", xlabel="epoch")
    ax[0].legend(ncols=2)
    ax[1].set_yscale("log")
    ax[1].set(title="test loss", xlabel="epoch"); ax[1].legend(ncols=2)
    save(fig, exp, "01_MAIN_training_vs_n",
         f"{tp} MAIN — training curves by #injected frequency pairs")

    # MAIN — headline metrics vs n
    recs, agg = records_and_agg(
        results, ["grok_epoch", "final_test_acc", "ablation_delta",
                  "we_power_injected", "injected_in_key", "n_key_freqs"], ["n"])
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    errbar_by(ax[0], agg, "grok_epoch", "tab:purple")
    for (n,), a in sorted(agg.items()):   # mark configs that never grok
        st = a["grok_epoch"]
        frac = (st["n"] if st else 0) / a["_n_runs"]
        if frac < 1:
            ax[0].annotate(f"{frac:.0%} grok", (n, (st["mean"] if st else 28000)),
                           textcoords="offset points", xytext=(0, 12),
                           ha="center", fontsize=8, color="tab:red")
    ax[0].set(title="grok epoch vs n", xlabel="#injected pairs", ylabel="epoch")
    errbar_by(ax[1], agg, "final_test_acc", "tab:red")
    ax[1].axhline(.99, ls=":", c="k", lw=.8)
    ax[1].set(title="final test acc vs n", xlabel="#injected pairs")
    errbar_by(ax[2], agg, "n_key_freqs", "tab:gray", label="#key freqs")
    errbar_by(ax[2], agg, "injected_in_key", "tab:green", label="injected ∈ key")
    ax[2].plot(sorted(n for (n,) in agg), sorted(n for (n,) in agg),
               ls=":", c="green", lw=1, label="all injected adopted")
    ax[2].set(title="neuron adoption vs n", xlabel="#injected pairs")
    ax[2].legend()
    save(fig, exp, "02_MAIN_headline_vs_n",
         f"{tp} MAIN — grok speed, accuracy, adoption vs #injected pairs")

    # USE — causal + embedding investment vs n
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    errbar_by(ax[0], agg, "ablation_delta", "tab:purple")
    ax[0].axhline(0, c="k", lw=.8)
    ax[0].set(title="ablation ΔCE vs n (>0 = oracle load-bearing)",
              xlabel="#injected pairs", ylabel="ΔCE")
    errbar_by(ax[1], agg, "we_power_injected", "tab:red")
    ax[1].set(title="W_E power @ injected vs n (trainable investment)",
              xlabel="#injected pairs", ylabel="power")
    save(fig, exp, "03_USE_vs_n",
         f"{tp} USE — injected features are load-bearing and amplified in W_E")

    # USE — mechanistic dynamics (seed-mean trajectories per n)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for (n,), runs in byn.items():
        if n == 0:
            continue
        e, m = snap_stack(runs, "logit_coeff_injected", reduce=sum)
        band(ax[0], e, m, cols[n], f"n={n}")
        e, m = snap_stack(runs, "excluded_loss_injected", reduce=sum)
        band(ax[1], e, m, cols[n], f"n={n}")
        e, m = snap_stack(runs, "injected_in_key_freqs", reduce=len)
        band(ax[2], e, m, cols[n], f"n={n}")
    ax[0].set(title="Σ logit coeff @ injected", xlabel="epoch")
    ax[1].set(title="Σ excluded loss @ injected (necessity ↑)", xlabel="epoch")
    ax[2].set(title="#injected freqs adopted by neurons", xlabel="epoch")
    for a in ax:
        a.legend(ncols=2)
    save(fig, exp, "04_USE_dynamics",
         f"{tp} USE (mechanistic) — readout, necessity, adoption over training")

    # VALIDATION — final W_E spectra (seed 0 of each n)
    ns = [n for (n,) in byn if n > 0]
    fig, ax = plt.subplots(1, len(ns), figsize=(3.2 * len(ns) + 2, 4),
                           sharey=True, squeeze=False)
    for a, n in zip(ax[0], ns):
        r = byn[(n,)][0]
        spec_v = r["snapshots"][-1].get("we_freq_power_full")
        if spec_v is None:
            continue
        fr = np.arange(1, len(spec_v) + 1)
        a.plot(fr, spec_v, c=cols[n])
        for f in (r.get("injected_freqs") or []):
            a.axvline(f, ls="--", c="green", lw=.8)
        a.set(title=f"n={n}", xlabel="freq")
    ax[0][0].set_ylabel("W_E power")
    save(fig, exp, "05_VAL_WE_spectra",
         f"{tp} VALIDATION — W_E concentrates on injected freqs (seed 0, green=injected)")


# --------------------------------------------------------------------------- #
# Exp02.1 — delayed injection (T=0 reference = exp01 grid)
# --------------------------------------------------------------------------- #
def fig_exp02_1():
    d = load_results("exp02_1")
    if not d:
        return
    results = list(d.values())
    # exp01 (same freqs/amp, inject from epoch 0) provides the T=0 row
    t0 = [r for r in load_results("exp01").values() if axes_of(r).get("n", 0) > 0]
    delays = sorted({axes_of(r).get("delay") for r in results})

    # MAIN — acc curves per T (n=2 canonical) + W_E power @ injected over time
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    dcols = colors_for(delays + [0], "plasma")
    for T in [0] + delays:
        runs = (t0 if T == 0 else
                [r for r in results if axes_of(r).get("delay") == T])
        runs = [r for r in runs if axes_of(r).get("n") == 2]
        if not runs:
            continue
        e, m = history_stack(runs, "test_acc")
        band(ax[0], e, m, dcols[T], f"inject@{T}")
        e, m = snap_stack(runs, "we_freq_power_injected", reduce=sum)
        band(ax[1], e, m, dcols[T], f"inject@{T}")
        if T:
            ax[0].axvline(T, c=dcols[T], ls=":", lw=1)
            ax[1].axvline(T, c=dcols[T], ls=":", lw=1)
    ax[0].axhline(.99, ls=":", c="k", lw=.8)
    ax[0].set(title="test acc (n=2; dotted = injection on)", xlabel="epoch")
    ax[0].legend()
    ax[1].set(title="W_E power @ injected (n=2)", xlabel="epoch")
    ax[1].legend()
    save(fig, "exp02_1", "01_MAIN_delayed",
         "Exp02.1 MAIN — delayed injection (mean±std over seeds)")

    # USE — final adoption vs n for each T (incl. T=0 from exp01)
    recs_T = {0: [sweep.final_record(r) for r in t0]}
    for T in delays:
        recs_T[T] = [sweep.final_record(r) for r in results
                     if axes_of(r).get("delay") == T]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    panels = [("we_power_injected", "W_E power @ injected (final)", 0),
              ("ablation_delta", "ablation ΔCE (final)", 1),
              ("injected_in_key", "#injected adopted by neurons", 2)]
    for key, title, i in panels:
        for T, recs in recs_T.items():
            agg = sweep.mean_std(recs, keys=[key], group_keys=["ax_n"])
            errbar_by(ax[i], agg, key, dcols[T], label=f"inject@{T}")
        ax[i].set(title=title, xlabel="#injected pairs")
        ax[i].legend()
    ax[1].axhline(0, c="k", lw=.8)
    save(fig, "exp02_1", "02_USE_adoption_vs_T",
         "Exp02.1 USE — late injection is ignored regardless of n")


# --------------------------------------------------------------------------- #
# Exp02.2 — amplitude sweep (laziness)
# --------------------------------------------------------------------------- #
def fig_exp02_2():
    d = load_results("exp02_2")
    if not d:
        return
    results = list(d.values())
    recs, agg = records_and_agg(
        results, ["grok_epoch", "final_test_acc", "ablation_delta",
                  "we_power_injected", "we_total_norm", "we_gini"],
        ["amp", "n"])
    amps = sorted({k[0] for k in agg})
    ns = sorted({k[1] for k in agg})
    cols = colors_for(ns)

    def lines(ax, key, fmt="o-"):
        for n in ns:
            xs, mus, sds = [], [], []
            for amp in amps:
                st = (agg.get((amp, n)) or {}).get(key)
                if st:
                    xs.append(amp); mus.append(st["mean"]); sds.append(st["std"])
            ax.errorbar(xs, mus, yerr=sds, fmt=fmt, color=cols[n], capsize=3,
                        label=f"n={n}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("oracle amplitude")
        ax.legend(ncols=2)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    lines(ax[0], "we_total_norm")
    ax[0].set_title("|W_E| vs amp (laziness: louder oracle → smaller W_E?)")
    lines(ax[1], "we_power_injected")
    ax[1].set_title("W_E power @ injected vs amp")
    lines(ax[2], "final_test_acc")
    ax[2].axhline(.99, ls=":", c="k", lw=.8)
    ax[2].set_title("final test acc vs amp")
    save(fig, "exp02_2", "01_MAIN_laziness",
         "Exp02.2 MAIN — embedding offload and performance vs oracle amplitude")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    lines(ax[0], "grok_epoch")
    ax[0].set_title("grok epoch vs amp")
    lines(ax[1], "ablation_delta")
    ax[1].axhline(0, c="k", lw=.8)
    ax[1].set_title("ablation ΔCE vs amp (causal dependence)")
    save(fig, "exp02_2", "02_USE_speed_and_dependence", None)


# --------------------------------------------------------------------------- #
# Exp03 — grok speed vs n (derived from exp01)
# --------------------------------------------------------------------------- #
def fig_exp03():
    d = load_results("exp01")
    if not d:
        return
    recs, agg = records_and_agg(list(d.values()),
                                ["grok_epoch", "final_test_acc"], ["n"])
    base = agg.get((0,), {}).get("grok_epoch")
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    errbar_by(ax[0], agg, "grok_epoch", "tab:purple")
    if base:
        ax[0].axhline(base["mean"], ls=":", c="k", lw=.8,
                      label=f"baseline {base['mean']:.0f}")
        ax[0].legend()
    ax[0].set(title="grok epoch vs #injected pairs (mean±std)",
              xlabel="#injected pairs", ylabel="epoch")
    xs, fracs = [], []
    for (n,), a in sorted(agg.items()):
        st = a["grok_epoch"]
        xs.append(n); fracs.append((st["n"] if st else 0) / a["_n_runs"])
    ax[1].bar(xs, fracs, color="tab:blue")
    ax[1].set(title="fraction of seeds that grok (acc ≥ .99 within 30k)",
              xlabel="#injected pairs", ylabel="fraction", ylim=(0, 1.05))
    save(fig, "exp03", "01_MAIN_speed",
         "Exp03 MAIN — grok speed and reliability vs #injected pairs (from exp01 grid)")


# --------------------------------------------------------------------------- #
# Exp04 — reliability × n
# --------------------------------------------------------------------------- #
def fig_exp04():
    d = load_results("exp04")
    if not d:
        return
    results = list(d.values())
    recs, agg = records_and_agg(
        results, ["final_test_acc", "ablation_delta", "we_power_injected",
                  "grok_epoch"], ["rel", "n"])
    rels = sorted({k[0] for k in agg}, reverse=True)
    ns = sorted({k[1] for k in agg})
    cols = colors_for(ns)

    # MAIN — heatmap of final test acc (rel × n)
    M = np.full((len(rels), len(ns)), np.nan)
    for (rel, n), a in agg.items():
        st = a["final_test_acc"]
        if st:
            M[rels.index(rel), ns.index(n)] = st["mean"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_yticks(range(len(rels)), [f"{r:g}" for r in rels])
    ax.set(xlabel="#injected pairs", ylabel="reliability")
    for i in range(len(rels)):
        for j in range(len(ns)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=9)
    fig.colorbar(im, label="final test acc (seed mean)")
    save(fig, "exp04", "01_MAIN_acc_heatmap",
         "Exp04 MAIN — final test acc vs (reliability × #pairs)")

    # USE — ablation ΔCE vs reliability, per n
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for n in ns:
        xs, mus, sds = [], [], []
        for rel in rels:
            st = (agg.get((rel, n)) or {}).get("ablation_delta")
            if st:
                xs.append(rel); mus.append(st["mean"]); sds.append(st["std"])
        ax[0].errorbar(xs, mus, yerr=sds, fmt="o-", color=cols[n], capsize=3,
                       label=f"n={n}")
        xs, mus, sds = [], [], []
        for rel in rels:
            st = (agg.get((rel, n)) or {}).get("we_power_injected")
            if st:
                xs.append(rel); mus.append(st["mean"]); sds.append(st["std"])
        ax[1].errorbar(xs, mus, yerr=sds, fmt="s--", color=cols[n], capsize=3,
                       label=f"n={n}")
    for a in ax:
        a.invert_xaxis(); a.legend(ncols=2)
    ax[0].axhline(0, c="k", lw=.8)
    ax[0].set(title="ablation ΔCE (>0 used · <0 harmful)", xlabel="reliability",
              ylabel="ΔCE")
    ax[1].set(title="W_E power @ base freqs", xlabel="reliability",
              ylabel="power")
    save(fig, "exp04", "02_USE_vs_reliability",
         "Exp04 USE — unreliable features flip from used to harmful")

    # VALIDATION — training curves at canonical n=2
    fig, ax = plt.subplots(figsize=(10, 5))
    rcols = colors_for(rels, "plasma")
    for rel in rels:
        runs = [r for r in results
                if axes_of(r).get("rel") == rel and axes_of(r).get("n") == 2]
        if runs:
            e, m = history_stack(runs, "test_acc")
            band(ax, e, m, rcols[rel], f"rel={rel:g}")
    ax.set(title="test acc by reliability (n=2, mean±std over seeds)",
           xlabel="epoch", ylabel="test acc")
    ax.legend()
    save(fig, "exp04", "03_VAL_curves_n2", None)


# --------------------------------------------------------------------------- #
# Exp05 — answer hints
# --------------------------------------------------------------------------- #
def fig_exp05():
    d = load_results("exp05")
    if not d:
        return
    results = list(d.values())
    byh = group_by(results, "hint")
    recs, agg = records_and_agg(
        results, ["grok_epoch", "final_test_acc", "n_key_freqs",
                  "ablation_delta"], ["hint"])
    names = [k[0] for k in agg]
    cols = colors_for(names, "tab10")

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    panels = [("grok_epoch", "grok epoch", 0),
              ("n_key_freqs", "#key freqs (fewer = simpler circuit?)", 1),
              ("ablation_delta", "ablation ΔCE (hint load-bearing?)", 2)]
    for key, title, i in panels:
        xs, mus, sds = [], [], []
        for name in names:
            st = agg[(name,)][key]
            xs.append(name)
            mus.append(st["mean"] if st else np.nan)
            sds.append(st["std"] if st else 0)
        ax[i].bar(range(len(xs)), mus, yerr=sds, capsize=4,
                  color=[cols[x] for x in xs])
        ax[i].set_xticks(range(len(xs)), xs, rotation=20, ha="right")
        ax[i].set_title(title)
    ax[2].axhline(0, c="k", lw=.8)
    save(fig, "exp05", "01_MAIN_hints",
         "Exp05 MAIN — weak answer hints: speed, circuit size, causal use (mean±std)")

    fig, ax = plt.subplots(figsize=(10, 5))
    for (name,), runs in byh.items():
        e, m = history_stack(runs, "test_acc")
        band(ax, e, m, cols[name], name)
    ax.axhline(.99, ls=":", c="k", lw=.8)
    ax.set(title="test acc by hint config (mean±std over seeds)",
           xlabel="epoch", ylabel="test acc")
    ax.legend()
    save(fig, "exp05", "02_VAL_curves", None)


# --------------------------------------------------------------------------- #
def main():
    figs = [f for n, f in sorted(globals().items()) if n.startswith("fig_")]
    for f in figs:
        try:
            f()
        except Exception as e:  # noqa: BLE001 — keep rendering the rest
            print(f"  !! {f.__name__} failed: {e}")


# %% render everything available
if __name__ == "__main__" or "ipykernel" in sys.modules:
    main()
    print("done")
