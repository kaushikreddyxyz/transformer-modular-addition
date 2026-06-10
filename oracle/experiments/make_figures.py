# %% [markdown]
# # make_figures — exhaustive per-experiment figures (slides + interpretation)
# For every experiment: MAIN (hypothesis) + VALIDATION (sanity / confound checks)
# + USE (was the injected feature actually used — through loss via ablation, and
# mechanistically via W_E Fourier spectrum / logit coeffs / key-freqs).
# Reads results/expNN/*.result.json; writes results/figures/<group>/<name>.png.

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

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "legend.fontsize": 9, "savefig.dpi": 130})
RES = f"{_root}/modular_addition/oracle/results"
FIG = f"{RES}/figures"
FR = np.arange(1, 57)  # frequencies 1..p//2


def load(exp, label):
    p = f"{RES}/{exp}/{label}.result.json"
    return json.load(open(p)) if os.path.exists(p) else None


def H(res, k):
    return [h["epoch"] for h in res["history"]], [h[k] for h in res["history"]]


def SN(res, k):
    e, v = [], []
    for s in res["snapshots"]:
        if s.get(k) is not None:
            e.append(s["epoch"]); v.append(s[k])
    return e, v


def ABL(res, sub):
    e, v = [], []
    for s in res["snapshots"]:
        a = s.get("ablation_test")
        if a and a.get(sub) is not None:
            e.append(s["epoch"]); v.append(a[sub])
    return e, v


def spec(res):
    return np.array(res["snapshots"][-1]["we_freq_power_full"])


def fabl(res, sub):
    a = res["snapshots"][-1].get("ablation_test") or {}
    return a.get(sub, np.nan)


def save(fig, group, name, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    d = f"{FIG}/{group}"; os.makedirs(d, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{d}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  {group}/{name}.png")


# %% Global — reproducibility + grokking-testbed sanity
def fig_global():
    bs = [("exp01", "baseline"), ("exp05", "baseline"), ("exp06", "nfreq0_baseline")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for (e, l), c in zip(bs, ["tab:blue", "tab:orange", "tab:green"]):
        r = load(e, l)
        if r:
            ep, v = H(r, "test_acc"); ax[0].plot(ep, v, c, alpha=.7, label=f"{e} baseline")
    ax[0].axhline(.99, ls=":", c="k", lw=.8)
    ax[0].set(title="Reproducibility: 3 independent seed-0 baselines overlap exactly",
              xlabel="epoch", ylabel="test acc"); ax[0].legend()
    b = load("exp01", "baseline")
    ep, trl = H(b, "train_loss"); _, tel = H(b, "test_loss")
    ax[1].semilogy(ep, trl, "tab:blue", label="train loss")
    ax[1].semilogy(ep, tel, "tab:red", label="test loss")
    if b["grok_epoch"]:
        ax[1].axvline(b["grok_epoch"], c="purple", ls=":", label=f"grok @ {b['grok_epoch']}")
    ax[1].set(title="Grokking testbed sanity: train→0 fast, test generalizes late",
              xlabel="epoch", ylabel="loss (log)"); ax[1].legend()
    save(fig, "00_global", "repro_and_grokking",
         "GLOBAL sanity — deterministic harness + textbook grokking baseline")


# %% Exp01 — uptake (baseline vs oracle [17,34])
def fig_exp01():
    b, o, INJ = load("exp01", "baseline"), load("exp01", "oracle_f17_34_amp1.0"), [17, 34]
    # MAIN 1 — training curves
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for r, l, c in [(b, "baseline", "tab:blue"), (o, "oracle [17,34]", "tab:red")]:
        ep, v = H(r, "test_acc"); ax[0].plot(ep, v, c, label=l)
        ep, v = H(r, "test_loss"); ax[1].semilogy(ep, v, c, label=l)
    ax[0].axhline(.99, ls=":", c="k", lw=.8); ax[0].set(title="test accuracy", xlabel="epoch"); ax[0].legend()
    ax[1].set(title="test loss (log)", xlabel="epoch"); ax[1].legend()
    save(fig, "exp01", "01_MAIN_training", "Exp01 MAIN — oracle = early boost but plateaus below full grok")
    # MAIN 2 — final W_E spectrum
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(FR, spec(b), "tab:blue", label="baseline", alpha=.8)
    ax.plot(FR, spec(o), "tab:red", label="oracle", alpha=.8)
    for f in INJ:
        ax.axvline(f, ls="--", c="green", lw=1)
    ax.set(title="final W_E Fourier power (green = injected 17,34)", xlabel="frequency", ylabel="power"); ax.legend()
    save(fig, "exp01", "02_MAIN_WE_spectrum", "Exp01 MAIN (mechanistic) — oracle concentrates W_E power at injected freqs")
    # USE — ablation through loss
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    e, on = ABL(o, "acc_on"); _, off = ABL(o, "acc_off")
    ax[0].plot(e, on, "tab:red", label="oracle on"); ax[0].plot(e, off, "tab:red", ls="--", label="oracle ablated")
    ax[0].fill_between(e, off, on, alpha=.15, color="red")
    ax[0].set(title="USE via ablation — test acc drops when oracle removed", xlabel="epoch", ylabel="test acc"); ax[0].legend()
    e, cof = ABL(o, "ce_off"); _, con = ABL(o, "ce_on")
    ax[1].plot(e, con, "tab:purple", label="CE oracle on"); ax[1].plot(e, cof, "tab:purple", ls="--", label="CE oracle ablated")
    ax[1].set(title="USE via ablation — test CE", xlabel="epoch", ylabel="CE"); ax[1].legend()
    save(fig, "exp01", "03_USE_ablation", "Exp01 USE (through loss) — injected features are load-bearing (ablation hurts)")
    # VALIDATION — mechanistic uptake over training
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    e, lc = SN(o, "logit_coeff_injected"); ax[0].plot(e, [sum(x) for x in lc], "tab:red", marker="o")
    ax[0].set(title="Σ logit coeff @ injected (readout uses them)", xlabel="epoch")
    e, xl = SN(o, "excluded_loss_injected"); ax[1].plot(e, [sum(x) for x in xl], "tab:orange", marker="o")
    ax[1].set(title="Σ excluded loss @ injected (necessity ↑)", xlabel="epoch")
    e, kf = SN(o, "key_freqs"); ax[2].plot(e, [len(k) for k in kf], "tab:gray", marker="o", label="#key freqs")
    e2, ik = SN(o, "injected_in_key_freqs"); ax[2].plot(e2, [len(k) for k in ik], "tab:green", marker="s", label="injected adopted")
    ax[2].axhline(2, ls=":", c="green"); ax[2].set(title="neurons adopt injected freqs", xlabel="epoch"); ax[2].legend()
    save(fig, "exp01", "04_USE_mechanistic", "Exp01 USE (mechanistic) — injected freqs enter neurons, readout, necessity")
    # VALIDATION — train-acc confound
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for r, l, c in [(b, "baseline", "tab:blue"), (o, "oracle", "tab:red")]:
        ep, v = H(r, "train_acc"); ax.plot(ep, v, c, label=f"{l} train")
        ep, v = H(r, "test_acc"); ax.plot(ep, v, c, ls="--", label=f"{l} test")
    ax.set(title="both memorize equally (train→1 fast); only generalization differs", xlabel="epoch", ylabel="acc"); ax.legend()
    save(fig, "exp01", "05_VAL_trainacc_confound", "Exp01 VALIDATION — difference is generalization, not memorization")


# %% Exp02 — amplitude + delayed
def fig_exp02():
    INJ = [17, 34]
    amps, av = ["amp0.5", "amp1.0", "amp2.0", "amp4.0"], [0.5, 1.0, 2.0, 4.0]
    A = [load("exp02", a) for a in amps]
    # MAIN — generalization vs amp
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(av, [r["history"][-1]["test_acc"] for r in A], "o-", c="tab:red")
    ax[0].axhline(.99, ls=":", c="k"); ax[0].set_xscale("log", base=2)
    ax[0].set(title="final test acc vs amp", xlabel="amp (log2)", ylabel="test acc")
    ax[1].plot(av, [r["grok_epoch"] or np.nan for r in A], "s-", c="tab:purple"); ax[1].set_xscale("log", base=2)
    ax[1].set(title="grok epoch vs amp (gap = never grokked)", xlabel="amp (log2)", ylabel="grok epoch")
    save(fig, "exp02", "01_MAIN_amp_generalization", "Exp02 MAIN — louder 2-freq oracle is WORSE (anchors to incomplete set)")
    # MAIN — mechanism vs amp
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].plot(av, [r["snapshots"][-1]["we_total_norm"] for r in A], "o-"); ax[0].set_xscale("log", base=2); ax[0].set(title="||W_E||", xlabel="amp")
    ax[1].plot(av, [sum(r["snapshots"][-1]["we_freq_power_injected"]) for r in A], "o-", c="tab:green"); ax[1].set_xscale("log", base=2); ax[1].set(title="W_E power @ injected", xlabel="amp")
    ax[2].plot(av, [len(r["snapshots"][-1]["key_freqs"]) for r in A], "o-", c="tab:gray"); ax[2].axhline(3, ls=":", c="r", label="~3 needed to grok"); ax[2].set_xscale("log", base=2); ax[2].set(title="# key freqs discovered", xlabel="amp"); ax[2].legend()
    save(fig, "exp02", "02_MAIN_amp_mechanism", "Exp02 MAIN (mechanism) — louder oracle → fewer freqs discovered (laziness)")
    # USE — ablation vs amp
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(av, [fabl(r, "acc_on") for r in A], "o-", label="oracle on")
    ax.plot(av, [fabl(r, "acc_off") for r in A], "s--", label="oracle ablated")
    ax.set_xscale("log", base=2); ax.set(title="gap = USE. amp0.5: no gap → oracle NOT needed at inference", xlabel="amp (log2)", ylabel="test acc"); ax.legend()
    save(fig, "exp02", "03_USE_amp_ablation", "Exp02 USE — quiet oracle (amp0.5) is internalized, not used at inference")
    # VAL — W_E spectra per amp
    fig, ax = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for axi, r, a in zip(ax, A, amps):
        axi.plot(FR, spec(r))
        for f in INJ:
            axi.axvline(f, ls="--", c="green", lw=1)
        axi.set(title=a, xlabel="freq")
    ax[0].set_ylabel("W_E power")
    save(fig, "exp02", "04_VAL_amp_WE_spectra", "Exp02 VALIDATION — louder oracle suppresses non-injected freqs in W_E")
    # MAIN — delayed
    d0, d4, d8 = load("exp02", "amp1.0"), load("exp02", "delay4000"), load("exp02", "delay8000")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for r, l, c, T in [(d0, "inject@0", "tab:red", 0), (d4, "inject@4000", "tab:orange", 4000), (d8, "inject@8000", "tab:green", 8000)]:
        ep, v = H(r, "test_acc"); ax[0].plot(ep, v, c, label=l)
        e2, wp = SN(r, "we_freq_power_injected"); ax[1].plot(e2, [sum(x) for x in wp], c, label=l)
        if T:
            ax[0].axvline(T, c=c, ls=":", lw=1); ax[1].axvline(T, c=c, ls=":", lw=1)
    ax[0].axhline(.99, ls=":", c="k"); ax[0].set(title="test acc (dotted = injection on)", xlabel="epoch"); ax[0].legend()
    ax[1].set(title="W_E power @ injected", xlabel="epoch"); ax[1].legend()
    save(fig, "exp02", "05_MAIN_delayed", "Exp02 MAIN — delayed injection")
    # USE — delayed
    fig, ax = plt.subplots(figsize=(9.5, 5))
    runs = [("inject@0", d0), ("inject@4000", d4), ("inject@8000", d8)]
    xs = np.arange(len(runs))
    ax.bar(xs - .2, [fabl(r, "acc_on") for _, r in runs], .4, label="oracle on")
    ax.bar(xs + .2, [fabl(r, "acc_off") for _, r in runs], .4, label="oracle ablated")
    for i, (_, r) in enumerate(runs):
        ax.text(i, 0.5, f"{len(r['snapshots'][-1]['injected_in_key_freqs'])}/2\ninj∈key", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([l for l, _ in runs])
    ax.set(title="delayed oracle is NOT used (no acc drop) and barely adopted", ylabel="test acc"); ax.legend()
    save(fig, "exp02", "06_USE_delayed", "Exp02 USE — late injection is ignored: model already solved it itself")


# %% Exp06 — n injected freqs
def fig_exp06():
    ns = [0, 1, 2, 3, 5, 8]
    lab = {0: "nfreq0_baseline", 1: "nfreq1", 2: "nfreq2", 3: "nfreq3", 5: "nfreq5", 8: "nfreq8"}
    R = {n: load("exp06", lab[n]) for n in ns}
    # MAIN — grok + acc vs n
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(ns, [R[n]["grok_epoch"] or np.nan for n in ns], "o-", c="tab:purple")
    ax[0].set(title="grok epoch vs #injected (gap = never)", xlabel="#injected freqs", ylabel="grok epoch")
    ax[1].plot(ns, [R[n]["history"][-1]["test_acc"] for n in ns], "o-", c="tab:red"); ax[1].axhline(.99, ls=":", c="k")
    ax[1].set(title="final test acc vs #injected", xlabel="#injected freqs", ylabel="test acc")
    save(fig, "exp06", "01_MAIN_grok_vs_n", "Exp06 MAIN — completeness threshold: ≥3 injected → grok @ ~400 vs 9800")
    # MAIN — curves
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for n in ns:
        ep, v = H(R[n], "test_acc"); ax.plot(ep, v, label=f"n={n}")
    ax.axhline(.99, ls=":", c="k"); ax.set_xlim(0, 12000); ax.set(title="test acc by #injected freqs", xlabel="epoch", ylabel="test acc"); ax.legend()
    save(fig, "exp06", "02_MAIN_testacc_curves", "Exp06 MAIN — n≥3 groks almost immediately; n≤2 stalls")
    # USE — ablation vs n
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, [fabl(R[n], "acc_on") for n in ns], "o-", label="oracle on")
    ax.plot(ns, [fabl(R[n], "acc_off") for n in ns], "s--", label="oracle ablated")
    ax.set(title="USE: gap = accuracy lost when oracle removed (n≥3 used + grok)", xlabel="#injected freqs", ylabel="test acc"); ax.legend()
    save(fig, "exp06", "03_USE_ablation_vs_n", "Exp06 USE (through loss) — n≥3 are genuinely used (ablation drops acc)")
    # VAL — key freqs vs n
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, [len(R[n]["snapshots"][-1]["key_freqs"]) for n in ns], "o-", c="tab:gray", label="#key freqs (total)")
    ax.plot(ns, ns, "k:", label="y = x (#injected)")
    ax.set(title="for n≥3 the model adopts EXACTLY the injected set", xlabel="#injected freqs", ylabel="#key freqs"); ax.legend()
    save(fig, "exp06", "04_USE_keyfreqs_vs_n", "Exp06 USE (mechanistic) — #key freqs == #injected for n≥3")
    # VAL — spectra
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for axi, n in zip(ax.flat, ns):
        axi.plot(FR, spec(R[n]))
        for f in R[n]["snapshots"][-1]["injected_freqs"]:
            axi.axvline(f, ls="--", c="green", lw=1)
        axi.set(title=f"n={n}", xlabel="freq")
    save(fig, "exp06", "05_VAL_WE_spectra", "Exp06 VALIDATION — final W_E spectrum (green=injected)")


# %% Exp04 — reliability
def fig_exp04():
    rels = [1.0, 0.75, 0.5, 0.25, 0.0]
    R = {r: load("exp04", f"rel{r}") for r in rels}
    # MAIN
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(rels, [R[r]["history"][-1]["test_acc"] for r in rels], "o-", c="tab:red"); ax[0].axhline(.99, ls=":", c="k")
    ax[0].invert_xaxis(); ax[0].set(title="final test acc vs reliability", xlabel="reliability", ylabel="test acc")
    ax[1].plot(rels, [fabl(R[r], "delta") for r in rels], "o-", c="tab:purple"); ax[1].axhline(0, c="k", lw=.8)
    ax[1].invert_xaxis(); ax[1].set(title="ablation ΔCE (>0 used · <0 harmful)", xlabel="reliability", ylabel="ΔCE")
    save(fig, "exp04", "01_MAIN_reliability", "Exp04 MAIN — below ~0.5 reliability the oracle becomes harmful noise")
    # USE / VAL — W_E base vs other
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    base = [R[r]["snapshots"][-1]["we_freq_power_full"][16] for r in rels]
    other = [max(p for k, p in enumerate(R[r]["snapshots"][-1]["we_freq_power_full"], 1) if k != 17) for r in rels]
    ax[0].plot(rels, base, "o-", label="W_E @ base freq 17"); ax[0].plot(rels, other, "s-", label="W_E @ best other")
    ax[0].invert_xaxis(); ax[0].set(title="model abandons oracle freq for its own", xlabel="reliability", ylabel="W_E power"); ax[0].legend()
    ax[1].plot(rels, [fabl(R[r], "acc_on") for r in rels], "o-", label="oracle on")
    ax[1].plot(rels, [fabl(R[r], "acc_off") for r in rels], "s--", label="oracle ablated")
    ax[1].invert_xaxis(); ax[1].set(title="USE: off>on at low rel = oracle hurts", xlabel="reliability", ylabel="test acc"); ax[1].legend()
    save(fig, "exp04", "02_USE_WE_and_ablation", "Exp04 USE — high rel used; low rel abandoned/harmful (caveat: single base freq)")
    # VAL — curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in rels:
        ep, v = H(R[r], "test_acc"); ax.plot(ep, v, label=f"rel={r}")
    ax.set(title="Exp04 VALIDATION — test acc by reliability", xlabel="epoch", ylabel="test acc"); ax.legend()
    save(fig, "exp04", "03_VAL_curves", None)


# %% Exp05 — answer hint
def fig_exp05():
    cfgs = ["baseline", "hint_mod10_onehot", "hint_div10_onehot", "hint_mod10_fourier"]
    cols = ["tab:blue", "tab:red", "tab:orange", "tab:green"]
    R = {c: load("exp05", c) for c in cfgs}
    # MAIN
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    nkf = [len(R[c]["snapshots"][-1]["key_freqs"]) for c in cfgs]
    ax[0].bar(range(len(cfgs)), nkf, color=cols); ax[0].axhline(nkf[0], ls=":", c="k", label="baseline")
    ax[0].set_xticks(range(len(cfgs))); ax[0].set_xticklabels(cfgs, rotation=20, ha="right")
    ax[0].set(title="# key freqs (hypothesis: FEWER → refuted)", ylabel="#key freqs"); ax[0].legend()
    ax[1].bar(range(len(cfgs)), [R[c]["grok_epoch"] or np.nan for c in cfgs], color=cols)
    ax[1].set_xticks(range(len(cfgs))); ax[1].set_xticklabels(cfgs, rotation=20, ha="right")
    ax[1].set(title="grok epoch (hints grok SLOWER)", ylabel="grok epoch")
    save(fig, "exp05", "01_MAIN_keyfreqs_grok", "Exp05 MAIN — weak answer hints do NOT reduce freq count; slow grokking")
    # USE — ablation
    fig, ax = plt.subplots(figsize=(9, 5))
    hints = cfgs[1:]
    xs = np.arange(len(hints))
    ax.bar(xs - .2, [fabl(R[c], "acc_on") for c in hints], .4, label="hint on")
    ax.bar(xs + .2, [fabl(R[c], "acc_off") for c in hints], .4, label="hint ablated")
    ax.set_xticks(xs); ax.set_xticklabels(hints, rotation=15, ha="right")
    ax.set(title="hints ARE used (big acc drop when ablated)", ylabel="test acc"); ax.legend()
    save(fig, "exp05", "02_USE_ablation", "Exp05 USE — the answer hint is load-bearing, but it doesn't simplify the freq circuit")
    # VAL — spectra
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c, col in zip(cfgs, cols):
        ax.plot(FR, spec(R[c]), col, label=c, alpha=.7)
    ax.set(title="Exp05 VALIDATION — hint models still build a full frequency set", xlabel="freq", ylabel="W_E power"); ax.legend()
    save(fig, "exp05", "03_VAL_WE_spectrum", None)


# %% run all
if __name__ == "__main__":
    for fn in [fig_global, fig_exp01, fig_exp02, fig_exp06, fig_exp04, fig_exp05]:
        print(fn.__name__)
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {fn.__name__} FAILED: {type(e).__name__}: {e}")
    print("\nAll figures under", FIG)
