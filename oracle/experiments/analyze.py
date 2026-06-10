# %% [markdown]
# # Analyze — read all experiment results and render figures
# Reads `results/expNN/*.result.json` and writes PNGs to `results/figures/`.
# Robust to missing/partial experiments (skips what isn't there).

# %% imports + path bootstrap
import sys
from pathlib import Path
try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = f"{_root}/modular_addition/oracle/results"
FIG = f"{RES}/figures"
os.makedirs(FIG, exist_ok=True)
INJ = [17, 34]


def load_dir(name):
    out = {}
    for p in glob.glob(f"{RES}/{name}/*.result.json"):
        r = json.load(open(p))
        out[r["label"]] = r
    return out


def H(res, key):
    return [h["epoch"] for h in res["history"]], [h[key] for h in res["history"]]


def S(res, key):
    return [s["epoch"] for s in res["snapshots"]], [s.get(key) for s in res["snapshots"]]


# %% Exp 01 — uptake (2x2)
def fig_exp01():
    d = load_dir("exp01")
    if not d:
        print("exp01: no data"); return
    b = d.get("baseline"); o = next((v for k, v in d.items() if k.startswith("oracle")), None)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for r, lab, c in [(b, "baseline", "tab:blue"), (o, "oracle [17,34]", "tab:red")]:
        if r is None:
            continue
        ax[0, 0].plot(*H(r, "test_acc"), c, label=lab)
        ax[0, 1].semilogy(*H(r, "test_loss"), c, label=lab)
    ax[0, 0].axhline(0.99, ls=":", c="k", lw=.8); ax[0, 0].set(title="test accuracy", xlabel="epoch")
    ax[0, 0].legend()
    ax[0, 1].set(title="test loss", xlabel="epoch"); ax[0, 1].legend()
    # final W_E Fourier spectrum
    for r, lab, c in [(b, "baseline", "tab:blue"), (o, "oracle", "tab:red")]:
        if r is None or not r["snapshots"]:
            continue
        power = r["snapshots"][-1]["we_freq_power_full"]
        ax[1, 0].plot(range(1, len(power) + 1), power, c, label=lab, alpha=.8)
    for f in INJ:
        ax[1, 0].axvline(f, ls="--", c="green", lw=.8)
    ax[1, 0].set(title="final W_E Fourier power (green=injected)", xlabel="frequency", ylabel="power")
    ax[1, 0].legend()
    # uptake over training (oracle)
    if o is not None and o["snapshots"]:
        ep, coeff = S(o, "logit_coeff_injected")
        ax[1, 1].plot(ep, [sum(c) for c in coeff], "tab:red", label="Σ logit coeff @injected")
        ep2, exc = S(o, "excluded_loss_injected")
        ax[1, 1].plot(ep2, [sum(e) for e in exc], "tab:orange", label="Σ excluded loss @injected")
        ax2 = ax[1, 1].twinx()
        ep3, nk = S(o, "key_freqs")
        ax2.plot(ep3, [len(k) for k in nk], "tab:gray", ls=":", label="#key_freqs")
        ax2.set_ylabel("#key_freqs")
        ax[1, 1].set(title="oracle uptake over training", xlabel="epoch"); ax[1, 1].legend(loc="center right")
    fig.tight_layout(); fig.savefig(f"{FIG}/exp01_uptake.png", dpi=110); plt.close(fig)
    print("wrote exp01_uptake.png")


# %% Exp 02 — laziness (amp sweep) + delayed injection
def fig_exp02():
    d = load_dir("exp02")
    if not d:
        print("exp02: no data"); return
    amps = sorted([k for k in d if k.startswith("amp")], key=lambda s: float(s[3:]))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # (1) we_power@injected trajectory per amp
    for k in amps:
        ep, wp = S(d[k], "we_freq_power_injected")
        ax[0].plot(ep, [sum(w) for w in wp], label=k)
    ax[0].set(title="W_E power @injected freqs vs epoch (amp sweep)", xlabel="epoch", ylabel="W_E power@inj")
    ax[0].legend()
    # (2) final ||W_E|| and we_power@inj vs amp
    av = [float(k[3:]) for k in amps]
    norm = [d[k]["snapshots"][-1]["we_total_norm"] for k in amps]
    powi = [sum(d[k]["snapshots"][-1]["we_freq_power_injected"]) for k in amps]
    ax[1].plot(av, norm, "o-", label="|W_E|"); ax[1].plot(av, powi, "s-", label="W_E power@inj")
    ax[1].set(title="laziness vs oracle amplitude", xlabel="amp"); ax[1].legend()
    # (3) delayed injection: we_power@inj vs epoch with T marked
    for k in [x for x in d if x.startswith("delay")]:
        T = int(k[5:])
        ep, wp = S(d[k], "we_freq_power_injected")
        line, = ax[2].plot(ep, [sum(w) for w in wp], label=k)
        ax[2].axvline(T, ls="--", c=line.get_color(), lw=.8)
    ax[2].set(title="delayed injection: W_E power @inj (dashed=T)", xlabel="epoch")
    ax[2].legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/exp02_laziness_delayed.png", dpi=110); plt.close(fig)
    print("wrote exp02_laziness_delayed.png")


# %% Exp 04 — reliability sweep
def fig_exp04():
    s = _load_summary("exp04")
    if not s:
        print("exp04: no summary"); return
    rels = sorted(s.values(), key=lambda r: -r["reliability"])
    x = [r["reliability"] for r in rels]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot(x, [r["ablation_delta"] for r in rels], "o-"); ax[0].set(title="ablation ΔCE vs reliability", xlabel="reliability")
    ax[1].plot(x, [r["we_power_base"] for r in rels], "o-", label="W_E pow@base_freq")
    ax[1].plot(x, [r["we_power_other_max"] for r in rels], "s-", label="W_E pow@best_other")
    ax[1].set(title="W_E: oracle freq vs self-discovered", xlabel="reliability"); ax[1].legend()
    ax[2].plot(x, [r["final_test_acc"] for r in rels], "o-", label="test_acc")
    gk = [(r["grok_epoch"] or np.nan) for r in rels]
    ax2 = ax[2].twinx(); ax2.plot(x, gk, "s--", c="tab:red", label="grok_epoch"); ax2.set_ylabel("grok_epoch")
    ax[2].set(title="generalization vs reliability", xlabel="reliability"); ax[2].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/exp04_reliability.png", dpi=110); plt.close(fig)
    print("wrote exp04_reliability.png")


# %% Exp 05 — answer hint
def fig_exp05():
    s = _load_summary("exp05")
    if not s:
        print("exp05: no summary"); return
    names = list(s.keys())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(names, [s[n]["n_key_freqs"] for n in names], color="tab:purple")
    ax[0].set(title="#key frequencies used", ylabel="#key_freqs"); ax[0].tick_params(axis="x", rotation=30)
    ax[1].bar(names, [s[n]["final_test_acc"] for n in names], color="tab:green")
    ax[1].set(title="final test accuracy"); ax[1].tick_params(axis="x", rotation=30); ax[1].axhline(0.99, ls=":", c="k")
    fig.tight_layout(); fig.savefig(f"{FIG}/exp05_answer_hint.png", dpi=110); plt.close(fig)
    print("wrote exp05_answer_hint.png")


# %% Exp 06 — n injected freqs
def fig_exp06():
    s = _load_summary("exp06")
    if not s:
        print("exp06: no summary"); return
    ks = sorted(s.keys(), key=int)
    n = [s[k]["n_inject"] for k in ks]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    gk = [(s[k]["grok_epoch"] or np.nan) for k in ks]
    ax[0].plot(n, gk, "o-"); ax[0].set(title="grok epoch vs #injected freqs", xlabel="#injected", ylabel="grok_epoch")
    ax[1].plot(n, [s[k]["final_test_acc"] for k in ks], "o-"); ax[1].axhline(0.99, ls=":", c="k")
    ax[1].set(title="final test_acc vs #injected freqs", xlabel="#injected")
    ax[2].plot(n, [s[k]["n_key_freqs"] for k in ks], "o-", label="#key_freqs total")
    ax[2].plot(n, n, "k:", label="y=x (#injected)")
    ax[2].set(title="#key_freqs vs #injected", xlabel="#injected"); ax[2].legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/exp06_nfreqs.png", dpi=110); plt.close(fig)
    print("wrote exp06_nfreqs.png")


def _load_summary(name):
    p = f"{RES}/{name}/summary.json"
    return json.load(open(p)) if os.path.exists(p) else None


# %% run all
if __name__ == "__main__":
    for fn in [fig_exp01, fig_exp02, fig_exp04, fig_exp05, fig_exp06]:
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__}: FAILED {type(e).__name__}: {e}")
    print("\nFigures in", FIG)
