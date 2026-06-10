# %% [markdown]
# # Exp 10 — mechanistic neuron-level proof of (non-)use
# Retrains 5 diagnostic configs (models weren't saved during the sweeps), then for
# each computes the dominant Fourier frequency of every MLP neuron. If the model
# *uses* the injected frequencies, its neurons cluster on them (green lines); if it
# ignores them, neurons cluster on the model's own freqs. Pairs the mechanistic
# picture with the loss-based ablation acc_off for each config.

# %% imports + path bootstrap
import sys
from pathlib import Path
try:
    _root = str(Path(__file__).resolve().parents[3])
except NameError:
    _root = "/root/oracle-encodings"
if _root not in sys.path:
    sys.path.insert(0, _root)

import dataclasses
import json
import os
import numpy as np
import torch as t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modular_addition import transformer
from modular_addition.oracle import inject, analysis, harness

plt.rcParams.update({"figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 11, "axes.titlesize": 11, "axes.titleweight": "bold", "savefig.dpi": 130})
device = t.device("cuda" if t.cuda.is_available() else "cpu")
RES = f"{_root}/modular_addition/oracle/results/exp10"
MOD = f"{RES}/models"
FIG = f"{_root}/modular_addition/oracle/results/figures/exp10"
for d in (RES, MOD, FIG):
    os.makedirs(d, exist_ok=True)
FRAC = 0.50   # a neuron counts as "specialized" to its freq if frac_explained > FRAC

# name, injected freqs (or None), amp, inject_from_epoch, num_epochs, tag
CONFIGS = [
    ("baseline",   None,                 1.0,    0, 12000, "no oracle"),
    ("nfreq2",     [17, 34],             1.0,    0, 16000, "used, no grok"),
    ("nfreq5",     [17, 34, 9, 25, 43],  1.0,    0,  8000, "used + grok"),
    ("amp0.5",     [17, 34],             0.5,    0,  8000, "steered, not needed"),
    ("delay8000",  [17, 34],             1.0, 8000, 13000, "ignored (late)"),
]

records = {}
for name, freqs, amp, t_on, epochs, tag in CONFIGS:
    cfg = dataclasses.replace(transformer.Config(), device=device, num_epochs=epochs, save_models=False)
    orc = inject.make_fourier_oracle(cfg, freqs, amp=amp) if freqs else None
    model, data = harness.setup(cfg, oracle_fn=orc)
    res = harness.train(cfg, model, data, num_epochs=epochs, eval_every=500,
                        snapshot_every=10**9, inject_from_epoch=t_on, run_dir=RES,
                        label=name, stop_after_grok=600, verbose=True)
    # mechanistic: per-neuron dominant frequency
    nf, frac = analysis.neuron_freq_histogram(model, cfg)
    spec_mask = frac > FRAC
    # loss-based: ablation
    abl = analysis.ablation_ce(model, data["test_x"], data["test_y"], cfg) if orc else None
    t.save(model.state_dict(), f"{MOD}/{name}.pt")
    records[name] = dict(
        tag=tag, injected=freqs or [], amp=amp, inject_from=t_on,
        grok_epoch=res["grok_epoch"], final_test_acc=round(res["history"][-1]["test_acc"], 4),
        n_specialized=int(spec_mask.sum()),
        neuron_freqs=nf.tolist(), neuron_frac=np.round(frac, 3).tolist(),
        acc_on=round(abl["acc_on"], 4) if abl else None,
        acc_off=round(abl["acc_off"], 4) if abl else None,
        delta_ce=round(abl["delta"], 4) if abl else None,
    )
    print(f"  -> {name}: grok={res['grok_epoch']} testacc={records[name]['final_test_acc']} "
          f"specialized_neurons={records[name]['n_specialized']} acc_off={records[name]['acc_off']}")

with open(f"{RES}/neuron_freqs.json", "w") as f:
    json.dump(records, f)

# %% figure — neuron dominant-frequency histograms
FR = np.arange(1, 57)
fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
for ax, (name, *_rest) in zip(axes.flat, CONFIGS):
    r = records[name]
    nf = np.array(r["neuron_freqs"]); frac = np.array(r["neuron_frac"])
    sel = nf[frac > FRAC]
    counts = np.array([(sel == f).sum() for f in FR])
    ax.bar(FR, counts, width=0.9, color="tab:gray")
    for f in r["injected"]:
        ax.axvline(f, ls="--", c="green", lw=1.2)
    off = f", acc on→off {r['acc_on']}→{r['acc_off']}" if r["acc_off"] is not None else ""
    ax.set_title(f"{name} ({r['tag']})\ngrok={r['grok_epoch']}, {r['n_specialized']} specialized neurons{off}", fontsize=10)
    ax.set_xlabel("neuron's dominant frequency"); ax.set_ylabel("# neurons")
axes.flat[-1].axis("off")
fig.suptitle("Exp10 — MLP neurons specialize to injected freqs (green) only when the oracle is used; "
             "ignored/baseline runs use the model's OWN freqs", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{FIG}/neuron_freq_histograms.png", bbox_inches="tight"); plt.close(fig)
print("wrote", f"{FIG}/neuron_freq_histograms.png")

# %% figure — specialization sanity (how sharply neurons pick one frequency)
fig, ax = plt.subplots(figsize=(9, 5))
for name, *_ in CONFIGS:
    fr = np.array(records[name]["neuron_frac"])
    ax.hist(fr, bins=20, histtype="step", lw=2, label=f"{name} ({records[name]['n_specialized']} > {FRAC})")
ax.axvline(FRAC, ls=":", c="k"); ax.set(title="Exp10 VALIDATION — neuron specialization (frac of variance from one freq)",
                                         xlabel="frac explained by best single freq", ylabel="# neurons")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{FIG}/neuron_specialization.png", bbox_inches="tight"); plt.close(fig)
print("wrote", f"{FIG}/neuron_specialization.png")
print("\n✅ exp10 done")
