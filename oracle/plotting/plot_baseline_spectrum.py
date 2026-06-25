"""Informal one-off: W_E and W_L frequency spectra for the NO-ORACLE baseline.

The baseline is exp01's n0 cell (axes freqs=[], no oracle frequencies injected).
exp01 only stored the W_E spectrum in its snapshots, so the W_L (neuron-logit-map)
spectrum is recomputed here from the final checkpoints. Both spectra are taken
from the SAME ep030000 checkpoint so they describe one model state.

This is deliberately NOT part of the figure suite -- just a clean reference of
what the embedding/readout frequency usage looks like with no oracle present.
No injection markers (there are no oracle freqs to mark).

    .venv/bin/python modular_addition/oracle/experiments/plot_baseline_spectrum.py [RESULTS_DIR]
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch as t
import matplotlib.pyplot as plt

from modular_addition import transformer
from modular_addition.oracle import analysis

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_common as pc  # noqa: E402

_HERE = Path(__file__).resolve().parent
DEFAULT_RES = _HERE.parents[0] / "results" / "run_20260612_200000"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RES
CKPT_DIR = RES / "exp01" / "checkpoints"
OUT = RES / "figures" / "exp01"
CKPT_NAME = "ep030000.pth"


def _amp(freq_power, p):
    """Per-frequency amplitude = sqrt(power / p), same convention as exp02_2."""
    return pc.amp_spectrum(freq_power, p)          # shared convention helper


def _spectra_for_ckpt(path):
    """(we_amp, wl_amp, p) for one baseline checkpoint, recomputed from weights."""
    obj = t.load(path, map_location="cpu", weights_only=False)
    sd = obj["model"]
    cfg = dataclasses.replace(transformer.Config(),
                              **{**obj["config"], "device": t.device("cpu")})
    basis = transformer.make_fourier_basis(cfg)

    W_E = sd["embed.W_E"]                              # (d_model, d_vocab)
    W_out = sd["blocks.0.mlp.W_out"]                  # (d_model, d_mlp)
    W_U = sd["unembed.W_U"]                            # (d_model, d_vocab)
    # W_L = neuron-logit map (d_mlp, p); we_fourier_power is generic over (rows, p)
    # matrices, so feeding it W_L reproduces analysis.wl_fourier_power exactly.
    W_L = (W_out.t() @ W_U)[:, :cfg.p]

    we = analysis.we_fourier_power(W_E, cfg, basis)
    wl = analysis.we_fourier_power(W_L, cfg, basis)
    return _amp(we["freq_power"], cfg.p), _amp(wl["freq_power"], cfg.p), cfg.p


def main():
    pc.set_style()
    ckpts = sorted(CKPT_DIR.glob(f"n0_s*/{CKPT_NAME}"))
    if not ckpts:
        raise SystemExit(f"no baseline checkpoints at {CKPT_DIR}/n0_s*/{CKPT_NAME}")

    we_specs, wl_specs, p = [], [], None
    for c in ckpts:
        we, wl, p = _spectra_for_ckpt(c)
        we_specs.append(we)
        wl_specs.append(wl)
    print(f"loaded {len(ckpts)} no-oracle baseline seeds (n0) | p={p}")

    freqs = np.arange(1, p // 2 + 1)
    fig, (axE, axL) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    for ax, specs, color, title, unit in (
            (axE, we_specs, "#1f77b4", "W_E spectrum", "residual units"),
            (axL, wl_specs, "#d62728", "W_L spectrum", "logit units")):
        for sp in specs:                              # per-seed, faint
            ax.plot(freqs, sp, color=color, alpha=0.28, lw=0.8)
        ax.plot(freqs, np.mean(specs, 0), color=color, lw=1.8)  # mean, bold
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("frequency")
        ax.set_ylabel(f"amplitude ({unit})")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.01)

    fig.suptitle(f"No-oracle baseline (exp01 n0, p={p}) — frequency spectra "
                 f"of W_E and W_L  [{len(ckpts)} seeds: faint, mean: bold]",
                 fontsize=12.5)
    pc.save(fig, OUT / "baseline_we_wl_spectrum.png", cap=(
        "Per-frequency amplitude sqrt(power/p) of the embedding W_E (input/number "
        "tokens) and the neuron-logit map W_L = W_out^T W_U (output/answer tokens) "
        "for the no-oracle baseline. No oracle frequencies are injected, so the "
        "peaks are the frequencies the model chose on its own."))
    print(f"wrote {OUT / 'baseline_we_wl_spectrum.png'}")


if __name__ == "__main__":
    main()
