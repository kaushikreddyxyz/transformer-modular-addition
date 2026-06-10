"""Uptake detectors for oracle injection.

Everything here answers one question: *did the model actually use our injected
frequencies?* We reuse the paper's progress measures (`transformer.calculate_*`)
plus a few additions (W_E Fourier spectrum, ablation ΔCE, Gini concentration).

All logit-space measures follow the repo convention: score on
``logits = model(all_data)[:, -1, :-1]`` (final position, drop the "=" class) over
the full p**2 grid.
"""
import numpy as np
import torch as t

from modular_addition import transformer, helpers


# --------------------------------------------------------------------------- #
# Shared context (compute the expensive grid/labels/masks once)
# --------------------------------------------------------------------------- #
def metric_context(config: transformer.Config, train):
    """Bundle the tensors the `transformer.calculate_*` functions need.

    `train` is the list of train (i, j, p) tuples (used to build the train/test
    boolean masks over the full p**2 grid).
    """
    p = config.p
    all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(config.device)
    labels = t.tensor([config.fn(i, j) for i, j, _ in all_data]).to(config.device)
    is_train, is_test = config.is_train_is_test(train)
    fourier_basis = transformer.make_fourier_basis(config)
    return dict(all_data=all_data, labels=labels, is_train=is_train,
                is_test=is_test, fourier_basis=fourier_basis)


def _grid_logits(model, all_data):
    """logits over the full grid at the read-out position, dropping the '=' class."""
    return model(all_data)[:, -1, :-1]


# --------------------------------------------------------------------------- #
# W_E structure: Fourier spectrum + norm + concentration (the "laziness" signals)
# --------------------------------------------------------------------------- #
def we_fourier_power(W_E, config: transformer.Config, fourier_basis=None):
    """Decompose the trainable embedding into Fourier power per frequency.

    Returns dict with:
      freqs        : [1 .. p//2]
      freq_power   : energy of W_E (over the p number-token columns) at each freq
      const_power  : energy on the constant component
      total_norm   : Frobenius norm of W_E[:, :p]
      gini         : Gini concentration of freq_power (1 => all energy on one freq)
    """
    if fourier_basis is None:
        fourier_basis = transformer.make_fourier_basis(config)
    p = config.p
    WE = W_E[:, :p].detach()                       # (d_model, p)  number tokens only
    coeffs = WE @ fourier_basis.T                  # (d_model, p_basis)
    power = coeffs.pow(2).sum(0)                    # (p_basis,)
    freqs = list(range(1, p // 2 + 1))
    freq_power = t.stack([power[2 * k - 1] + power[2 * k] for k in freqs]).cpu().numpy()
    return dict(freqs=np.array(freqs), freq_power=freq_power,
                const_power=power[0].item(), total_norm=WE.norm().item(),
                gini=gini(freq_power))


def gini(x):
    """Gini coefficient of a non-negative vector (0 = uniform, ->1 = concentrated)."""
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = x.size
    s = x.sum()
    if s == 0 or n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * s))


# --------------------------------------------------------------------------- #
# Reused progress measures, restricted to an arbitrary frequency set
# --------------------------------------------------------------------------- #
def key_freqs(model, config, ctx):
    """Frequencies the MLP neurons specialize to (transformer.calculate_key_freqs)."""
    kf = list(int(k) for k in transformer.calculate_key_freqs(config, model, ctx["all_data"]))
    model.remove_all_hooks()   # calculate_key_freqs leaves caching hooks attached
    return kf


def excluded_loss(model, config, ctx, freqs, logits=None):
    """Train loss after deleting each freq's cos/sin(x+y) component (necessity).

    Returns list aligned with `freqs`. Rising vs baseline => the model relies on
    those frequencies even to fit the training set.
    """
    if logits is None:
        logits = _grid_logits(model, ctx["all_data"])
    return transformer.calculate_excluded_loss(
        config=config, fourier_basis=ctx["fourier_basis"], key_freqs=list(freqs),
        is_train=ctx["is_train"], is_test=ctx["is_test"], labels=ctx["labels"], logits=logits)


def trig_loss(model, config, ctx, freqs, mode="all", logits=None):
    """Loss using ONLY the cos/sin(x+y) components at `freqs` (sufficiency).

    Low (≈ full loss) => those frequencies alone explain the model's behaviour.
    """
    if logits is None:
        logits = _grid_logits(model, ctx["all_data"])
    return transformer.calculate_trig_loss(
        config=config, model=model, train=None, logits=logits, key_freqs=list(freqs),
        fourier_basis=ctx["fourier_basis"], all_data=ctx["all_data"],
        is_train=ctx["is_train"], is_test=ctx["is_test"], labels=ctx["labels"], mode=mode).item()


def logit_coefficients(model, config, ctx, logits=None):
    """Per-frequency coefficient of logits on cos(w(x+y-z)) (the usage scalar).

    Returns np.array indexed by w-1 for w in [1 .. p//2].
    """
    if logits is None:
        logits = _grid_logits(model, ctx["all_data"])
    coeffs = transformer.calculate_coefficients(
        logits=logits, fourier_basis=ctx["fourier_basis"], key_freqs=None,
        p=config.p, device=config.device)
    return helpers.to_numpy(coeffs)


# --------------------------------------------------------------------------- #
# Causal ablation: zero the oracle at inference and measure the CE jump
# --------------------------------------------------------------------------- #
@t.no_grad()
def ablation_ce(model, x, y, config):
    """Cross-entropy with the oracle ON vs OFF (model.inject toggled).

    Returns dict(ce_on, ce_off, delta, acc_on, acc_off). A large positive `delta`
    on held-out data == the model causally depends on the injected feature.
    """
    was = model.inject
    model.inject = True
    ce_on, acc_on = _ce_acc(model, x, y, config)
    model.inject = False
    ce_off, acc_off = _ce_acc(model, x, y, config)
    model.inject = was
    return dict(ce_on=ce_on, ce_off=ce_off, delta=ce_off - ce_on,
                acc_on=acc_on, acc_off=acc_off)


@t.no_grad()
def _ce_acc(model, x, y, config):
    logits = model(x)[:, -1]
    ce = helpers.cross_entropy_high_precision(logits, y).item()
    acc = (logits[:, :config.p].argmax(-1) == y).float().mean().item()
    return ce, acc


# --------------------------------------------------------------------------- #
# One-call snapshot bundling all uptake metrics (used as harness snapshot_fn)
# --------------------------------------------------------------------------- #
def uptake_snapshot(model, config, ctx, injected_freqs, data=None):
    """Full uptake report at the current model state. Cheap enough for ~1k-epoch cadence."""
    logits = _grid_logits(model, ctx["all_data"])
    kf = key_freqs(model, config, ctx)
    inj = [int(f) for f in injected_freqs]
    we = we_fourier_power(model.embed.W_E, config, ctx["fourier_basis"])
    coeffs = logit_coefficients(model, config, ctx, logits=logits)
    snap = dict(
        key_freqs=kf,
        injected_freqs=inj,
        injected_in_key_freqs=[k for k in inj if k in kf],
        excluded_loss_injected=excluded_loss(model, config, ctx, inj, logits=logits) if inj else [],
        trig_loss_injected=trig_loss(model, config, ctx, inj, logits=logits) if inj else None,
        trig_loss_keyfreqs=trig_loss(model, config, ctx, kf, logits=logits) if kf else None,
        we_total_norm=we["total_norm"],
        we_gini=we["gini"],
        we_freq_power_injected=[float(we["freq_power"][k - 1]) for k in inj] if inj else [],
        we_freq_power_top=_top_freqs(we["freq_power"], 6),
        we_freq_power_full=we["freq_power"].tolist(),
        logit_coeff_injected=[float(coeffs[k - 1]) for k in inj] if inj else [],
        logit_coeff_top=_top_freqs(coeffs, 6),
        logit_coeff_full=coeffs.tolist(),
    )
    if data is not None and model.oracle_fn is not None:
        snap["ablation_test"] = ablation_ce(model, data["test_x"], data["test_y"], config)
    return snap


def _top_freqs(power, k):
    power = np.asarray(power)
    order = np.argsort(power)[::-1][:k]
    return [(int(i + 1), float(power[i])) for i in order]


# --------------------------------------------------------------------------- #
# Per-neuron dominant frequency (internals of calculate_key_freqs, exposed)
# --------------------------------------------------------------------------- #
@t.no_grad()
def neuron_freq_histogram(model, config, all_data=None):
    """Dominant Fourier frequency of every MLP neuron (final position).

    Same computation as transformer.calculate_key_freqs but returns the per-neuron
    arrays instead of just the unique set. Returns (neuron_freqs[d_mlp],
    frac_explained[d_mlp]); a neuron is "specialized" to its freq when frac_explained
    is near 1. Histogramming neuron_freqs (optionally filtered by frac_explained)
    shows whether neurons cluster on the injected frequencies.
    """
    if all_data is None:
        p = config.p
        all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(config.device)
    cache = {}
    model.remove_all_hooks(); model.cache_all(cache); model(all_data); model.remove_all_hooks()
    acts = cache["blocks.0.mlp.hook_post"][:, -1]
    acts = acts - acts.mean(0, keepdim=True)
    fb = transformer.make_fourier_basis(config)
    fna = helpers.fft2d(acts, p=config.p, fourier_basis=fb).reshape(config.p, config.p, config.d_mlp)
    denom = fna.pow(2).sum((0, 1))                      # (d_mlp,)
    freqs = np.zeros(config.d_mlp, dtype=int)
    fracs = np.zeros(config.d_mlp)
    for f in range(1, config.p // 2):
        num = helpers.extract_freq_2d(fna, f, p=config.p).pow(2).sum((0, 1))  # (d_mlp,)
        fe = (num / denom).cpu().numpy()
        better = fe > fracs
        fracs[better] = fe[better]
        freqs[better] = f
    return freqs, fracs
