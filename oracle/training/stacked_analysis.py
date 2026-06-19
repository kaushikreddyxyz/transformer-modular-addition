"""Vectorized uptake snapshots — analysis.py, batched over the model axis.

Each function mirrors a counterpart in analysis.py / transformer.py exactly, with
a leading model axis M, so one snapshot computes the full uptake report for every
model in a stack at once (the snapshots are otherwise ~half the suite's GPU-hours
and would become the bottleneck behind fast stacked training). `_stacked_check.py`
asserts every field here matches the per-model analysis.uptake_snapshot to fp
tolerance.

Per-model splits differ (each seed shuffles its own train/test), so the grid
labels and the train mask carry a model axis; the input grid and Fourier basis
depend only on p and are shared.
"""
import numpy as np
import torch as t
import torch.nn.functional as F

from modular_addition import transformer
from modular_addition.oracle.training import stacked


# --------------------------------------------------------------------------- #
# Shared context for one stack
# --------------------------------------------------------------------------- #
def stacked_context(specs, cfg, device):
    """Grid tokens (shared), per-model labels + train mask, Fourier basis."""
    p = cfg.p
    all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)],
                        dtype=t.long, device=device)                  # (p²,3)
    fb = transformer.make_fourier_basis(cfg).to(device)              # (p_basis,p)
    labels, is_train = [], []
    import dataclasses
    for s in specs:
        c = dataclasses.replace(cfg, seed=s["seed"],
                                frac_train=s.get("config", {}).get("frac_train",
                                                                   cfg.frac_train))
        train_pairs, _ = transformer.gen_train_test(c)
        lab = t.tensor([c.fn(int(i), int(j)) for i, j, _ in all_data.tolist()],
                       dtype=t.long, device=device)
        tr = {(i, j) for i, j, _ in train_pairs}
        mask = t.tensor([(i, j) in tr for i, j, _ in all_data.tolist()],
                        dtype=t.bool, device=device)
        labels.append(lab); is_train.append(mask)
    return dict(all_data=all_data, fb=fb,
                labels=t.stack(labels), is_train=t.stack(is_train))


GRID_BATCH = 16384


@t.no_grad()
def grid_forward(params, specs, cfg, all_data, gate, mask, want_mlp=False,
                 batch=GRID_BATCH):
    """Stacked forward over the full p² grid (chunked). Returns logits
    (M,p²,p) at the read-out position with the '=' class dropped, and (if
    want_mlp) the final-position MLP post-acts (M,p²,d_mlp)."""
    M = gate.shape[0]
    p = cfg.p
    idx_m = t.arange(M, device=all_data.device)[:, None, None]
    g = gate[:, None, None, None]
    logits_chunks, mlp_chunks = [], []
    for i in range(0, all_data.shape[0], batch):
        chunk = all_data[i:i + batch]                                # (c,3)
        xb = chunk.unsqueeze(0).expand(M, -1, -1)                    # (M,c,3)
        term, _, _ = stacked.build_oracle_term(specs, cfg, xb)
        if want_mlp:
            logits, post = stacked.forward_acts(params, xb, mask, idx_m,
                                                g * term, cfg.act_type)
            mlp_chunks.append(post[:, :, -1, :])
        else:
            logits = stacked.forward(params, xb, mask, idx_m, g * term, cfg.act_type)
        logits_chunks.append(logits[:, :, -1, :p])                  # drop '='
    logits = t.cat(logits_chunks, dim=1)
    if want_mlp:
        return logits, t.cat(mlp_chunks, dim=1)
    return logits


# --------------------------------------------------------------------------- #
# W_E / W_L Fourier power (batched we_fourier_power / wl_fourier_power)
# --------------------------------------------------------------------------- #
def _freq_power(power):
    """power (M,p_basis) -> freq_power (M,p//2): cos(2k-1)+sin(2k) per freq k."""
    n = (power.shape[1] - 1) // 2
    cos = power[:, 1:2 * n + 1:2]
    sin = power[:, 2:2 * n + 1:2]
    return cos + sin


def _gini(x):
    """Batched Gini of non-negative rows x (M,n) -> (M,) (matches analysis.gini)."""
    xs = x.abs().sort(dim=1).values
    n = xs.shape[1]
    idx = t.arange(1, n + 1, device=x.device, dtype=xs.dtype)
    s = xs.sum(1)
    num = ((2 * idx - n - 1) * xs).sum(1)
    out = num / (n * s)
    return t.where(s == 0, t.zeros_like(out), out)


def we_power(params, cfg, fb):
    p = cfg.p
    WE = params["embed.W_E"][:, :, :p]                               # (M,d_model,p)
    coeffs = t.einsum("mdp,fp->mdf", WE, fb)
    power = coeffs.pow(2).sum(1)                                     # (M,p_basis)
    fp = _freq_power(power)
    return dict(freq_power=fp, const_power=power[:, 0],
                total_norm=WE.norm(dim=(1, 2)), gini=_gini(fp))


def wl_power(params, cfg, fb):
    p = cfg.p
    WL = t.einsum("mds,mdv->msv", params["blocks.0.mlp.W_out"],
                  params["unembed.W_U"])[:, :, :p]                   # (M,d_mlp,p)
    coeffs = t.einsum("msp,fp->msf", WL, fb)
    power = coeffs.pow(2).sum(1)
    fp = _freq_power(power)
    return dict(freq_power=fp, const_power=power[:, 0],
                total_norm=WL.norm(dim=(1, 2)), gini=_gini(fp))


# --------------------------------------------------------------------------- #
# Logit coefficients (batched _coefficients_lowmem)
# --------------------------------------------------------------------------- #
def logit_coeffs(logits, cfg):
    """(M,p//2): per-frequency coeff of logits on cos(w(x+y-z))."""
    p, device = cfg.p, logits.device
    x = t.arange(p, device=device, dtype=t.int32)
    m = ((x[:, None, None] + x[None, :, None] - x[None, None, :]) % p) \
        .reshape(p * p, p)
    out = []
    for w in range(1, p // 2 + 1):
        cos = t.cos(((m * w) % p).to(t.float32) * (2 * t.pi / p))
        cos = cos / cos.pow(2).sum().sqrt()
        out.append(t.einsum("qp,mqp->m", cos, logits))
    return t.stack(out, dim=1)                                       # (M,p//2)


# --------------------------------------------------------------------------- #
# Per-neuron dominant frequency (batched neuron_freq_histogram)
# --------------------------------------------------------------------------- #
def neuron_freqs(mlp_post, cfg, fb):
    """mlp_post (M,p²,d_mlp) -> per-neuron best freq (M,d_mlp) and frac (M,d_mlp).
    Mirrors analysis.neuron_freq_histogram (range(1, p//2), centered acts)."""
    p, d_mlp = cfg.p, cfg.d_mlp
    acts = mlp_post - mlp_post.mean(1, keepdim=True)
    sq = acts.reshape(acts.shape[0], p, p, d_mlp)
    fna = t.einsum("mxyz,fx,Fy->mfFz", sq, fb, fb)                   # (M,p,p,d_mlp)
    denom = fna.pow(2).sum((1, 2))                                   # (M,d_mlp)
    freqs = t.zeros(acts.shape[0], d_mlp, dtype=t.long, device=acts.device)
    fracs = t.zeros(acts.shape[0], d_mlp, device=acts.device)
    for f in range(1, p // 2):
        idx = [0, 2 * f - 1, 2 * f]
        sub = fna[:, idx][:, :, idx]                                 # (M,3,3,d_mlp)
        fe = sub.pow(2).sum((1, 2)) / denom
        better = fe > fracs
        fracs = t.where(better, fe, fracs)
        freqs = t.where(better, t.full_like(freqs, f), freqs)
    return freqs, fracs


# --------------------------------------------------------------------------- #
# (x+y) Fourier component projection (batched get_component_cos/sin_xpy)
# --------------------------------------------------------------------------- #
def _basis_term(fb, xi, yi):
    return (fb[xi][:, None] * fb[yi][None, :]).flatten()            # (p²,)


def _cos_dir(fb, f):
    return (_basis_term(fb, 2 * f - 1, 2 * f - 1)
            - _basis_term(fb, 2 * f, 2 * f)) / np.sqrt(2)


def _sin_dir(fb, f):
    return (_basis_term(fb, 2 * f, 2 * f - 1)
            + _basis_term(fb, 2 * f - 1, 2 * f)) / np.sqrt(2)


def _component(direction, logits):
    """outer(dir, dir @ logits): (M,p²,p), batched get_component_*_xpy."""
    proj = t.einsum("q,mqp->mp", direction, logits)
    return t.einsum("q,mp->mqp", direction, proj)


def _comp_sum(freqs_per_model, fb, logits, union):
    """Per-model sum of cos+sin (x+y) components over each model's freq set.
    `union` is the sorted distinct freqs; we accumulate one freq at a time so
    only one (M,p²,p) component is materialized at once. Returns (M,p²,p)."""
    M = logits.shape[0]
    out = t.zeros_like(logits)
    sel = t.zeros(M, len(union), device=logits.device)
    pos = {f: i for i, f in enumerate(union)}
    for m, fs in enumerate(freqs_per_model):
        for f in fs:
            sel[m, pos[int(f)]] = 1.0
    for i, f in enumerate(union):
        comp = _component(_cos_dir(fb, f), logits) + _component(_sin_dir(fb, f), logits)
        out = out + sel[:, i][:, None, None] * comp
    return out


# --------------------------------------------------------------------------- #
# High-precision CE over a masked subset (batched test_logits / harness loss)
# --------------------------------------------------------------------------- #
def _ce_rows(logits, labels, mask=None):
    """-mean logprob[label] over rows (per model). mask (M,p²) bool selects a
    subset; None => all rows. Returns (M,)."""
    lp = F.log_softmax(logits.to(t.float32), dim=-1)
    pick = t.gather(lp, -1, labels[..., None]).squeeze(-1)          # (M,p²)
    if mask is None:
        return -pick.mean(1)
    mf = mask.float()
    return -(pick * mf).sum(1) / mf.sum(1)


# --------------------------------------------------------------------------- #
# Full snapshot, returning one analysis-compatible dict per model
# --------------------------------------------------------------------------- #
@t.no_grad()
def uptake_snapshot(params, specs, cfg, sctx, epoch, has_oracle, inj_from,
                    test_x, test_y, test_term, mask, injected=None):
    """Batched analysis.uptake_snapshot. Returns a list of M snapshot dicts
    with exactly the fields analysis.uptake_snapshot emits.

    `injected` (list per model) is the authoritative injected-frequency set from
    sweep.build_oracle — pass it so kinds whose freqs aren't literal in the spec
    (e.g. organic_we's donor-derived modes) score correctly. If None, falls back
    to each spec's oracle['freqs'] (fourier/perexample, where it is literal)."""
    M = len(specs)
    p = cfg.p
    fb = sctx["fb"]
    all_data, labels = sctx["all_data"], sctx["labels"]
    is_train = sctx["is_train"]
    gate = stacked.inject_gate(has_oracle, inj_from, epoch)

    logits, mlp_post = grid_forward(params, specs, cfg, all_data, gate, mask,
                                    want_mlp=True)                  # (M,p²,p),(M,p²,d_mlp)
    we = we_power(params, cfg, fb)
    wl = wl_power(params, cfg, fb)
    coeffs = logit_coeffs(logits, cfg)                              # (M,p//2)
    nfreqs, _ = neuron_freqs(mlp_post, cfg, fb)                     # (M,d_mlp)

    # per-model key freqs / injected freqs
    key_per = [sorted({int(x) for x in nfreqs[m].tolist()}) for m in range(M)]
    if injected is not None:
        inj_per = [[int(f) for f in inj] for inj in injected]
    else:
        inj_per = [[int(f) for f in (s.get("oracle") or {}).get("freqs", [])]
                   for s in specs]

    # excluded loss (per injected freq, per model) — needs per-model train CE
    inj_union = sorted({f for fs in inj_per for f in fs})
    excl = [[0.0] * len(fs) for fs in inj_per]
    for f in inj_union:
        comp = _component(_cos_dir(fb, f), logits) + _component(_sin_dir(fb, f), logits)
        ce = _ce_rows(logits - comp, labels, is_train)             # (M,)
        for m, fs in enumerate(inj_per):
            for k, fk in enumerate(fs):
                if fk == f:
                    excl[m][k] = float(ce[m])

    # trig loss (sufficiency) over injected and key freq sets, with bias corr.
    def trig(freqs_per_model, union):
        if not union:
            return [None] * M
        trig_logits = _comp_sum(freqs_per_model, fb, logits, union)
        bc = (logits - trig_logits).mean(1, keepdim=True)
        ce = _ce_rows(trig_logits + bc, labels)                    # (M,) mode='all'
        return [float(ce[m]) if freqs_per_model[m] else None for m in range(M)]
    trig_inj = trig(inj_per, inj_union)
    # key_freqs may include 0 (unspecialized neurons); transformer.calculate_trig_loss
    # includes that freq-0 component (with its negative-index Fourier quirk), so we
    # pass the key set verbatim — _cos_dir/_sin_dir reproduce the f=0 term exactly.
    key_union = sorted({f for fs in key_per for f in fs})
    trig_key = trig(key_per, key_union)

    # ablation on test set (only where there is an oracle)
    g = gate[:, None, None, None]
    idx_m = t.arange(M, device=test_x.device)[:, None, None]
    lo_on = stacked.forward(params, test_x, mask, idx_m, g * test_term, cfg.act_type)
    lo_off = stacked.forward(params, test_x, mask, idx_m, 0 * test_term, cfg.act_type)
    ce_on, acc_on = stacked.ce_acc(lo_on[:, :, -1, :], test_y, p)
    ce_off, acc_off = stacked.ce_acc(lo_off[:, :, -1, :], test_y, p)

    we_fp, wl_fp = we["freq_power"], wl["freq_power"]
    snaps = []
    for m in range(M):
        kf, inj = key_per[m], inj_per[m]
        snap = dict(
            epoch=epoch,
            key_freqs=kf, injected_freqs=inj,
            injected_in_key_freqs=[k for k in inj if k in kf],
            excluded_loss_injected=excl[m] if inj else [],
            trig_loss_injected=trig_inj[m] if inj else None,
            trig_loss_keyfreqs=trig_key[m] if kf else None,
            we_total_norm=float(we["total_norm"][m]), we_gini=float(we["gini"][m]),
            we_freq_power_injected=[float(we_fp[m, k - 1]) for k in inj] if inj else [],
            we_freq_power_top=_top(we_fp[m], 6),
            we_freq_power_full=we_fp[m].tolist(),
            wl_total_norm=float(wl["total_norm"][m]), wl_gini=float(wl["gini"][m]),
            wl_freq_power_injected=[float(wl_fp[m, k - 1]) for k in inj] if inj else [],
            wl_freq_power_top=_top(wl_fp[m], 6),
            wl_freq_power_full=wl_fp[m].tolist(),
            logit_coeff_injected=[float(coeffs[m, k - 1]) for k in inj] if inj else [],
            logit_coeff_top=_top(coeffs[m], 6),
            logit_coeff_full=coeffs[m].tolist(),
        )
        if has_oracle[m]:
            snap["ablation_test"] = dict(
                ce_on=float(ce_on[m]), ce_off=float(ce_off[m]),
                delta=float(ce_off[m] - ce_on[m]),
                acc_on=float(acc_on[m]), acc_off=float(acc_off[m]))
        snaps.append(snap)
    return snaps


def _top(power, k):
    power = np.asarray(power.tolist() if hasattr(power, "tolist") else power)
    order = np.argsort(power)[::-1][:k]
    return [(int(i + 1), float(power[i])) for i in order]
