"""Stacked-tensor (batched-ensemble) training for the oracle grokking testbed.

WHY THIS EXISTS
---------------
The grokking models are tiny (1-layer transformer, d_model~128). A single
training run is ~85% kernel-launch + Python/autograd overhead and ~15% actual
compute (the attention ops are minute: d_head=32, seq=3). Process-level
parallelism (runner.py) hides some of that but every worker still launches its
own per-epoch kernel storm and the streaming multiprocessors sit mostly idle.

The fix is to train MANY models in ONE process as a single batched model: stack
each weight tensor along a leading "model" axis M and replace every per-model
matmul with one batched einsum. M models then cost ~one model's worth of kernel
launches per epoch, and the matmuls grow M-fold so the SMs actually fill up.
Models that differ only in per-model SCALARS/MASKS — seed (init + data split),
learning rate, weight decay, oracle content, delayed-injection epoch, frozen
parameters, early-stop budget — all stack; only tensor SHAPE
(p, d_model, d_mlp, n_ctx, train/test split sizes, oracle mechanism) forces a
separate stack. See `group_key`.

SCOPE (phase 1): the training core — reproducible stacked init, the batched
forward, a hand-written stacked AdamW with per-model lr/wd + frozen-param masks
+ per-model early-stop, and the fourier/none oracle mechanism (exp01, exp02_1,
exp02_2, exp06, exp07, exp08). The per-example-corrupt (exp04) and answer-hint
(exp05) oracles, the vectorized uptake snapshots, and the per-model file
emission that makes this a drop-in for sweep/runner are layered on top of this
core once the speedup and bit-correctness are confirmed.

DETERMINISM
-----------
Per-model INIT and DATA are reproduced bit-for-bit from each model's seed (the
weight-tensor RNG order mirrors transformer.Transformer.__init__ exactly, and
the data split reuses transformer.gen_train_test). The TRAINING TRAJECTORY is
*not* bit-identical to the per-model path because batched matmul (bmm) kernels
reduce in a different order than the per-model mm kernels — the result is an
equally valid, run-to-run deterministic trajectory, statistically equivalent in
grokking behaviour (validated separately). The stacked AdamW reproduces
torch.optim.AdamW's math exactly (decoupled weight decay, betas, eps, the
LambdaLR 10-step warmup), so a non-frozen, non-stopped model matches the
reference optimiser step-for-step up to that bmm-vs-mm floating-point floor.
"""
import math

import numpy as np
import torch as t
import torch.nn.functional as F

from modular_addition import transformer, helpers


# --------------------------------------------------------------------------- #
# Grouping: which specs may share one stack
# --------------------------------------------------------------------------- #
def _oracle_mechanism(ospec):
    """Collapse oracle kinds to their batched MECHANISM.

    'none' rides along inside any mechanism (its contribution is a zero table /
    zero gate), so a baseline run stacks with its injected siblings. The three
    real mechanisms need different batched code paths and so never co-stack.
    """
    kind = (ospec or {}).get("kind", "none")
    if kind in ("none", "fourier", "organic_we"):
        return "fourier"          # per-token-id additive table (none => zeros)
    if kind == "perexample_corrupt":
        return "perexample"
    if kind == "answer_hint":
        return "answer_hint"
    raise ValueError(f"unknown oracle kind {kind!r}")


def group_key(spec):
    """Stackability signature: specs with equal key share identical tensor
    shapes and a common batched forward, so they may train in one stack.

    Everything NOT in this key (seed, lr, weight_decay, amp, freqs,
    reliability, inject_from_epoch, freeze, stop_after_grok, grok_acc) is a
    per-model scalar/mask handled inside the stack.
    """
    cfg = spec.get("config", {})
    return (
        spec["p"], spec["d_model"], 4 * spec["d_model"],          # p, d_model, d_mlp
        cfg.get("frac_train", 0.3), spec["num_epochs"],
        cfg.get("fn_name", "add"), cfg.get("act_type", "ReLU"),  # labels, activation
        _oracle_mechanism(spec.get("oracle")),
    )


# --------------------------------------------------------------------------- #
# Reproducible per-model init + data, stacked along axis 0
# --------------------------------------------------------------------------- #
# state_dict parameter names, matching transformer.Transformer exactly, so a
# de-stacked slice loads into OracleTransformer with strict=True.
PARAM_NAMES = ("embed.W_E", "pos_embed.W_pos",
               "blocks.0.attn.W_K", "blocks.0.attn.W_Q", "blocks.0.attn.W_V",
               "blocks.0.attn.W_O",
               "blocks.0.mlp.W_in", "blocks.0.mlp.b_in",
               "blocks.0.mlp.W_out", "blocks.0.mlp.b_out",
               "unembed.W_U")


def init_one(seed, cfg):
    """One model's init, reproducing transformer.Transformer.__init__'s RNG
    order EXACTLY (so the stacked init is bit-identical to the per-model path).

    Order of torch.randn draws: W_E, W_pos, W_K, W_Q, W_V, W_O, W_in, W_out,
    W_U (b_in/b_out are zeros and consume no RNG). Tensors are built on CPU
    under the seeded default generator, matching the original which constructs
    on CPU then .to(device).
    """
    helpers.set_seed(seed)
    dm, dv = cfg.d_model, cfg.d_vocab
    nh, dh, dmlp, nctx = cfg.num_heads, cfg.d_head, cfg.d_mlp, cfg.n_ctx
    rs = np.sqrt(dm)
    return {
        "embed.W_E":        t.randn(dm, dv) / rs,
        "pos_embed.W_pos":  t.randn(nctx, dm) / rs,
        "blocks.0.attn.W_K": t.randn(nh, dh, dm) / rs,
        "blocks.0.attn.W_Q": t.randn(nh, dh, dm) / rs,
        "blocks.0.attn.W_V": t.randn(nh, dh, dm) / rs,
        "blocks.0.attn.W_O": t.randn(dm, dh * nh) / rs,
        "blocks.0.mlp.W_in": t.randn(dmlp, dm) / rs,
        "blocks.0.mlp.b_in": t.zeros(dmlp),
        "blocks.0.mlp.W_out": t.randn(dm, dmlp) / rs,
        "blocks.0.mlp.b_out": t.zeros(dm),
        "unembed.W_U":      t.randn(dm, dv) / np.sqrt(dv),
    }


def stack_init(seeds, cfg, device):
    """Stack per-seed inits into leaf parameter tensors of shape (M, *)."""
    per = [init_one(s, cfg) for s in seeds]
    params = {}
    for name in PARAM_NAMES:
        p = t.stack([d[name] for d in per], dim=0).to(device)
        p.requires_grad_(True)
        params[name] = p
    return params


def stack_data(seeds, cfg):
    """Per-seed train/test splits, stacked. Shapes are identical across seeds
    (same p, frac_train) so they batch; only the row CONTENT differs.

    Returns (train_x (M,Btr,3) long, train_y (M,Btr) long, test_x, test_y).
    Mirrors harness.prepare / transformer.gen_train_test (which self-seeds
    python random by config.seed).
    """
    import dataclasses
    trx, trY, tex, teY = [], [], [], []
    for s in seeds:
        c = dataclasses.replace(cfg, seed=s)
        train_pairs, test_pairs = transformer.gen_train_test(c)

        def to_xy(pairs):
            x = t.tensor(pairs, dtype=t.long)                       # (n,3)
            y = t.tensor([c.fn(i, j) for i, j, _ in pairs], dtype=t.long)
            return x, y
        tx, ty = to_xy(train_pairs)
        ex, ey = to_xy(test_pairs)
        trx.append(tx); trY.append(ty); tex.append(ex); teY.append(ey)
    dev = cfg.device
    return (t.stack(trx).to(dev), t.stack(trY).to(dev),
            t.stack(tex).to(dev), t.stack(teY).to(dev))


# --------------------------------------------------------------------------- #
# Stacked oracle: precompute each model's additive term by REUSING inject.py
# --------------------------------------------------------------------------- #
# The oracle is a frozen function of the input tokens ONLY (a fixed additive
# residual term), so we never reimplement it batched: we call each model's
# existing, validated oracle_fn ONCE on that model's tokens and stack. Any
# oracle kind that satisfies the oracle_fn(x)->(B,n_ctx,d_model) contract works
# with zero new code — the robustness property ModelGrid relies on.
def build_oracle_term(specs, cfg, x):
    """Precompute the raw (ungated) additive oracle term for token batch `x`.

    `x` is (M, B, n_ctx) long (each model's own tokens). Returns
    (term (M,B,n_ctx,d_model), has_oracle (M,) bool, inject_from (M,) long).
    'none' models contribute a zero term. cfg supplies p/d_model/device; the
    per-model oracle params come from each spec (seed is irrelevant to the
    oracle except perexample's map_seed, which lives in the spec).
    """
    from modular_addition.oracle import sweep
    terms, has, inj_from = [], [], []
    for m, s in enumerate(specs):
        ofn, _ = sweep.build_oracle(s.get("oracle"), cfg)
        if ofn is None:
            terms.append(t.zeros(x.shape[1], cfg.n_ctx, cfg.d_model, device=x.device))
            has.append(False)
        else:
            terms.append(ofn(x[m]))                      # (B,n_ctx,d_model)
            has.append(True)
        inj_from.append(int(s.get("inject_from_epoch", 0)))
    return (t.stack(terms),
            t.tensor(has, device=x.device),
            t.tensor(inj_from, dtype=t.long, device=x.device))


def inject_gate(has_oracle, inject_from, epoch):
    """Per-model on/off gate (M,) float at `epoch` (delayed-injection aware)."""
    return ((inject_from <= epoch) & has_oracle).float()


# --------------------------------------------------------------------------- #
# Batched forward (1-layer transformer, no LayerNorm — it's disabled upstream)
# --------------------------------------------------------------------------- #
def forward_acts(P, x, mask, idx_m, oracle_add=None, act="ReLU"):
    """Stacked forward returning (logits (M,B,nctx,d_vocab), mlp_post
    (M,B,nctx,d_mlp)). mlp_post is the blocks.0.mlp.hook_post activation the
    uptake analysis reads. Each einsum is the per-model einsum from
    transformer.py with a leading 'm'. P: dict of (M,*) params; x: (M,B,nctx)
    long; mask: (nctx,nctx) causal buffer.
    """
    M, B, nctx = x.shape
    we_t = P["embed.W_E"].transpose(1, 2)              # (M,d_vocab,d_model)
    h = we_t[idx_m, x]                                  # (M,B,nctx,d_model)
    if oracle_add is not None:
        h = h + oracle_add
    h = h + P["pos_embed.W_pos"][:, None, :, :]
    # attention
    k = t.einsum("mihd,mbpd->mbiph", P["blocks.0.attn.W_K"], h)
    q = t.einsum("mihd,mbpd->mbiph", P["blocks.0.attn.W_Q"], h)
    v = t.einsum("mihd,mbpd->mbiph", P["blocks.0.attn.W_V"], h)
    d_head = P["blocks.0.attn.W_K"].shape[2]
    scores = t.einsum("mbiph,mbiqh->mbiqp", k, q)
    scores = t.tril(scores) - 1e10 * (1 - mask[:nctx, :nctx])
    attn = F.softmax(scores / np.sqrt(d_head), dim=-1)
    z = t.einsum("mbiph,mbiqp->mbiqh", v, attn)        # (M,B,heads,q,d_head)
    z_flat = z.permute(0, 1, 3, 2, 4).reshape(M, B, nctx, -1)   # m b q (i h)
    attn_out = t.einsum("mdf,mbqf->mbqd", P["blocks.0.attn.W_O"], z_flat)
    h = h + attn_out
    # mlp
    pre = t.einsum("mfd,mbpd->mbpf", P["blocks.0.mlp.W_in"], h) \
        + P["blocks.0.mlp.b_in"][:, None, None, :]
    post = F.gelu(pre) if act == "GeLU" else F.relu(pre)
    mlp_out = t.einsum("mdf,mbpf->mbpd", P["blocks.0.mlp.W_out"], post) \
        + P["blocks.0.mlp.b_out"][:, None, None, :]
    h = h + mlp_out
    logits = t.einsum("mbpd,mdv->mbpv", h, P["unembed.W_U"])
    return logits, post


def forward(P, x, mask, idx_m, oracle_add=None, act="ReLU"):
    """Stacked forward returning just logits (M,B,nctx,d_vocab)."""
    return forward_acts(P, x, mask, idx_m, oracle_add, act)[0]


def ce_acc(logits_last, y, p):
    """Per-model high-precision CE loss (M,) and accuracy (M,).

    Mirrors helpers.cross_entropy_high_precision (log_softmax over the full
    vocab, gather the label, negative mean) and harness.loss_acc's accuracy
    (argmax over the p number-token logits only).
    """
    lp = F.log_softmax(logits_last.to(t.float32), dim=-1)          # (M,B,V)
    pick = t.gather(lp, -1, y[..., None]).squeeze(-1)              # (M,B)
    loss = -pick.mean(dim=1)                                        # (M,)
    acc = (logits_last[..., :p].argmax(-1) == y).float().mean(dim=1)
    return loss, acc


# --------------------------------------------------------------------------- #
# Stacked AdamW: torch.optim.AdamW math, per-model lr/wd + frozen/stopped masks
# --------------------------------------------------------------------------- #
class StackedAdamW:
    """One optimiser over all stacked params with per-model lr & weight decay.

    Reproduces torch.optim.AdamW exactly per model: decoupled weight decay
    applied to the param (scaled by the *scheduled* lr), Adam moments with bias
    correction, eps inside the denom. `train_mask[name]` is a (M,1,...) float
    in {0,1}: 0 freezes that param for that model (no update, matching
    requires_grad_(False)). `active` (M,) gates whole models off after they
    early-stop. Both gate the *entire* update (decay + step), so a frozen /
    stopped slice is left bit-untouched.
    """

    def __init__(self, params, base_lr, weight_decay, train_mask,
                 betas=(0.9, 0.98), eps=1e-8):
        self.params = params                  # name -> (M,*) leaf tensor
        self.base_lr = base_lr                # (M,) tensor
        self.wd = weight_decay                # (M,) tensor
        self.b1, self.b2 = betas
        self.eps = eps
        self.train_mask = train_mask          # name -> (M,1,...) float or None
        self.m = {k: t.zeros_like(v) for k, v in params.items()}
        self.v = {k: t.zeros_like(v) for k, v in params.items()}
        self.step_t = 0

    @staticmethod
    def _bcast(vec, ndim):
        return vec.view(vec.shape[0], *([1] * (ndim - 1)))

    def step(self, epoch, active):
        """One optimiser step. `epoch` (0-indexed) drives the LambdaLR warmup
        min(epoch/10, 1); `active` is a (M,) float gate (0 once a model stops)."""
        self.step_t += 1
        bc1 = 1 - self.b1 ** self.step_t
        bc2 = 1 - self.b2 ** self.step_t
        warm = min(epoch / 10.0, 1.0)
        for k, p in self.params.items():
            g = p.grad
            if g is None:
                continue
            nd = p.dim()
            mask = self.train_mask.get(k)                  # (M,1,...) or None
            gate = self._bcast(active, nd)
            if mask is not None:
                gate = gate * mask
            # Freeze gradient where the update is gated off so frozen moments
            # don't drift (irrelevant to the param, but keeps state clean).
            g = g * gate
            self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = (self.v[k].sqrt() / math.sqrt(bc2)).add_(self.eps)
            lr = self._bcast(self.base_lr * warm, nd)      # scheduled lr (M,1,..)
            wd = self._bcast(self.wd, nd)
            update = lr / bc1 * (self.m[k] / denom) + lr * wd * p
            p.data.add_(-(gate * update))
            p.grad = None


# --------------------------------------------------------------------------- #
# Training driver for one stack
# --------------------------------------------------------------------------- #
def make_base_config(spec, device):
    """Per-model Config: applies every valid Config override in spec['config']
    (frac_train, fn_name, act_type, lr, weight_decay, ...). For a stack the
    group_key guarantees the shape/forward/label-affecting fields are equal, so
    specs[0]'s base config is correct for the whole stack; per-model lr/wd are
    read straight from the specs by the optimiser, not from this Config."""
    import dataclasses
    from modular_addition.oracle import sweep
    valid = {f.name for f in dataclasses.fields(transformer.Config)}
    overrides = {k: v for k, v in spec.get("config", {}).items() if k in valid}
    overrides.setdefault("frac_train", 0.3)
    return sweep.make_config(seed=spec["seed"], p=spec["p"],
                             d_model=spec["d_model"],
                             num_epochs=spec["num_epochs"], device=device,
                             **overrides)


def build_masks(specs, params):
    """Per-model frozen-param masks from each spec's `freeze` patterns.

    A param is frozen for model m if any of m's freeze patterns is a substring
    of the param name (matching sweep.freeze_params). Returns name -> (M,1,...)
    float mask, or {} entries omitted when no model freezes that name.
    """
    M = len(specs)
    masks = {}
    for name, p in params.items():
        col = t.ones(M, device=p.device)
        any_frozen = False
        for i, s in enumerate(specs):
            pats = s.get("freeze") or []
            if any(pat in name for pat in pats):
                col[i] = 0.0
                any_frozen = True
        if any_frozen:
            masks[name] = col.view(M, *([1] * (p.dim() - 1)))
    # Mirror sweep.freeze_params: a non-empty freeze pattern matching no parameter
    # is a typo — fail loudly instead of silently training the "frozen" weights
    # (and emitting a misleading frozen_params=[] artifact).
    for s in specs:
        pats = list(s.get("freeze") or [])
        if pats and not any(any(pat in n for pat in pats) for n in PARAM_NAMES):
            raise ValueError(f"freeze patterns {pats} matched no parameters")
    return masks


def train_group(specs, device, eval_every=200, progress=False):
    """Train one stackable group of specs together. Returns per-model history
    dicts + grok epochs + final stacked params.

    This is the integration core: per-model lr/weight_decay, frozen params,
    delayed injection, and early-stop are all handled per model. Snapshots and
    file emission are layered on elsewhere; this returns the scalar history
    (loss/acc) that the figures' history series and grok_epoch need.
    """
    assert len({group_key(s) for s in specs}) == 1, "specs not co-stackable"
    M = len(specs)
    cfg = make_base_config(specs[0], device)
    p = cfg.p
    num_epochs = specs[0]["num_epochs"]

    params = stack_init([s["seed"] for s in specs], cfg, device)
    train_x, train_y, test_x, test_y = stack_data([s["seed"] for s in specs], cfg)
    train_term, has_oracle, inj_from = build_oracle_term(specs, cfg, train_x)
    test_term, _, _ = build_oracle_term(specs, cfg, test_x)
    mask = t.tril(t.ones(cfg.n_ctx, cfg.n_ctx, device=device))
    idx_m = t.arange(M, device=device)[:, None, None]
    act = cfg.act_type

    def cfg_field(s, name, default):
        return s.get("config", {}).get(name, default)
    base_lr = t.tensor([cfg_field(s, "lr", cfg.lr) for s in specs],
                       dtype=t.float32, device=device)
    wd = t.tensor([cfg_field(s, "weight_decay", cfg.weight_decay) for s in specs],
                  dtype=t.float32, device=device)
    grok_acc = t.tensor([s.get("grok_acc", 0.99) for s in specs], device=device)
    INF = num_epochs + 1
    stop_budget = t.tensor([s.get("stop_after_grok") if s.get("stop_after_grok")
                            is not None else INF for s in specs],
                           dtype=t.long, device=device)

    opt = StackedAdamW(params, base_lr, wd, build_masks(specs, params))

    histories = [[] for _ in range(M)]
    grok_epoch = t.full((M,), -1, dtype=t.long, device=device)
    stop_epoch = t.full((M,), INF, dtype=t.long, device=device)
    active = t.ones(M, device=device)

    rng = range(num_epochs)
    if progress:
        from tqdm.auto import tqdm
        rng = tqdm(rng, desc=f"stack[{M}]", unit="ep")

    @t.no_grad()
    def evaluate(epoch):
        g = inject_gate(has_oracle, inj_from, epoch)[:, None, None, None]
        tr = forward(params, train_x, mask, idx_m, g * train_term, act)
        te = forward(params, test_x, mask, idx_m, g * test_term, act)
        trl, tra = ce_acc(tr[:, :, -1, :], train_y, p)
        tel, tea = ce_acc(te[:, :, -1, :], test_y, p)
        we_norm = params["embed.W_E"][:, :, :p].norm(dim=(1, 2))
        return trl, tra, tel, tea, we_norm

    for epoch in rng:
        gate = inject_gate(has_oracle, inj_from, epoch)[:, None, None, None]
        logits = forward(params, train_x, mask, idx_m, gate * train_term, act)
        loss_m, _ = ce_acc(logits[:, :, -1, :], train_y, p)
        (loss_m * active).sum().backward()
        opt.step(epoch, active)

        final = epoch == num_epochs - 1
        if epoch % eval_every == 0 or final:
            trl, tra, tel, tea, we = evaluate(epoch)
            newly = (grok_epoch < 0) & (tea >= grok_acc)
            grok_epoch = t.where(newly, t.full_like(grok_epoch, epoch), grok_epoch)
            grokked = grok_epoch >= 0
            stop_epoch = t.where(grokked & (stop_epoch > num_epochs),
                                 grok_epoch + stop_budget, stop_epoch)
            for i in range(M):
                histories[i].append(dict(
                    epoch=epoch,
                    train_loss=float(trl[i]), test_loss=float(tel[i]),
                    train_acc=float(tra[i]), test_acc=float(tea[i]),
                    we_norm=float(we[i]),
                    injecting=bool((inj_from[i] <= epoch) and has_oracle[i])))
        # deactivate models past their early-stop epoch
        active = (t.arange(M, device=device) * 0 + epoch < stop_epoch).float()
        if float(active.sum()) == 0:
            break

    return dict(
        params=params,
        histories=histories,
        grok_epoch=[int(g) if g >= 0 else None for g in grok_epoch.tolist()],
        config=cfg, specs=specs)
