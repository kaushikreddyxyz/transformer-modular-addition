"""CPU equivalence check: stacked path vs the per-model harness path.

Proves (without a GPU) that the batched implementation in stacked.py reproduces
the original per-model training:

  1. INIT is bit-identical (stacked slice == OracleTransformer state_dict).
  2. FORWARD at init matches the per-model OracleTransformer forward (both the
     no-oracle and fourier-oracle paths), to fp tolerance.
  3. TRAINING matches: per-epoch post-step train loss from train_group tracks a
     faithful re-implementation of harness.train's inner loop (AdamW betas
     (0.9,0.98) + LambdaLR 10-step warmup + cross_entropy_high_precision), to
     a tight floating-point tolerance (bmm vs mm reduction order is the only
     source of drift).

Run: python -m modular_addition.oracle.tests.test_stacked
"""
import dataclasses

import torch as t
import torch.optim as optim

from modular_addition import transformer, helpers
from modular_addition.oracle import analysis, inject, sweep
from modular_addition.oracle.training import harness, stacked, stacked_analysis


def per_model_reference(spec, cfg, n_epochs):
    """Faithful re-implementation of harness.train's inner loop for one model,
    returning the post-step train loss at every epoch."""
    oracle_fn, _ = sweep.build_oracle(spec.get("oracle"), cfg)
    model, data = harness.setup(cfg, oracle_fn=oracle_fn)
    frozen = sweep.freeze_params(model, spec.get("freeze"))
    inj_from = spec.get("inject_from_epoch", 0)
    has_oracle = oracle_fn is not None
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    sched = optim.lr_scheduler.LambdaLR(opt, lambda step: min(step / 10, 1))
    losses = []
    for epoch in range(n_epochs):
        if has_oracle:
            model.inject = epoch >= inj_from
        logits = model(data["train_x"])[:, -1]
        loss = helpers.cross_entropy_high_precision(logits, data["train_y"])
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        with t.no_grad():
            logits = model(data["train_x"])[:, -1]
            losses.append(helpers.cross_entropy_high_precision(
                logits, data["train_y"]).item())
    return model, data, frozen, losses


def main():
    dev = t.device("cpu")
    P, D, EPOCHS = 23, 32, 40
    # Co-stackable specs: a baseline + two fourier models with DIFFERENT freqs,
    # plus a per-model weight_decay difference, to exercise the per-model paths.
    specs = [
        sweep.spec(exp="chk", label="none_s0", seed=0, oracle=dict(kind="none"),
                   p=P, d_model=D, num_epochs=EPOCHS, config=dict(weight_decay=1.0)),
        sweep.spec(exp="chk", label="f25_s1", seed=1,
                   oracle=dict(kind="fourier", freqs=[2, 5], amp=1.0),
                   p=P, d_model=D, num_epochs=EPOCHS, config=dict(weight_decay=0.5)),
        sweep.spec(exp="chk", label="f3_s2", seed=2,
                   oracle=dict(kind="fourier", freqs=[3], amp=2.0),
                   p=P, d_model=D, num_epochs=EPOCHS, config=dict(weight_decay=1.0)),
    ]
    cfgs = [stacked.make_base_config(s, dev) for s in specs]
    cfgs = [dataclasses.replace(c, seed=s["seed"],
                                weight_decay=s["config"]["weight_decay"])
            for c, s in zip(cfgs, specs)]

    # ---- 1 & 2: init + forward equivalence at init -------------------------- #
    params = stacked.stack_init([s["seed"] for s in specs], cfgs[0], dev)
    max_init = 0.0
    for i, (s, cfg) in enumerate(zip(specs, cfgs)):
        oracle_fn, _ = sweep.build_oracle(s.get("oracle"), cfg)
        model, data = harness.setup(cfg, oracle_fn=oracle_fn)
        sd = model.state_dict()
        for name in stacked.PARAM_NAMES:
            max_init = max(max_init, (params[name][i] - sd[name]).abs().max().item())
    print(f"[1] init max|Δ| (stacked slice vs state_dict): {max_init:.2e}")
    assert max_init == 0.0, "init must be bit-identical"

    # forward at init over each model's own train_x
    train_x, train_y, test_x, test_y = stacked.stack_data(
        [s["seed"] for s in specs], cfgs[0])
    mask = t.tril(t.ones(cfgs[0].n_ctx, cfgs[0].n_ctx))
    idx_m = t.arange(len(specs))[:, None, None]
    train_term, has_oracle, inj_from = stacked.build_oracle_term(specs, cfgs[0], train_x)
    gate = stacked.inject_gate(has_oracle, inj_from, 0)[:, None, None, None]
    with t.no_grad():
        stk_logits = stacked.forward(params, train_x, mask, idx_m, gate * train_term)
    max_fwd = 0.0
    for i, (s, cfg) in enumerate(zip(specs, cfgs)):
        oracle_fn, _ = sweep.build_oracle(s.get("oracle"), cfg)
        model, data = harness.setup(cfg, oracle_fn=oracle_fn)
        model.inject = True
        with t.no_grad():
            ref = model(data["train_x"])
        max_fwd = max(max_fwd, (stk_logits[i] - ref).abs().max().item())
    print(f"[2] forward-at-init max|Δ| (none + fourier paths): {max_fwd:.2e}")
    assert max_fwd < 1e-4, "forward at init diverged"

    # ---- 3: training-trajectory equivalence -------------------------------- #
    ref_losses = []
    for s, cfg in zip(specs, cfgs):
        _, _, _, losses = per_model_reference(s, cfg, EPOCHS)
        ref_losses.append(losses)

    out = stacked.train_group(specs, dev, eval_every=1)
    max_rel = 0.0
    for i in range(len(specs)):
        stk = [h["train_loss"] for h in out["histories"][i]]
        ref = ref_losses[i]
        for a, b in zip(stk, ref):
            max_rel = max(max_rel, abs(a - b) / (abs(b) + 1e-9))
    print(f"[3] training post-step train-loss max relative Δ over {EPOCHS} "
          f"epochs: {max_rel:.2e}")
    print(f"    final losses  stacked={[round(h[-1]['train_loss'],5) for h in out['histories']]}")
    print(f"    final losses  per-model={[round(r[-1],5) for r in ref_losses]}")
    assert max_rel < 1e-3, "training trajectory diverged beyond fp tolerance"

    print("\n✅ stacked path reproduces the per-model path "
          "(init exact; forward & training within fp tolerance).")
    check_snapshots()


def check_snapshots():
    """Compare the batched uptake snapshot against per-model analysis.uptake_snapshot
    field by field, on trained params."""
    dev = t.device("cpu")
    P, D, EPOCHS = 23, 32, 60
    specs = [
        sweep.spec(exp="snap", label="none_s0", seed=0, oracle=dict(kind="none"),
                   p=P, d_model=D, num_epochs=EPOCHS),
        sweep.spec(exp="snap", label="f25_s1", seed=1,
                   oracle=dict(kind="fourier", freqs=[2, 5], amp=1.0),
                   p=P, d_model=D, num_epochs=EPOCHS),
        sweep.spec(exp="snap", label="f3_s2", seed=2,
                   oracle=dict(kind="fourier", freqs=[3], amp=2.0),
                   p=P, d_model=D, num_epochs=EPOCHS),
    ]
    cfg = stacked.make_base_config(specs[0], dev)
    out = stacked.train_group(specs, dev, eval_every=EPOCHS)
    params = out["params"]

    # stacked snapshot
    train_x, train_y, test_x, test_y = stacked.stack_data([s["seed"] for s in specs], cfg)
    test_term, has_oracle, inj_from = stacked.build_oracle_term(specs, cfg, test_x)
    mask = t.tril(t.ones(cfg.n_ctx, cfg.n_ctx))
    sctx = stacked_analysis.stacked_context(specs, cfg, dev)
    epoch = EPOCHS - 1
    stk = stacked_analysis.uptake_snapshot(params, specs, cfg, sctx, epoch,
                                           has_oracle, inj_from, test_x, test_y,
                                           test_term, mask)

    # per-model reference
    worst = {}

    def track(key, a, b):
        import numpy as np
        a = np.asarray(a, dtype=float).ravel(); b = np.asarray(b, dtype=float).ravel()
        if a.size == 0 and b.size == 0:
            return
        d = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))
        worst[key] = max(worst.get(key, 0.0), d)

    discrete_ok = True
    for m, s in enumerate(specs):
        c = dataclasses.replace(cfg, seed=s["seed"])
        oracle_fn, injected = sweep.build_oracle(s.get("oracle"), c)
        model = inject.OracleTransformer(c, oracle_fn=oracle_fn).to(dev)
        with t.no_grad():
            named = dict(model.named_parameters())
            for name in stacked.PARAM_NAMES:
                named[name].copy_(params[name][m])
        model.inject = bool(has_oracle[m])
        data = harness.prepare(c)
        ctx = analysis.metric_context(c, data["train_pairs"])
        ref = analysis.uptake_snapshot(model, c, ctx, injected, data=data)
        sm = stk[m]
        # discrete
        if set(sm["key_freqs"]) != set(ref["key_freqs"]):
            discrete_ok = False
            print(f"  key_freqs differ m{m}: stk={sm['key_freqs']} ref={ref['key_freqs']}")
        # continuous fields
        for key in ("we_total_norm", "we_gini", "wl_total_norm", "wl_gini",
                    "we_freq_power_full", "wl_freq_power_full", "logit_coeff_full",
                    "we_freq_power_injected", "wl_freq_power_injected",
                    "logit_coeff_injected", "excluded_loss_injected"):
            track(key, sm[key], ref[key])
        for key in ("trig_loss_injected", "trig_loss_keyfreqs"):
            if sm[key] is not None and ref[key] is not None:
                track(key, [sm[key]], [ref[key]])
        if "ablation_test" in ref:
            for kk in ("ce_on", "ce_off", "delta", "acc_on", "acc_off"):
                track(f"abl_{kk}", [sm["ablation_test"][kk]], [ref["ablation_test"][kk]])

    print("\n[snapshot] max relative Δ (stacked vs per-model analysis), by field:")
    for k in sorted(worst):
        print(f"    {k:28s} {worst[k]:.2e}")
    mx = max(worst.values())
    assert discrete_ok, "key_freqs mismatch"
    assert mx < 1e-3, f"snapshot field diverged: {mx:.2e}"
    print(f"  key_freqs exact match; max field Δ {mx:.2e}")
    print("\n✅ vectorized snapshots match per-model analysis.uptake_snapshot.")


if __name__ == "__main__":
    main()
