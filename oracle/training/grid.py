"""ModelGrid — train a heterogeneous list of run specs as a few batched stacks.

ModelGrid is the seamless entry point: hand it the specs from any experiment(s)
and it (1) partitions them into stackable groups by shape signature, (2) routes
anything the batched fast path can't faithfully reproduce to the proven
per-model path (`sweep.execute`) so novel future experiments never train wrong,
just unaccelerated, and (3) trains each group as one (memory-chunked) stack,
emitting per-model result.json / jsonl / checkpoints byte-compatible with
harness.train so make_figures / push_to_hf / backfill_wl keep working unchanged.

The fast path assumes the model harness.setup actually builds: a 1-layer
transformer, no LayerNorm, ReLU/GeLU MLP, and a frozen additive oracle that is a
function of the input tokens (every current oracle kind). Specs that violate any
assumption fall back. See `fast_path_ok`.
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch as t

from modular_addition.oracle import sweep
from modular_addition.oracle.training import harness, stacked, stacked_analysis


SUPPORTED_ORACLE_KINDS = {"none", "fourier", "organic_we",
                          "perexample_corrupt", "answer_hint"}


def fast_path_ok(spec):
    """True if the batched forward faithfully reproduces this spec's model.

    Conservative on purpose: anything outside the stacked forward's assumptions
    (multi-layer, LayerNorm, exotic activation, unknown oracle mechanism) is
    sent to the per-model fallback rather than silently mis-trained.
    """
    cfg = spec.get("config", {})
    if cfg.get("num_layers", 1) != 1:
        return False
    if cfg.get("use_ln", False):
        return False
    if cfg.get("act_type", "ReLU") not in ("ReLU", "GeLU"):
        return False
    kind = (spec.get("oracle") or {}).get("kind", "none")
    return kind in SUPPORTED_ORACLE_KINDS


def _per_model_bytes(spec):
    """Rough peak training bytes/model (fwd acts + grads + Adam + oracle term),
    used to pick a memory-safe stack size for any shape."""
    p, d_model = spec["p"], spec["d_model"]
    frac = spec.get("config", {}).get("frac_train", 0.3)
    n_train = int(frac * p * p)
    d_mlp, n_ctx = 4 * d_model, 3
    act = n_train * n_ctx * d_mlp * 4              # one big activation
    return 8 * act                                 # ~8x for graph+grads+adam+oracle


class ModelGrid:
    def __init__(self, specs, stack_size=None, target_gb=14.0):
        self.target_gb = target_gb
        self.stack_size = stack_size
        groups = defaultdict(list)
        self.fallback = []
        for s in specs:
            if fast_path_ok(s):
                groups[stacked.group_key(s)].append(s)
            else:
                self.fallback.append(s)
        self.chunks = []           # list of (group_key, [specs]) chunks to stack
        for key, gs in groups.items():
            if len(gs) < 2:        # nothing to batch — use the proven path
                self.fallback.extend(gs)
                continue
            cap = self._cap(gs[0])
            for i in range(0, len(gs), cap):
                self.chunks.append((key, gs[i:i + cap]))

    def _cap(self, spec):
        if self.stack_size:
            return self.stack_size
        budget = self.target_gb * 1e9
        return max(2, min(128, int(budget / _per_model_bytes(spec))))

    def plan(self):
        lines = [f"{len(self.chunks)} stacked chunk(s), "
                 f"{len(self.fallback)} fallback (per-model) spec(s):"]
        for key, gs in self.chunks:
            exps = sorted({s["exp"] for s in gs})
            lines.append(f"  stack[{len(gs):>3}]  p={key[0]} d={key[1]} "
                         f"ep={key[4]} mech={key[-1]}  ({'+'.join(exps)})")
        if self.fallback:
            fb = sorted({s["exp"] for s in self.fallback})
            lines.append(f"  fallback: {len(self.fallback)} spec(s) ({'+'.join(fb)})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def run(self, device, *, results_dir=None, use_wandb=False, force=False,
            progress=True):
        results_dir = results_dir or str(sweep.RESULTS_DIR)
        counts = {"trained": 0, "skip": 0, "fallback": 0}
        for key, gs in self.chunks:
            n = self._train_chunk(gs, device, results_dir, force, use_wandb,
                                  progress)
            counts["trained"] += n
            counts["skip"] += len(gs) - n
        for s in self.fallback:
            sweep.execute(s, device=device, use_wandb=use_wandb, force=force,
                          verbose=False)
            counts["fallback"] += 1
        return counts

    # ------------------------------------------------------------------ #
    def _train_chunk(self, specs, device, results_dir, force, use_wandb, progress):
        # resumability: drop specs already done (unless force)
        todo = [s for s in specs
                if force or not _result_path(results_dir, s).exists()]
        if not todo:
            return 0
        specs = todo
        M = len(specs)
        cfg = stacked.make_base_config(specs[0], device)
        p, n_ctx = cfg.p, cfg.n_ctx
        num_epochs = specs[0]["num_epochs"]
        cfgs = [stacked.make_base_config(s, device) for s in specs]
        cfg_dicts = [harness._config_dict(c) for c in cfgs]

        params = stacked.stack_init([s["seed"] for s in specs], cfg, device)
        train_x, train_y, test_x, test_y = stacked.stack_data(
            [s["seed"] for s in specs], cfg)
        train_term, has_oracle, inj_from = stacked.build_oracle_term(specs, cfg, train_x)
        test_term, _, _ = stacked.build_oracle_term(specs, cfg, test_x)
        mask = t.tril(t.ones(n_ctx, n_ctx, device=device))
        idx_m = t.arange(M, device=device)[:, None, None]
        act = cfg.act_type

        def field(s, name, default):
            return s.get("config", {}).get(name, default)
        base_lr = t.tensor([field(s, "lr", cfg.lr) for s in specs], device=device)
        wd = t.tensor([field(s, "weight_decay", cfg.weight_decay) for s in specs],
                      device=device)
        grok_acc = t.tensor([s.get("grok_acc", 0.99) for s in specs], device=device)
        INF = num_epochs + 1
        budget = t.tensor([s.get("stop_after_grok") or INF for s in specs],
                          dtype=t.long, device=device)
        eval_every = t.tensor([s.get("eval_every", sweep.EVAL_EVERY) for s in specs],
                              dtype=t.long, device=device)
        snap_every = t.tensor([s.get("snapshot_every", sweep.SNAPSHOT_EVERY)
                               for s in specs], dtype=t.long, device=device)
        want_snap = [bool(s.get("snapshots", True)) for s in specs]
        ckpt_sets = [set(s.get("ckpt_epochs") or harness.CKPT_EPOCHS) for s in specs]
        injected = [sweep.build_oracle(s.get("oracle"), cfg)[1] for s in specs]
        frozen = [[n for n in stacked.PARAM_NAMES
                   if any(pat in n for pat in (s.get("freeze") or []))]
                  for s in specs]

        opt = stacked.StackedAdamW(params, base_lr, wd,
                                   stacked.build_masks(specs, params))
        histories = [[] for _ in range(M)]
        snaps = [[] for _ in range(M)]
        grok_epoch = t.full((M,), -1, dtype=t.long, device=device)
        stop_epoch = t.full((M,), INF, dtype=t.long, device=device)
        active = t.ones(M, device=device)
        sctx = [None]
        t0 = time.time()

        @t.no_grad()
        def do_eval(epoch):
            g = stacked.inject_gate(has_oracle, inj_from, epoch)[:, None, None, None]
            tr = stacked.forward(params, train_x, mask, idx_m, g * train_term, act)
            te = stacked.forward(params, test_x, mask, idx_m, g * test_term, act)
            trl, tra = stacked.ce_acc(tr[:, :, -1, :], train_y, p)
            tel, tea = stacked.ce_acc(te[:, :, -1, :], test_y, p)
            we = params["embed.W_E"][:, :, :p].norm(dim=(1, 2))
            return trl, tra, tel, tea, we

        def save_ckpt(m, epochs_done, **meta):
            d = os.path.join(results_dir, specs[m]["exp"], "checkpoints",
                             specs[m]["label"])
            os.makedirs(d, exist_ok=True)
            sd = {n: params[n][m].detach().cpu() for n in stacked.PARAM_NAMES}
            sd["blocks.0.attn.mask"] = mask.detach().cpu()
            t.save(dict(model=sd, epochs_done=epochs_done, label=specs[m]["label"],
                        config=cfg_dicts[m], inject_from_epoch=int(inj_from[m]),
                        **meta),
                   os.path.join(d, f"ep{epochs_done:06d}.pth"))

        rng = range(num_epochs)
        if progress:
            from tqdm.auto import tqdm
            rng = tqdm(rng, desc=f"stack[{M}] {specs[0]['exp']}", unit="ep",
                       miniters=200, mininterval=0.5)

        for epoch in rng:
            gate = stacked.inject_gate(has_oracle, inj_from, epoch)[:, None, None, None]
            logits = stacked.forward(params, train_x, mask, idx_m, gate * train_term, act)
            loss_m, _ = stacked.ce_acc(logits[:, :, -1, :], train_y, p)
            (loss_m * active).sum().backward()
            opt.step(epoch, active)

            final = epoch == num_epochs - 1
            stopping = (active > 0) & (epoch >= stop_epoch)            # (M,) bool
            due_eval = ((epoch % eval_every == 0) | final | stopping) & (active > 0)
            due_snap = (((epoch % snap_every == 0) | final | stopping)
                        & (active > 0) & t.tensor(want_snap, device=device))

            # per-model checkpoints scheduled at (epoch+1) in their ckpt set
            for m in range(M):
                if active[m] > 0 and (epoch + 1) in ckpt_sets[m]:
                    save_ckpt(m, epoch + 1)

            if bool(due_eval.any()):
                trl, tra, tel, tea, we = do_eval(epoch)
                newly = (grok_epoch < 0) & (tea >= grok_acc)
                grok_epoch = t.where(newly, t.full_like(grok_epoch, epoch), grok_epoch)
                got = (grok_epoch >= 0) & (stop_epoch > num_epochs)
                stop_epoch = t.where(got, grok_epoch + budget, stop_epoch)
                lr_now = base_lr * min(epoch / 10.0, 1.0)
                wall = round(time.time() - t0, 2)
                for m in range(M):
                    if not due_eval[m]:
                        continue
                    tl, vl = float(trl[m]), float(tel[m])
                    histories[m].append(dict(
                        epoch=epoch, train_loss=tl, test_loss=vl,
                        train_acc=float(tra[m]), test_acc=float(tea[m]),
                        log_train_loss=float(np.log(max(tl, 1e-12))),
                        log_test_loss=float(np.log(max(vl, 1e-12))),
                        we_norm=float(we[m]),
                        injecting=bool((inj_from[m] <= epoch) and has_oracle[m]),
                        lr=float(lr_now[m]), wall_s=wall))

            if bool(due_snap.any()):
                if sctx[0] is None:
                    sctx[0] = stacked_analysis.stacked_context(specs, cfg, device)
                ss = stacked_analysis.uptake_snapshot(
                    params, specs, cfg, sctx[0], epoch, has_oracle, inj_from,
                    test_x, test_y, test_term, mask, injected=injected)
                for m in range(M):
                    if due_snap[m]:
                        snaps[m].append(ss[m])

            if bool(stopping.any()):
                for m in range(M):
                    if stopping[m] and (epoch + 1) not in ckpt_sets[m]:
                        save_ckpt(m, epoch + 1, stopped_at_grok=True)
            # active for the NEXT step: a model updates through its stop epoch
            # inclusive (matching harness, which steps then breaks), so it goes
            # inactive once (epoch+1) passes stop_epoch.
            active = ((epoch + 1) <= stop_epoch).float()
            if float(active.sum()) == 0:
                break

        # ---- emit per-model artifacts (byte-compatible with harness) ------ #
        wall = round(time.time() - t0, 2)
        for m in range(M):
            ge = int(grok_epoch[m]) if grok_epoch[m] >= 0 else None
            result = dict(history=histories[m], snapshots=snaps[m], grok_epoch=ge,
                          label=specs[m]["label"], num_epochs=num_epochs,
                          config=cfg_dicts[m], inject_from_epoch=int(inj_from[m]),
                          wall_s=wall, spec=specs[m],
                          injected_freqs=list(injected[m]), frozen_params=frozen[m])
            d = os.path.join(results_dir, specs[m]["exp"])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"{specs[m]['label']}.jsonl"), "w") as f:
                for rec in histories[m]:
                    f.write(json.dumps(rec) + "\n")
            with open(os.path.join(d, f"{specs[m]['label']}.result.json"), "w") as f:
                json.dump(result, f, indent=2, default=harness._json_default)
        if use_wandb:
            _log_aggregate(specs, histories, grok_epoch)
        return M


def _result_path(results_dir, s):
    return _P(results_dir) / s["exp"] / f"{s['label']}.result.json"


def _P(x):
    from pathlib import Path
    return Path(x)


def _log_aggregate(specs, histories, grok_epoch):
    """One wandb run summarizing the stack (median/min/max curves, frac grokked)."""
    try:
        import wandb
        run = wandb.init(project=harness.WANDB_PROJECT,
                         group=specs[0]["exp"],
                         name=f"{specs[0]['exp']}/stack_{len(specs)}",
                         reinit=True)
        # align on the shortest history; log distribution per logged epoch
        L = min(len(h) for h in histories)
        for i in range(L):
            tel = [h[i]["test_loss"] for h in histories]
            tea = [h[i]["test_acc"] for h in histories]
            run.log(dict(epoch=histories[0][i]["epoch"],
                         test_loss_med=float(np.median(tel)),
                         test_loss_max=float(np.max(tel)),
                         test_acc_med=float(np.median(tea)),
                         frac_grokked=float(np.mean([a >= 0.99 for a in tea]))),
                    step=histories[0][i]["epoch"])
        run.summary["n_models"] = len(specs)
        run.summary["n_grokked"] = int((grok_epoch >= 0).sum())
        run.finish()
    except Exception as e:  # noqa: BLE001 — monitoring must never kill a run
        print(f"(wandb aggregate skipped: {e})")
