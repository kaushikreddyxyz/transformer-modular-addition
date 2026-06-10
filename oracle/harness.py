"""Fast, reproducible, wandb-free training harness for oracle experiments.

Why not reuse `transformer.Trainer`? It rebuilds the labels with a Python list
comprehension on every `full_loss` call (the dominant cost) and is coupled to
wandb. Here we precompute all tensors once and log to JSONL, so a 30k-epoch
p=113 run takes well under a couple of minutes on the 4090, and many experiments
can run back-to-back.

Reproducibility: `setup()` mirrors `Trainer.__init__` order exactly
(set_seed -> build model -> gen_train_test), so a given Config + seed trains the
same model whether or not an oracle is attached (the oracle is frozen and only
adds a deterministic constant to the forward pass).
"""
import json
import os
import time

import numpy as np
import torch as t
import torch.optim as optim

from modular_addition import transformer, helpers
from modular_addition.oracle.inject import OracleTransformer


# --------------------------------------------------------------------------- #
# Data / model setup
# --------------------------------------------------------------------------- #
def prepare(config: transformer.Config):
    """Precompute train/test/all tensors once (labels included)."""
    train_pairs, test_pairs = transformer.gen_train_test(config)   # self-seeds python random

    def to_xy(pairs):
        x = t.tensor(pairs, dtype=t.long, device=config.device)            # (n, 3)
        y = t.tensor([config.fn(i, j) for i, j, _ in pairs],
                     dtype=t.long, device=config.device)                   # (n,)
        return x, y

    train_x, train_y = to_xy(train_pairs)
    test_x, test_y = to_xy(test_pairs)
    return dict(train_pairs=train_pairs, test_pairs=test_pairs,
                train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


def setup(config: transformer.Config, oracle_fn=None, inject: bool = True):
    """Reproducible (model, data) construction matching Trainer.__init__ order."""
    helpers.set_seed(config.seed)
    model = OracleTransformer(config, oracle_fn=oracle_fn, inject=inject).to(config.device)
    data = prepare(config)
    return model, data


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@t.no_grad()
def loss_acc(model, x, y, config):
    logits = model(x)[:, -1]
    loss = helpers.cross_entropy_high_precision(logits, y).item()
    acc = (logits[:, :config.p].argmax(-1) == y).float().mean().item()
    return loss, acc


@t.no_grad()
def evaluate(model, data, config, epoch):
    tr_loss, tr_acc = loss_acc(model, data["train_x"], data["train_y"], config)
    te_loss, te_acc = loss_acc(model, data["test_x"], data["test_y"], config)
    we = model.embed.W_E[:, :config.p]
    return dict(epoch=epoch, train_loss=tr_loss, test_loss=te_loss,
                train_acc=tr_acc, test_acc=te_acc,
                log_train_loss=float(np.log(max(tr_loss, 1e-12))),
                log_test_loss=float(np.log(max(te_loss, 1e-12))),
                we_norm=we.norm().item(), injecting=bool(model.inject))


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(config: transformer.Config, model, data, *, num_epochs,
          eval_every=100, snapshot_every=2000, snapshot_fn=None,
          inject_from_epoch=0, run_dir=None, label="run",
          grok_acc=0.99, stop_after_grok=None, verbose=True):
    """Train `model` full-batch; log scalars every `eval_every`, heavy uptake
    metrics every `snapshot_every` (via `snapshot_fn(model, epoch)`).

    `inject_from_epoch` gates the oracle on at that epoch (delayed injection).
    `stop_after_grok`: if set, stop this many epochs after the first epoch with
    test_acc >= grok_acc (saves time once grokking is clearly complete).
    Returns dict(history, snapshots, grok_epoch, label, config).
    """
    opt = optim.AdamW(model.parameters(), lr=config.lr,
                      weight_decay=config.weight_decay, betas=(0.9, 0.98))
    sched = optim.lr_scheduler.LambdaLR(opt, lambda step: min(step / 10, 1))

    history, snapshots = [], []
    grok_epoch = None
    jsonl = None
    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)
        jsonl = open(os.path.join(run_dir, f"{label}.jsonl"), "w")

    has_oracle = getattr(model, "oracle_fn", None) is not None
    t0 = time.time()
    for epoch in range(num_epochs):
        if has_oracle:
            model.inject = epoch >= inject_from_epoch

        logits = model(data["train_x"])[:, -1]
        loss = helpers.cross_entropy_high_precision(logits, data["train_y"])
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()

        final_epoch = epoch == num_epochs - 1
        stopping = (stop_after_grok is not None and grok_epoch is not None
                    and epoch >= grok_epoch + stop_after_grok)
        if epoch % eval_every == 0 or final_epoch or stopping:
            rec = evaluate(model, data, config, epoch)
            rec["lr"] = sched.get_last_lr()[0]
            rec["wall_s"] = round(time.time() - t0, 2)
            history.append(rec)
            if jsonl:
                jsonl.write(json.dumps(rec) + "\n"); jsonl.flush()
            if grok_epoch is None and rec["test_acc"] >= grok_acc:
                grok_epoch = epoch
            if verbose and (epoch % (eval_every * 10) == 0 or final_epoch or stopping):
                print(f"[{label}] ep {epoch:6d}  "
                      f"train {rec['train_loss']:.4f}/{rec['train_acc']:.3f}  "
                      f"test {rec['test_loss']:.4f}/{rec['test_acc']:.3f}  "
                      f"|W_E| {rec['we_norm']:.2f}  inj={rec['injecting']}")

        if snapshot_fn is not None and (epoch % snapshot_every == 0 or final_epoch or stopping):
            snap = snapshot_fn(model, epoch)
            snap["epoch"] = epoch
            snapshots.append(snap)

        if stopping:
            break

    if jsonl:
        jsonl.close()

    result = dict(history=history, snapshots=snapshots, grok_epoch=grok_epoch,
                  label=label, num_epochs=num_epochs,
                  config=_config_dict(config), inject_from_epoch=inject_from_epoch,
                  wall_s=round(time.time() - t0, 2))
    if run_dir is not None:
        with open(os.path.join(run_dir, f"{label}.result.json"), "w") as f:
            json.dump(result, f, indent=2, default=_json_default)
    if verbose:
        print(f"[{label}] done in {result['wall_s']}s  grok_epoch={grok_epoch}")
    return result


def _config_dict(config):
    import dataclasses
    d = dataclasses.asdict(config)
    d["device"] = str(config.device)
    return d


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
