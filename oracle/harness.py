"""Fast, reproducible training harness for oracle experiments.

Why not reuse `transformer.Trainer`? It rebuilds the labels with a Python list
comprehension on every `full_loss` call (the dominant cost). Here we precompute
all tensors once, so a 30k-epoch p=113 run takes well under a couple of minutes
on the 4090, and many experiments can run back-to-back.

Logging: every `train()` call is its own wandb run (project `oracle-encodings`,
grouped by experiment) with scalars logged at `step=epoch` so charts populate
live. JSONL + result.json files are written alongside as the source of truth
for `make_figures.py`; wandb is the monitoring/comparison layer. Set
`use_wandb=False` (or env `WANDB_MODE=disabled`) for offline/smoke usage.

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
import wandb
from tqdm.auto import tqdm

from modular_addition import transformer, helpers
from modular_addition.oracle.inject import OracleTransformer

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "oracle-encodings")
# wandb's default init banner + finish summary (~15 lines per run × 356 runs)
# drowns the runner's progress bar. Silent mode redirects all of it to
# wandb/run-*/logs/debug.log; metrics still sync to the web UI unchanged.
# Export WANDB_SILENT=false to get the chatty console back.
os.environ.setdefault("WANDB_SILENT", "true")

# "Model after N epochs of training" — saved when epoch+1 == N, so 30000 is the
# fully-trained model under the default num_epochs=30_000.
CKPT_EPOCHS = (10, 100, 1000, 5000, 7500, 10_000, 15_000, 25_000, 30_000)


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
def _wandb_init(use_wandb, label, config, *, group=None, extra_config=None,
                inject_from_epoch=0, num_epochs=None, verbose=True):
    """Start a wandb run for this training run; returns the run or None.

    Console-quiet (WANDB_SILENT above); in verbose mode prints the run URL as
    the single line of wandb output. Never raises: experiments must survive a
    missing login / offline laptop.
    """
    if not use_wandb or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        wcfg = _config_dict(config)
        wcfg.update(dict(label=label, inject_from_epoch=inject_from_epoch,
                         num_epochs=num_epochs))
        wcfg.update(extra_config or {})
        name = f"{group}/{label}" if group else label
        run = wandb.init(project=WANDB_PROJECT, group=group, name=name,
                         config=wcfg, reinit=True)
        if verbose and getattr(run, "url", None):
            print(f"[{label}] wandb → {run.url}")
        return run
    except Exception as e:  # noqa: BLE001 — wandb failure must not kill a sweep
        print(f"[{label}] wandb.init failed ({e}); continuing without wandb")
        return None


def train(config: transformer.Config, model, data, *, num_epochs,
          eval_every=100, snapshot_every=2000, snapshot_fn=None,
          inject_from_epoch=0, run_dir=None, label="run",
          grok_acc=0.99, stop_after_grok=None, verbose=True,
          use_wandb=True, wandb_group=None, wandb_config=None,
          ckpt_epochs=CKPT_EPOCHS, result_extra=None):
    """Train `model` full-batch; log scalars every `eval_every`, heavy uptake
    metrics every `snapshot_every` (via `snapshot_fn(model, epoch)`).

    `inject_from_epoch` gates the oracle on at that epoch (delayed injection).
    `stop_after_grok`: if set, stop this many epochs after the first epoch with
    test_acc >= grok_acc (off by default — experiments train to num_epochs).
    Scalars go to wandb (`step=epoch`, so charts populate live) and to JSONL;
    `wandb_group` should be the experiment name, `wandb_config` any sweep axes
    (n, seed, amp, ...) you want filterable in the wandb UI.
    Checkpoints (trainable weights only — the oracle is frozen and re-creatable
    from the run spec) land in `{run_dir}/checkpoints/{label}/ep{N:06d}.pth`
    after N epochs of training, for N in `ckpt_epochs` (None/() disables).
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
    wrun = _wandb_init(use_wandb, label, config, group=wandb_group,
                       extra_config=wandb_config,
                       inject_from_epoch=inject_from_epoch,
                       num_epochs=num_epochs, verbose=verbose)

    has_oracle = getattr(model, "oracle_fn", None) is not None
    ckpt_set = set(ckpt_epochs or ())
    # Per-epoch progress bar in verbose (sequential) mode; pool workers run
    # with verbose=False and report through the runner's run-level bar instead.
    pbar = tqdm(range(num_epochs), desc=label, unit="ep", miniters=500,
                mininterval=0) if verbose else None
    epochs = pbar if pbar is not None else range(num_epochs)
    t0 = time.time()
    try:
        for epoch in epochs:
            if has_oracle:
                model.inject = epoch >= inject_from_epoch

            logits = model(data["train_x"])[:, -1]
            loss = helpers.cross_entropy_high_precision(logits, data["train_y"])
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()

            if run_dir is not None and (epoch + 1) in ckpt_set:
                save_checkpoint(model, config, run_dir, label, epoch + 1,
                                inject_from_epoch=inject_from_epoch)

            final_epoch = epoch == num_epochs - 1
            stopping = (stop_after_grok is not None and grok_epoch is not None
                        and epoch >= grok_epoch + stop_after_grok)
            if epoch % eval_every == 0 or final_epoch or stopping:
                rec = evaluate(model, data, config, epoch)
                rec["lr"] = float(sched.get_last_lr()[0])
                rec["wall_s"] = round(time.time() - t0, 2)
                history.append(rec)
                if jsonl:
                    jsonl.write(json.dumps(rec) + "\n"); jsonl.flush()
                if wrun:
                    wrun.log(rec, step=epoch)
                if grok_epoch is None and rec["test_acc"] >= grok_acc:
                    grok_epoch = epoch
                if pbar is not None:
                    pbar.set_postfix(train=f"{rec['train_loss']:.4f}",
                                     test=f"{rec['test_loss']:.4f}",
                                     acc=f"{rec['test_acc']:.3f}",
                                     grok=grok_epoch)

            if snapshot_fn is not None and (epoch % snapshot_every == 0 or final_epoch or stopping):
                snap = snapshot_fn(model, epoch)
                snap["epoch"] = epoch
                snapshots.append(snap)
                if wrun:
                    wrun.log(_snapshot_scalars(snap), step=epoch)

            if stopping:
                break
    finally:
        if pbar is not None:
            pbar.close()
        if jsonl:
            jsonl.close()

    result = dict(history=history, snapshots=snapshots, grok_epoch=grok_epoch,
                  label=label, num_epochs=num_epochs,
                  config=_config_dict(config), inject_from_epoch=inject_from_epoch,
                  wall_s=round(time.time() - t0, 2), **(result_extra or {}))
    if run_dir is not None:
        with open(os.path.join(run_dir, f"{label}.result.json"), "w") as f:
            json.dump(result, f, indent=2, default=_json_default)
    if wrun:
        wrun.summary["grok_epoch"] = grok_epoch
        wrun.summary["final_test_acc"] = history[-1]["test_acc"] if history else None
        wrun.finish()
    if verbose:
        print(f"[{label}] done in {result['wall_s']}s  grok_epoch={grok_epoch}")
    return result


def save_checkpoint(model, config, run_dir, label, epochs_done, **meta):
    """Save trainable weights + config + metadata after `epochs_done` epochs."""
    d = os.path.join(run_dir, "checkpoints", label)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"ep{epochs_done:06d}.pth")
    t.save(dict(model=model.state_dict(), epochs_done=epochs_done, label=label,
                config=_config_dict(config), **meta), path)
    return path


def _snapshot_scalars(snap):
    """Flatten a snapshot to wandb-loggable scalars (skip lists/specs).

    Sums per-frequency lists (excluded loss, W_E power, logit coeffs at the
    injected freqs) into single trackable scalars; nested ablation dict becomes
    `snap/ablation_*`.
    """
    out = {}
    for k, v in snap.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"snap/{k}"] = v
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            if k in ("key_freqs", "injected_freqs", "injected_in_key_freqs"):
                out[f"snap/{k}_count"] = len(v)
            else:
                out[f"snap/{k}_sum"] = float(sum(v))
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    out[f"snap/{k}_{kk}"] = vv
    return out


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
