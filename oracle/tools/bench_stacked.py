"""GPU benchmark: stacked training vs the per-model path (run on the H100).

Measures the lever that matters — training throughput (model·epochs / second) —
for the current per-model loop vs the stacked loop, sweeps the stack size to
find where the GPU saturates, reports peak VRAM, and re-runs the bit-equivalence
check on the real CUDA kernels.

Usage (on the GPU box, repo root):
  python -m modular_addition.oracle.tools.bench_stacked                  # defaults
  python -m modular_addition.oracle.tools.bench_stacked --epochs 3000 \
      --stack-sizes 8,16,32,64,128 --p 113 --d-model 128
  python -m modular_addition.oracle.tools.bench_stacked --equiv-only     # just correctness

The training loop here is bare (no snapshots / wandb / jsonl) so it isolates the
raw training speedup; uptake snapshots are a separate cost addressed elsewhere.
"""
import argparse
import dataclasses
import time

import torch as t
import torch.optim as optim

from modular_addition import helpers
from modular_addition.oracle import sweep
from modular_addition.oracle.training import harness, stacked


def make_specs(p, d_model, n_models, epochs):
    """A co-stackable fourier grid (exp01-shaped): vary n (freq count) & seed."""
    n_list = [0, 1, 2, 3, 5, 6, 8]
    specs = []
    s = 0
    while len(specs) < n_models:
        n = n_list[len(specs) % len(n_list)]
        seed = s
        freqs = sweep.pick_freqs(n, p=p)
        oracle = dict(kind="fourier", freqs=freqs, amp=1.0) if n else dict(kind="none")
        specs.append(sweep.spec(exp="bench", label=f"n{n}_s{seed}", seed=seed,
                                oracle=oracle, p=p, d_model=d_model,
                                num_epochs=epochs))
        if len(specs) % len(n_list) == 0:
            s += 1
    return specs[:n_models]


def train_one_bare(spec, cfg, epochs):
    """harness.train's inner loop with no logging/snapshots — pure training."""
    oracle_fn, _ = sweep.build_oracle(spec.get("oracle"), cfg)
    model, data = harness.setup(cfg, oracle_fn=oracle_fn)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    sched = optim.lr_scheduler.LambdaLR(opt, lambda step: min(step / 10, 1))
    has_oracle = oracle_fn is not None
    for epoch in range(epochs):
        if has_oracle:
            model.inject = True
        logits = model(data["train_x"])[:, -1]
        loss = helpers.cross_entropy_high_precision(logits, data["train_y"])
        loss.backward(); opt.step(); sched.step(); opt.zero_grad()


def sync(dev):
    if dev.type == "cuda":
        t.cuda.synchronize()


def vram_peak_gb(dev):
    return t.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0.0


def bench(args):
    dev = t.device(args.device)
    sizes = [int(x) for x in args.stack_sizes.split(",")]
    biggest = max(sizes)
    print(f"device={dev}  p={args.p}  d_model={args.d_model}  epochs={args.epochs}")
    print(f"train batch (frac_train=0.3): ~{int(0.3 * args.p * args.p)} examples\n")

    # ---- per-model sequential baseline (subset, then extrapolate) ---------- #
    probe = make_specs(args.p, args.d_model, min(biggest, args.seq_probe), args.epochs)
    cfgs = [dataclasses.replace(stacked.make_base_config(s, dev), seed=s["seed"])
            for s in probe]
    train_one_bare(probe[0], cfgs[0], 50)              # warmup / cudnn autotune
    sync(dev)
    if dev.type == "cuda":
        t.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    for s, c in zip(probe, cfgs):
        train_one_bare(s, c, args.epochs)
    sync(dev)
    seq_dt = time.time() - t0
    seq_per_model = seq_dt / len(probe)
    seq_thru = len(probe) * args.epochs / seq_dt
    print(f"[per-model sequential] {len(probe)} models in {seq_dt:.1f}s  "
          f"= {seq_per_model:.2f}s/model  ({seq_thru:,.0f} model·ep/s)  "
          f"peak {vram_peak_gb(dev):.2f} GB")
    print("  (their production pool runs this ~2-3x faster via worker parallelism)\n")

    # ---- stacked, swept over stack size ----------------------------------- #
    print("[stacked] stack size sweep:")
    best = None
    for M in sizes:
        specs = make_specs(args.p, args.d_model, M, args.epochs)
        if dev.type == "cuda":
            t.cuda.reset_peak_memory_stats(dev)
        sync(dev); t0 = time.time()
        stacked.train_group(specs, dev, eval_every=args.epochs)  # eval only at end
        sync(dev)
        dt = time.time() - t0
        thru = M * args.epochs / dt
        speedup = seq_per_model / (dt / M)
        vram = vram_peak_gb(dev)
        print(f"  M={M:>4}  {dt:6.1f}s  = {dt / M:6.3f}s/model  "
              f"({thru:>9,.0f} model·ep/s)  {speedup:5.1f}x vs seq  peak {vram:5.2f} GB")
        if best is None or thru > best[1]:
            best = (M, thru, speedup)
    print(f"\n  best throughput at M={best[0]}: {best[2]:.1f}x vs per-model "
          f"sequential (~{best[2] / 2.5:.1f}x vs a 2.5x worker pool)")


def equiv(args):
    """Re-run the init/forward/training equivalence on the real device."""
    dev = t.device(args.device)
    P, D, EP = args.p, args.d_model, 60
    specs = [
        sweep.spec(exp="eq", label="none_s0", seed=0, oracle=dict(kind="none"),
                   p=P, d_model=D, num_epochs=EP),
        sweep.spec(exp="eq", label="f_s1", seed=1,
                   oracle=dict(kind="fourier", freqs=sweep.pick_freqs(3, p=P), amp=1.0),
                   p=P, d_model=D, num_epochs=EP),
    ]
    cfgs = [dataclasses.replace(stacked.make_base_config(s, dev), seed=s["seed"])
            for s in specs]
    # reference per-model post-step train losses
    refs = []
    for s, c in zip(specs, cfgs):
        oracle_fn, _ = sweep.build_oracle(s.get("oracle"), c)
        model, data = harness.setup(c, oracle_fn=oracle_fn)
        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=c.lr, weight_decay=c.weight_decay, betas=(0.9, 0.98))
        sched = optim.lr_scheduler.LambdaLR(opt, lambda st: min(st / 10, 1))
        ls = []
        for ep in range(EP):
            if oracle_fn is not None:
                model.inject = True
            lo = model(data["train_x"])[:, -1]
            loss = helpers.cross_entropy_high_precision(lo, data["train_y"])
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            with t.no_grad():
                ls.append(helpers.cross_entropy_high_precision(
                    model(data["train_x"])[:, -1], data["train_y"]).item())
        refs.append(ls)
    out = stacked.train_group(specs, dev, eval_every=1)
    mx = max(abs(h["train_loss"] - r) / (abs(r) + 1e-9)
             for i, r_list in enumerate(refs)
             for h, r in zip(out["histories"][i], r_list))
    print(f"[equiv on {dev}] max relative train-loss Δ over {EP} epochs: {mx:.2e}")
    print("  PASS" if mx < 1e-3 else "  FAIL — investigate kernel nondeterminism")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--stack-sizes", default="8,16,32,64,128")
    ap.add_argument("--seq-probe", type=int, default=8,
                    help="how many models to time on the per-model path")
    ap.add_argument("--equiv-only", action="store_true")
    args = ap.parse_args()
    equiv(args)
    if not args.equiv_only:
        print()
        bench(args)


if __name__ == "__main__":
    main()
