"""End-to-end ModelGrid check (CPU): partition logic + artifact compatibility.

Verifies (1) specs partition into the right stacked chunks vs per-model fallback
(unsupported architectures and singletons fall back), and (2) ModelGrid emits
result.json / jsonl / checkpoints that match the per-model sweep.execute path:
same keys, same history/snapshot cadence, matching grok_epoch, final losses
within fp tolerance, frozen weights held, early-stop truncation identical, and
checkpoints that load into OracleTransformer with strict=True.

Run: python -m modular_addition.oracle.tests.test_grid
"""
import json
import shutil
import tempfile
from pathlib import Path

import torch as t

from modular_addition.oracle import inject, sweep
from modular_addition.oracle.training import grid, harness, stacked


def small(label, **kw):
    base = dict(exp="g", p=23, d_model=32, num_epochs=40,
                eval_every=10, snapshot_every=20,
                ckpt_epochs=(10, 20, 40))
    base.update(kw)
    return sweep.spec(label=label, **base)


def test_partition():
    specs = [
        small("a", seed=0, oracle=dict(kind="fourier", freqs=[2, 5])),
        small("b", seed=1, oracle=dict(kind="fourier", freqs=[3])),
        small("c", seed=2, oracle=dict(kind="none")),
        # unsupported architecture -> fallback
        small("deep", seed=0, oracle=dict(kind="none"), config=dict(num_layers=2)),
        # unique shape (singleton) -> fallback
        small("big", seed=0, oracle=dict(kind="none"), d_model=64),
    ]
    g = grid.ModelGrid(specs, stack_size=2)
    n_stacked = sum(len(c[1]) for c in g.chunks)
    fb = {s["label"] for s in g.fallback}
    print(g.plan())
    assert "deep" in fb and "big" in fb, fb
    assert n_stacked == 3, n_stacked          # a,b,c stack (chunked by 2 -> 2 chunks)
    assert len(g.chunks) == 2, "stack_size=2 should split 3 into 2 chunks"
    print("✅ partition: unsupported + singleton fall back; group chunked.\n")


def run_harness(specs, d):
    sweep.RESULTS_DIR = Path(d)
    for s in specs:
        sweep.execute(s, device=t.device("cpu"), use_wandb=False, verbose=False)


def load(d, s):
    return json.load(open(Path(d) / s["exp"] / f"{s['label']}.result.json"))


def test_emission():
    specs = [
        small("none_s0", seed=0, oracle=dict(kind="none")),
        small("f_s1", seed=1, oracle=dict(kind="fourier", freqs=[2, 5])),
        small("frozen_s2", seed=2, oracle=dict(kind="fourier", freqs=[3]),
              freeze=["embed.W_E"]),
        small("delay_s3", seed=3, oracle=dict(kind="fourier", freqs=[2]),
              inject_from_epoch=15),
        # early stop: grok_acc=0 -> "groks" at epoch 0, stops 10 later
        small("stop_s0", seed=0, oracle=dict(kind="fourier", freqs=[5]),
              grok_acc=0.0, stop_after_grok=10),
    ]
    da, db = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        grid.ModelGrid(specs, stack_size=8).run(t.device("cpu"), results_dir=da,
                                                use_wandb=False, progress=False)
        run_harness(specs, db)
        worst = 0.0
        for s in specs:
            A, B = load(da, s), load(db, s)
            assert set(A) == set(B), f"{s['label']} top-level keys differ"
            assert [h["epoch"] for h in A["history"]] == [h["epoch"] for h in B["history"]], \
                f"{s['label']} history cadence differs"
            assert A["grok_epoch"] == B["grok_epoch"], \
                f"{s['label']} grok_epoch {A['grok_epoch']} vs {B['grok_epoch']}"
            assert A["frozen_params"] == B["frozen_params"], s["label"]
            assert [sn["epoch"] for sn in A["snapshots"]] == [sn["epoch"] for sn in B["snapshots"]], \
                f"{s['label']} snapshot cadence differs"
            fa, fb_ = A["history"][-1], B["history"][-1]
            for k in ("train_loss", "test_loss"):
                worst = max(worst, abs(fa[k] - fb_[k]) / (abs(fb_[k]) + 1e-9))
            # checkpoints exist at same epochs and load strict=True
            ca = sorted((Path(da) / s["exp"] / "checkpoints" / s["label"]).glob("*.pth"))
            cb = sorted((Path(db) / s["exp"] / "checkpoints" / s["label"]).glob("*.pth"))
            assert [p.name for p in ca] == [p.name for p in cb], \
                f"{s['label']} ckpt epochs differ: {[p.name for p in ca]} vs {[p.name for p in cb]}"
            ck = t.load(ca[-1], weights_only=False)
            cfg = sweep.make_config(seed=s["seed"], p=s["p"], d_model=s["d_model"])
            ofn, _ = sweep.build_oracle(s["oracle"], cfg)
            inject.OracleTransformer(cfg, oracle_fn=ofn).load_state_dict(
                ck["model"], strict=True)
        print(f"[emission] final-loss max rel Δ (stacked vs harness): {worst:.2e}")
        assert worst < 1e-3, worst
        print("✅ emission: keys, cadence, grok_epoch, freeze, early-stop, and "
              "checkpoints all match the per-model path.")
    finally:
        shutil.rmtree(da, ignore_errors=True); shutil.rmtree(db, ignore_errors=True)


def test_freeze_typo_raises():
    """A misspelled freeze pattern must fail loudly (like sweep.freeze_params),
    not silently train the 'frozen' weights."""
    specs = [small("a", seed=0, oracle=dict(kind="none")),
             small("typo", seed=1, oracle=dict(kind="none"),
                   freeze=["embed.w_e"])]   # wrong case -> matches nothing
    try:
        grid.ModelGrid(specs, stack_size=8).run(
            t.device("cpu"), results_dir=tempfile.mkdtemp(), use_wandb=False,
            progress=False)
    except ValueError as e:
        assert "matched no parameters" in str(e)
        print("✅ freeze typo raises ValueError (no silent mis-train).")
        return
    raise AssertionError("expected ValueError for non-matching freeze pattern")


if __name__ == "__main__":
    test_partition()
    test_emission()
    test_freeze_typo_raises()
