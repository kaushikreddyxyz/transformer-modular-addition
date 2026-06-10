"""Shared sweep machinery for oracle experiments.

Every experiment is a *grid of run specs* (plain JSON-serializable dicts), so
the same spec can be executed sequentially in a notebook cell, farmed out by
`experiments/runner.py` across GPU workers, or re-created later to interpret a
HuggingFace checkpoint. `execute()` is idempotent: a spec whose result.json
already exists is skipped, which makes sweeps resumable.

Standard axes (project-wide):
  SEEDS  — 4 seeds per configuration
  N_LIST — number of injected oracle frequency pairs; 0 = no-oracle baseline
"""
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch as t

from modular_addition import transformer
from modular_addition.oracle import analysis, harness, inject

SEEDS = [0, 1, 2, 3]
N_LIST = [0, 1, 2, 3, 5, 6, 8]          # oracle frequency pairs; 0 = baseline
NUM_EPOCHS = 30_000
EVAL_EVERY = 200
SNAPSHOT_EVERY = 2000
AMP_DEFAULT = 1.0

# Fresh sweep output lands here (created on first run); pre-sweep results were
# archived to results_legacy/. Override with ORACLE_RESULTS_DIR (e.g. scratch
# disk on a rented GPU box) — make_figures and push_to_hf follow it too.
RESULTS_DIR = Path(os.environ.get("ORACLE_RESULTS_DIR")
                   or Path(__file__).resolve().parent / "results")

# Canonical p=113 frequency pool. Prefixes are nested (n=2 ⊂ n=3 ⊂ ...) so the
# n-sweep compares supersets; the first two are exp01's historical [17, 34].
FREQ_POOL_113 = [17, 34, 9, 25, 43, 50, 13, 47]


def pick_freqs(n, p=113, pool_seed=0):
    """First `n` freqs of a deterministic nested pool for modulus `p`.

    For p=113 this is the canonical pool above (continuity with the original
    exp01/exp02 runs). For any other p, a seeded permutation of [1, p//2] —
    still nested across n, still identical across model seeds.
    """
    if n == 0:
        return []
    if p == 113 and n <= len(FREQ_POOL_113):
        return FREQ_POOL_113[:n]
    rng = np.random.RandomState(pool_seed)
    perm = rng.permutation(np.arange(1, p // 2 + 1))
    assert n <= len(perm), f"n={n} exceeds available freqs for p={p}"
    return [int(x) for x in perm[:n]]


def pick_device():
    if t.cuda.is_available():
        return t.device("cuda")
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")


def make_config(*, seed=0, p=113, d_model=128, num_epochs=NUM_EPOCHS,
                device=None, **overrides):
    """Config with d_vocab/d_mlp wired to p/d_model.

    (The Config dataclass defaults d_vocab=p+1 and d_mlp=4*d_model at *class
    definition time*, so they do NOT follow p/d_model through
    dataclasses.replace — they must be set explicitly.)
    """
    device = device or pick_device()
    return dataclasses.replace(
        transformer.Config(),
        p=p, d_vocab=p + 1, d_model=d_model, d_mlp=4 * d_model,
        seed=seed, num_epochs=num_epochs, device=device, save_models=False,
        **overrides)


# --------------------------------------------------------------------------- #
# Oracle factory: JSON spec -> (oracle_fn, injected_freqs)
# --------------------------------------------------------------------------- #
def build_oracle(ospec, cfg):
    """Build an oracle from its JSON spec; returns (oracle_fn, injected_freqs).

    kinds:
      none                — baseline
      fourier             — frozen per-token Fourier features at fixed freqs
      perexample_corrupt  — per-example freqs: base freq with prob `reliability`,
                            else uniform random (one independently corrupted map
                            per pair); `map_seed` controls the corruption draw
      answer_hint         — weak hint about c=(i+j)%p at the "=" position
    """
    kind = (ospec or {}).get("kind", "none")
    if kind == "none":
        return None, []
    amp = ospec.get("amp", AMP_DEFAULT)
    if kind == "fourier":
        freqs = ospec["freqs"]
        return inject.make_fourier_oracle(cfg, freqs, amp=amp), list(freqs)
    if kind == "perexample_corrupt":
        base_freqs = ospec["freqs"]
        rel = ospec["reliability"]
        map_seed = ospec.get("map_seed", 0)
        maps = [inject.freq_map_corrupt(cfg, f, reliability=rel,
                                        seed=map_seed * 101 + k)
                for k, f in enumerate(base_freqs)]
        return (inject.make_perexample_multifreq_oracle(cfg, maps, amp=amp),
                list(base_freqs))
    if kind == "answer_hint":
        orc = inject.make_answer_hint_oracle(
            cfg, hint=ospec["hint"], modulus=ospec.get("modulus", 10),
            amp=amp, code=ospec.get("code", "onehot"))
        return orc, []
    raise ValueError(f"unknown oracle kind {kind!r}")


# --------------------------------------------------------------------------- #
# Run specs
# --------------------------------------------------------------------------- #
def spec(*, exp, label, seed, oracle=None, axes=None, p=113, d_model=128,
         num_epochs=NUM_EPOCHS, inject_from_epoch=0, eval_every=EVAL_EVERY,
         snapshot_every=SNAPSHOT_EVERY, snapshots=True, ckpt_epochs=None,
         config=None):
    """A JSON-serializable description of one training run."""
    return dict(exp=exp, label=label, seed=seed,
                oracle=oracle or dict(kind="none"), axes=axes or {},
                p=p, d_model=d_model, num_epochs=num_epochs,
                inject_from_epoch=inject_from_epoch, eval_every=eval_every,
                snapshot_every=snapshot_every, snapshots=snapshots,
                ckpt_epochs=ckpt_epochs, config=config or {})


def result_path(s):
    return RESULTS_DIR / s["exp"] / f"{s['label']}.result.json"


def execute(s, device=None, use_wandb=True, force=False, verbose=True):
    """Run one spec end-to-end (idempotent). Returns the result dict."""
    rpath = result_path(s)
    if rpath.exists() and not force:
        if verbose:
            print(f"[{s['exp']}/{s['label']}] SKIP (result exists)")
        return json.load(open(rpath))

    cfg = make_config(seed=s["seed"], p=s["p"], d_model=s["d_model"],
                      num_epochs=s["num_epochs"], device=device,
                      **s.get("config", {}))
    oracle_fn, injected = build_oracle(s["oracle"], cfg)
    model, data = harness.setup(cfg, oracle_fn=oracle_fn)

    snapshot_fn = None
    if s.get("snapshots", True):
        ctx = analysis.metric_context(cfg, data["train_pairs"])

        def _snap(m, _epoch, _inj=injected, _d=data):
            return analysis.uptake_snapshot(m, cfg, ctx, injected_freqs=_inj,
                                            data=_d)
        snapshot_fn = _snap

    run_dir = str(RESULTS_DIR / s["exp"])
    ckpts = s.get("ckpt_epochs") or harness.CKPT_EPOCHS
    res = harness.train(
        cfg, model, data, num_epochs=s["num_epochs"],
        eval_every=s.get("eval_every", EVAL_EVERY),
        snapshot_every=s.get("snapshot_every", SNAPSHOT_EVERY),
        snapshot_fn=snapshot_fn, inject_from_epoch=s.get("inject_from_epoch", 0),
        run_dir=run_dir, label=s["label"], verbose=verbose,
        use_wandb=use_wandb, wandb_group=s["exp"],
        wandb_config=dict(s.get("axes", {})), ckpt_epochs=ckpts,
        result_extra=dict(spec=s, injected_freqs=injected))
    return res


def run_all(specs, use_wandb=True, force=False, verbose=True):
    """Sequential fallback executor (notebook cells / single GPU)."""
    from tqdm.auto import tqdm
    out = []
    grid = tqdm(specs, desc="grid", unit="run", disable=len(specs) < 2)
    for s in grid:
        grid.set_postfix_str(f"{s['exp']}/{s['label']}")
        out.append(execute(s, use_wandb=use_wandb, force=force, verbose=verbose))
    return out


# --------------------------------------------------------------------------- #
# Aggregation helpers (used by summaries and make_figures)
# --------------------------------------------------------------------------- #
def final_record(res):
    """Headline per-run record from a result dict (snapshots optional)."""
    s = (res.get("snapshots") or [{}])[-1]
    h = res["history"][-1]
    abl = s.get("ablation_test") or {}
    spec_ = res.get("spec", {})
    rec = dict(label=res["label"], grok_epoch=res["grok_epoch"],
               final_test_acc=h["test_acc"], final_train_acc=h["train_acc"],
               final_test_loss=h["test_loss"], we_norm=h.get("we_norm"),
               **{f"ax_{k}": v for k, v in spec_.get("axes", {}).items()})
    if s:
        inj = res.get("injected_freqs") or s.get("injected_freqs") or []
        rec.update(
            n_key_freqs=len(s.get("key_freqs", [])),
            key_freqs=s.get("key_freqs"),
            injected_in_key=len(s.get("injected_in_key_freqs", [])),
            n_injected=len(inj),
            ablation_delta=abl.get("delta"),
            acc_on=abl.get("acc_on"), acc_off=abl.get("acc_off"),
            we_total_norm=s.get("we_total_norm"), we_gini=s.get("we_gini"),
            we_power_injected=(float(sum(s["we_freq_power_injected"]))
                               if s.get("we_freq_power_injected") else 0.0))
    return rec


def mean_std(records, keys, group_keys):
    """Group records by `group_keys` (e.g. axes minus seed) and aggregate.

    Returns {group_tuple: {key: {mean, std, n, vals}}} for numeric `keys`;
    None values are dropped per-key (e.g. grok_epoch on runs that never grok).
    """
    groups = {}
    for r in records:
        g = tuple(r.get(k) for k in group_keys)
        groups.setdefault(g, []).append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        agg: dict = {"_n_runs": len(rs)}
        for k in keys:
            vals = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            agg[k] = (dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                           n=len(vals), vals=vals) if vals else None)
        out[g] = agg
    return out


def fmt_stat(stat, prec=0, plus=False):
    """'mean±std (n=k)' for a mean_std() entry; '—' for missing."""
    if not stat:
        return "—"
    sign = "+" if plus else ""
    return f"{stat['mean']:{sign}.{prec}f}±{stat['std']:.{prec}f} (n={stat['n']})"


def write_summary(exp, payload):
    d = RESULTS_DIR / exp
    os.makedirs(d, exist_ok=True)
    with open(d / "summary.json", "w") as f:
        json.dump(payload, f, indent=2, default=_default)
    print(f"[{exp}] summary.json written")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
