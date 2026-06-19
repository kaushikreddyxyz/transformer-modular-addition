"""Backfill the W_L (neuron-logit-map) Fourier spectrum into result.json final
snapshots, for runs trained before analysis.py recorded the wl_* fields.

W_L = W_out^T W_U is a pure-weight quantity, so it is recomputed *exactly* from
each cell's final saved checkpoint -- no retraining. Adds the same fields that
analysis.record() now writes (wl_total_norm, wl_gini, wl_freq_power_injected /
_top / _full). Additive + idempotent: a cell that already has wl_freq_power_full
is skipped unless --force.

Usage:  python backfill_wl.py RESULTS_DIR EXP [--force]
   e.g. python backfill_wl.py .../results/run_20260616_110846 exp08
"""
import sys
import os
import re
import json
import glob
import dataclasses
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root on path
from modular_addition import transformer                       # noqa: E402
from modular_addition.oracle.inject import OracleTransformer    # noqa: E402
from modular_addition.oracle import analysis                    # noqa: E402


def _cfg(cfg_dict):
    """Rebuild the (frozen) Config from a result.json config dict, on CPU."""
    fields = {f.name for f in dataclasses.fields(transformer.Config)}
    kw = {k: v for k, v in cfg_dict.items() if k in fields and k != "device"}
    return transformer.Config(**kw, device=t.device("cpu"))


def _final_ckpt(run_dir, label):
    cks = glob.glob(os.path.join(run_dir, "checkpoints", label, "ep*.pth"))
    if not cks:
        return None
    return max(cks, key=lambda p: int(re.search(r"ep(\d+)\.pth", p).group(1)))


def backfill(results_dir, exp, force=False):
    rjs = sorted(glob.glob(os.path.join(results_dir, exp, "*.result.json")))
    if not rjs:
        raise SystemExit(f"no *.result.json in {os.path.join(results_dir, exp)}")
    done = skipped = no_ckpt = 0
    for rj in rjs:
        r = json.load(open(rj))
        snaps = r.get("snapshots") or []
        if not snaps:
            print(f"  no snapshots: {os.path.basename(rj)}"); continue
        fs = snaps[-1]
        if "wl_freq_power_full" in fs and not force:
            skipped += 1; continue
        label = r.get("label") or os.path.basename(rj).replace(".result.json", "")
        ck = _final_ckpt(os.path.dirname(rj), label)
        if ck is None:
            print(f"  no checkpoint for {label}"); no_ckpt += 1; continue

        cfg = _cfg(r["config"])
        model = OracleTransformer(cfg, oracle_fn=None, inject=False).eval()
        sd = t.load(ck, map_location="cpu")
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(sd, strict=True)
        with t.no_grad():
            wl = analysis.wl_fourier_power(model, cfg)

        fp = wl["freq_power"]
        inj = fs.get("injected_freqs") or []
        fs["wl_total_norm"] = float(wl["total_norm"])
        fs["wl_gini"] = float(wl["gini"])
        fs["wl_freq_power_injected"] = [float(fp[k - 1]) for k in inj] if inj else []
        fs["wl_freq_power_top"] = analysis._top_freqs(fp, 6)
        fs["wl_freq_power_full"] = fp.tolist()

        json.dump(r, open(rj, "w"), indent=2)
        done += 1
        print(f"  backfilled {label:14s} from {os.path.basename(ck)}")
    print(f"done: {done} backfilled, {skipped} already had wl_*, {no_ckpt} missing ckpt")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    if len(args) < 2:
        raise SystemExit(__doc__)
    backfill(args[0], args[1], force=force)
