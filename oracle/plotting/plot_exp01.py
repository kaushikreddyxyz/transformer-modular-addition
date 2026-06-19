"""Exp01 figures — the curated suite (see plot_suite.py) on the exp01 model
(p=113, frac_train=0.3, 30k ep; n-sweep [0,1,2,3,5,6,8]).

Same clean, subplot-only suite used for exp06, so figures/exp01 and
figures/exp06 are directly comparable. Run:
    .venv/bin/python modular_addition/oracle/experiments/plot_exp01.py [results_dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_suite import build  # noqa: E402

DEFAULT_RES = (Path(__file__).resolve().parents[1] / "results" /
               "run_20260612_200000")

if __name__ == "__main__":
    res = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_RES)
    build(res, "exp01", "Exp01")
