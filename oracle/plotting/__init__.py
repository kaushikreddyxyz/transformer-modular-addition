"""Figure generation for the oracle experiments.

  plot_common / plot_suite    - shared styling + the curated subplot suite
  plot_exp*                    - per-experiment figure scripts
  plot_baseline_spectrum       - baseline W_E spectrum figure
  make_figures                 - render every available figure

These are run as scripts (each adds its own dir to sys.path for sibling
imports), e.g. `python -m modular_addition.oracle.plotting.make_figures` or
`python modular_addition/oracle/plotting/plot_exp01.py [results_dir]`.
"""
