"""CPU equivalence tests for the stacked-training stack.

  test_stacked - init/forward/training + vectorized-snapshot equivalence vs the
                 per-model path (stacked.py, stacked_analysis.py)
  test_grid    - ModelGrid partition + artifact compatibility vs sweep.execute

Run: python -m modular_addition.oracle.tests.test_stacked
     python -m modular_addition.oracle.tests.test_grid
"""
