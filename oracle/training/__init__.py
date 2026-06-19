"""Training engines for the oracle testbed.

  harness          - per-model full-batch trainer (one model per call)
  stacked          - batched-ensemble primitives (many models as one stack)
  stacked_analysis - vectorized uptake snapshots over a stack
  grid             - ModelGrid: group specs into stacks + per-model fallback

Kept import-light on purpose: `oracle.sweep` imports `training.harness`, so this
package must not eagerly import `grid`/`stacked` (which import `sweep`) or it
would form an import cycle. Import the submodules directly, e.g.
`from modular_addition.oracle.training import grid`.
"""
