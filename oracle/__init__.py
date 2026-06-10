"""Oracle-feature injection for the modular-addition grokking testbed.

We plant precomputed linear "oracle features" (Fourier cos/sin at chosen
frequencies — which for Z/p ARE the irreducible representations) into the
residual stream and test whether a gradient-trained model uses them.

Submodules:
  inject   - OracleTransformer + oracle constructors (frozen additive features)
  analysis - uptake detectors (W_E Fourier spectrum, ablation ΔCE, reused
             progress measures: key_freqs / excluded / trig / coefficients)
  harness  - fast, reproducible, wandb-free trainer with JSONL logging
"""
from modular_addition.oracle.inject import (
    OracleTransformer,
    make_fourier_oracle,
    make_perexample_freq_oracle,
    freq_map_reliable,
    freq_map_corrupt,
    freq_map_jitter,
    make_answer_hint_oracle,
)
from modular_addition.oracle import analysis, harness

__all__ = [
    "OracleTransformer",
    "make_fourier_oracle",
    "make_perexample_freq_oracle",
    "freq_map_reliable",
    "freq_map_corrupt",
    "freq_map_jitter",
    "make_answer_hint_oracle",
    "analysis",
    "harness",
]
