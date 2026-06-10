"""Oracle-feature injection mechanism.

The injection mirrors positional encodings (project_high_level.md): a *frozen*
(non-trainable) vector that is linearly ADDED into the residual stream right
after token embedding. The vector is a fixed function of the input tokens, so it
can carry precomputed Fourier features ("oracle frequencies") per token, or a
weak hint about the answer per example.

Design notes
------------
* `OracleTransformer` subclasses the project's `transformer.Transformer` and only
  overrides `forward` to add `oracle_fn(tokens)` after `self.embed`. Everything
  else (hook points, blocks, unembed, `cache_all`) is inherited, so all of the
  progress-measure analysis code keeps working unchanged.
* The oracle is held outside `nn.Parameter` (plain tensors captured in a closure),
  so it never receives gradient and is unaffected by weight decay — it is a true
  fixed feature. The trainable `W_E` lives alongside it; effective embedding is
  `W_E[:, n] + oracle`, which lets us watch whether `W_E` goes "lazy".
* `model.inject` gates injection on/off — used for (a) delayed injection (turn on
  at epoch T) and (b) ablation at inference (turn off, measure ΔCE).

Frequency convention matches `helpers.make_fourier_basis`: w_k = 2*pi*k/p.
"""
import math

import numpy as np
import torch as t

from modular_addition import transformer


class OracleTransformer(transformer.Transformer):
    """`transformer.Transformer` + a frozen additive oracle after token embedding.

    Parameters
    ----------
    config : transformer.Config
    oracle_fn : callable | None
        Maps a token tensor ``x`` of shape ``(batch, n_ctx)`` (long) to an additive
        residual term of shape ``(batch, n_ctx, d_model)``. ``None`` => no oracle
        (behaves exactly like the base Transformer, useful as a baseline).
    inject : bool
        Gate. When False the oracle is skipped even if ``oracle_fn`` is set.
    """

    def __init__(self, config: transformer.Config, oracle_fn=None, inject: bool = True):
        super().__init__(config, use_cache=False)
        self.oracle_fn = oracle_fn
        self.inject = inject

    def forward(self, x):
        h = self.embed(x)                       # (batch, n_ctx, d_model)
        if self.inject and self.oracle_fn is not None:
            h = h + self.oracle_fn(x)           # frozen, no grad flows into oracle
        h = self.pos_embed(h)
        for block in self.blocks:
            h = block(h)
        h = self.unembed(h)
        return h


# --------------------------------------------------------------------------- #
# Per-token Fourier oracle: O[n] = amp * [cos(w_k n), sin(w_k n) for k in freqs]
# --------------------------------------------------------------------------- #
def make_fourier_oracle(config: transformer.Config, freqs, amp: float = 1.0,
                        dims=None, device=None):
    """Frozen per-token-id oracle table supplying Fourier features.

    For each number token ``n in [0, p)`` and each frequency ``k in freqs`` we place
    ``amp*cos(w_k n)`` and ``amp*sin(w_k n)`` into two reserved residual dimensions.
    The "=" token (id ``p``) gets the zero vector. Per-token oracle norm is
    ``amp*sqrt(len(freqs))`` (each cos/sin pair is unit norm), so ``amp`` is the
    per-frequency amplitude / "intensity of the hint".

    Returns a callable ``fn(x)`` with attributes ``.table, .freqs, .dims, .amp``.
    """
    p, d_model, d_vocab = config.p, config.d_model, config.d_vocab
    device = device or config.device
    freqs = list(freqs)
    if dims is None:
        dims = list(range(2 * len(freqs)))
    assert len(dims) == 2 * len(freqs), "need exactly 2 dims (cos, sin) per frequency"
    assert max(dims) < d_model, "oracle dims exceed d_model"

    table = t.zeros(d_vocab, d_model, device=device)
    n = t.arange(p, device=device, dtype=t.float32)
    for idx, k in enumerate(freqs):
        ang = 2 * math.pi * k * n / p
        table[:p, dims[2 * idx]] = amp * t.cos(ang)
        table[:p, dims[2 * idx + 1]] = amp * t.sin(ang)

    def fn(x):
        return table[x]                          # (batch, n_ctx, d_model)

    fn.table, fn.freqs, fn.dims, fn.amp = table, freqs, dims, amp
    fn.kind = "fourier"
    return fn


# --------------------------------------------------------------------------- #
# Per-example variable-frequency oracle (for the "unreliable oracle" experiment)
# --------------------------------------------------------------------------- #
def _freq_map_to_tensor(freq_map, p, device):
    fm = t.as_tensor(np.asarray(freq_map), dtype=t.long, device=device)
    assert fm.shape == (p, p), f"freq_map must be (p, p) = ({p},{p})"
    return fm


def make_perexample_freq_oracle(config: transformer.Config, freq_map, amp: float = 1.0,
                                dims=(0, 1), device=None):
    """Oracle whose frequency varies *per example* (i, j).

    Example ``(i, j)`` uses frequency ``freq_map[i, j]`` for BOTH number tokens, so
    the number token at position 0 gets ``amp*[cos(w i), sin(w i)]`` and position 1
    gets ``amp*[cos(w j), sin(w j)]`` in ``dims`` (a single cos/sin pair). When
    ``freq_map`` is constant ``k0`` this reduces to ``make_fourier_oracle([k0])``
    on that pair; as ``freq_map`` becomes noisy the hint becomes unreliable.

    Returns ``fn(x)`` with attributes ``.freq_map, .dims, .amp``.
    """
    p, d_model = config.p, config.d_model
    device = device or config.device
    fm = _freq_map_to_tensor(freq_map, p, device)
    cd, sd = dims

    def fn(x):
        i = x[..., 0]                            # (batch,)
        j = x[..., 1]
        w = 2 * math.pi * fm[i, j].to(t.float32) / p
        out = t.zeros(*x.shape, d_model, device=x.device)
        out[..., 0, cd] = amp * t.cos(w * i.to(t.float32))
        out[..., 0, sd] = amp * t.sin(w * i.to(t.float32))
        out[..., 1, cd] = amp * t.cos(w * j.to(t.float32))
        out[..., 1, sd] = amp * t.sin(w * j.to(t.float32))
        return out

    fn.freq_map, fn.dims, fn.amp = fm, dims, amp
    fn.kind = "perexample_freq"
    return fn


def make_perexample_multifreq_oracle(config: transformer.Config, freq_maps,
                                     amp: float = 1.0, dims=None, device=None):
    """n independent per-example frequency maps; pair k uses dims (2k, 2k+1).

    Generalizes `make_perexample_freq_oracle` to multiple cos/sin pairs: example
    (i, j) gets, for each map k with frequency w = freq_maps[k][i, j],
    ``amp*[cos(w i), sin(w i)]`` at position 0 and the same in j at position 1.
    The full-grid lookup table (p, p, n_ctx, d_model) is precomputed once, so
    the per-step cost is a single gather instead of trig on every forward.
    """
    p, d_model, n_ctx = config.p, config.d_model, config.n_ctx
    device = device or config.device
    fms = [_freq_map_to_tensor(fm, p, device) for fm in freq_maps]
    n = len(fms)
    if dims is None:
        dims = list(range(2 * n))
    assert len(dims) == 2 * n, "need exactly 2 dims (cos, sin) per map"
    assert max(dims) < d_model, "oracle dims exceed d_model"

    ii = t.arange(p, device=device).view(p, 1).expand(p, p).to(t.float32)
    jj = t.arange(p, device=device).view(1, p).expand(p, p).to(t.float32)
    table = t.zeros(p, p, n_ctx, d_model, device=device)
    for k, fm in enumerate(fms):
        w = 2 * math.pi * fm.to(t.float32) / p                  # (p, p)
        table[:, :, 0, dims[2 * k]] = amp * t.cos(w * ii)
        table[:, :, 0, dims[2 * k + 1]] = amp * t.sin(w * ii)
        table[:, :, 1, dims[2 * k]] = amp * t.cos(w * jj)
        table[:, :, 1, dims[2 * k + 1]] = amp * t.sin(w * jj)

    def fn(x):
        return table[x[..., 0], x[..., 1]]                      # (batch, n_ctx, d_model)

    fn.freq_maps, fn.dims, fn.amp = fms, dims, amp
    fn.kind = "perexample_multifreq"
    return fn


def freq_map_reliable(config, base_freq: int):
    """(p, p) map that is constant `base_freq` everywhere (perfectly reliable)."""
    p = config.p
    return np.full((p, p), int(base_freq), dtype=np.int64)


def freq_map_corrupt(config, base_freq: int, reliability: float, seed: int = 0):
    """`base_freq` with prob `reliability`, else a uniform random freq in [1, p//2].

    reliability=1.0 -> perfectly reliable; 0.0 -> pure noise.
    """
    p = config.p
    rng = np.random.RandomState(seed)
    fm = np.full((p, p), int(base_freq), dtype=np.int64)
    flip = rng.rand(p, p) >= reliability
    rand_freqs = rng.randint(1, p // 2 + 1, size=(p, p))
    fm[flip] = rand_freqs[flip]
    return fm


def freq_map_jitter(config, base_freq: int, sigma: float, seed: int = 0):
    """`base_freq` + round(sigma * N(0,1)) per example, clipped to [1, p//2].

    sigma=0 -> reliable; larger sigma -> frequency varies more across examples.
    """
    p = config.p
    rng = np.random.RandomState(seed)
    noise = np.round(sigma * rng.randn(p, p)).astype(np.int64)
    fm = np.clip(int(base_freq) + noise, 1, p // 2)
    return fm.astype(np.int64)


# --------------------------------------------------------------------------- #
# Per-example answer-hint oracle: weak side info about c = (i + j) mod p
# --------------------------------------------------------------------------- #
def make_answer_hint_oracle(config: transformer.Config, hint: str = "mod", modulus: int = 10,
                            amp: float = 1.0, code: str = "onehot", dims=None,
                            pos: int = 2, device=None):
    """Inject a weak feature about the answer ``c = (i + j) mod p`` at position ``pos``
    (default the "=" token, which is also the read-out position).

    hint : 'mod' -> h = c % modulus  (narrows c to ~p/modulus candidates)
           'div' -> h = c // modulus
    code : 'onehot' -> h encoded as amp * e_h in `n_classes` dims (strong, explicit)
           'fourier'-> h encoded as amp * [cos(2*pi*h/n_classes), sin(...)] in 2 dims (weaker)

    Returns ``fn(x)`` with attributes ``.codes, .n_classes, .dims, .hint, .modulus``.
    """
    p, d_model = config.p, config.d_model
    device = device or config.device

    if hint == "mod":
        n_classes = modulus
        def hfn(c):
            return c % modulus
    elif hint == "div":
        n_classes = (p + modulus - 1) // modulus
        def hfn(c):
            return c // modulus
    else:
        raise ValueError(f"unknown hint {hint!r}")

    if code == "onehot":
        if dims is None:
            dims = list(range(n_classes))
        assert len(dims) == n_classes
        codes = amp * t.eye(n_classes, device=device)          # (n_classes, n_classes)
    elif code == "fourier":
        if dims is None:
            dims = [0, 1]
        assert len(dims) == 2
        h = t.arange(n_classes, device=device, dtype=t.float32)
        ang = 2 * math.pi * h / n_classes
        codes = amp * t.stack([t.cos(ang), t.sin(ang)], dim=1)  # (n_classes, 2)
    else:
        raise ValueError(f"unknown code {code!r}")

    dims_t = t.as_tensor(dims, dtype=t.long, device=device)

    def fn(x):
        i = x[..., 0]
        j = x[..., 1]
        c = (i + j) % p
        h = hfn(c)                                              # (batch,)
        out = t.zeros(*x.shape, d_model, device=x.device)
        out[..., pos, dims_t] = codes[h]                        # place hint code at `pos`
        return out

    fn.codes, fn.n_classes, fn.dims = codes, n_classes, dims
    fn.hint, fn.modulus, fn.pos, fn.code = hint, modulus, pos, code
    fn.kind = "answer_hint"
    return fn
