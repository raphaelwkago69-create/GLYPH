"""
==============================================================================
INT-INFER -- integer-only GPT-2 fingerprint engine (protocol v4)
==============================================================================
Every arithmetic operation in the consensus path is EXACT integer math:

  * Activations are int64 tensors holding Q16.16 fixed-point values.
  * Weights are converted once (deterministically, on CPU) to Q12 integers.
  * Matrix multiplies run in float64 where every intermediate partial sum is
    a bounded exact integer (< 2^53). IEEE-754 makes every such operation
    exact, so the result is identical on any chip, any accumulation order,
    any BLAS. Bounds are enforced by explicit clamps (see BOUNDS below).
  * LayerNorm, softmax and GELU are integer re-implementations built from
    add / mul / floor-div / compare / hardcoded integer constants only.
    No libm (exp/tanh/sqrt) output ever enters the consensus path.

Result: inference_hash(prompt, salt) is bit-identical on CPU and GPU
*by construction*, not empirically. This is the fix for the v3 block-1693
cross-hardware boundary flip.

BOUNDS (why nothing overflows and every float64 op is exact):
  activations clamped to |v| <= 4096      -> Q16 |q| <= 2^28, Q12 |q| <= 2^24
  q/k head vectors clamped to |v| <= 512  -> Q12 |q| <= 2^21
  weights clamped to |w| <= 16            -> Q12 |q| <= 2^16
  linear (K<=1024 per chunk): |sum| <= 1024 * 2^24 * 2^16 = 2^50 < 2^53  OK
  qk^T   (K=64):              |sum| <= 64 * 2^21 * 2^21   = 2^48 < 2^53  OK
  probs@V (K=T<=512):         |sum| <= 512 * 2^16 * 2^24  = 2^49 < 2^53  OK

Research written and developed by Claude Fable 5.
==============================================================================
"""
import hashlib, json, math, os

F = 16                     # fixed-point fractional bits for activations (Q16)
ONE = 1 << F               # 65536
WF = 12                    # fractional bits for weights (Q12)
ACT_CLAMP = 4096 * ONE     # activation magnitude bound (value 4096.0)
QK_CLAMP = 512 * ONE       # q/k head-vector bound (value 512.0)
MASK_NEG = -(1 << 40)      # "minus infinity" for causal masking
K_CHUNK = 1024             # max contraction length per exact float64 matmul

# hardcoded Q16 integer constants -- never computed via libm at runtime
LOG2E_Q16 = 94548          # round(log2(e) * 65536)
SQRT_2_OVER_PI_Q16 = 52293 # round(sqrt(2/pi) * 65536)
GELU_C_Q16 = 2931          # round(0.044715 * 65536)
# cubic Hermite for 2^f on f in [0,1): p(0)=1, p'(0)=ln2, p(1)=2, p'(1)=2ln2
# p(f) = 1 + c1 f + c2 f^2 + c3 f^3   (max abs error ~1.2e-3)
EXP2_C1_Q16 = 45426        # round(ln(2) * 65536)
EXP2_C2_Q16 = 14904        # round((1 - ln2 - (3 ln2 - 2)) * 65536)
EXP2_C3_Q16 = 5207         # round((3 ln2 - 2) * 65536)

GRID = 100                 # fingerprint quantization grid (protocol constant)

_W = None                  # converted integer weights (consensus, hashed)
_LNF = None                # final layernorm, generation only -- NOT in WEIGHTS_HASH
_tok = None
_device = None
_torch = None
WEIGHTS_HASH = None        # sha256 over the converted integer weights

# ------------------------------------------------------------ helpers ------

def _fdiv(a, b):
    return _torch.div(a, b, rounding_mode='floor')

def _rshift_round(x, s):
    # round-half-up in floor semantics: floor((x + 2^(s-1)) / 2^s). Exact.
    return _torch.div(x + (1 << (s - 1)), 1 << s, rounding_mode='floor')

def _matmul_exact(a_int, w_int, out_shift, bias=None):
    """a_int @ w_int with every float64 partial sum an exact integer.
    Contraction is chunked to K_CHUNK so the bound in BOUNDS holds for any
    accumulation order a BLAS might choose. Returns int64, >> out_shift."""
    K = a_int.shape[-1]
    acc = None
    for k0 in range(0, K, K_CHUNK):
        part = (a_int[..., k0:k0 + K_CHUNK].double()
                @ w_int[k0:k0 + K_CHUNK].double()).long()
        acc = part if acc is None else acc + part
    y = _rshift_round(acc, out_shift)
    if bias is not None:
        y = y + bias
    return y

def _isqrt(v):
    """Integer sqrt of an int64 tensor. float64 sqrt is correctly rounded
    (IEEE-754 requirement), so it lands within 1; integer compares fix it."""
    s = _torch.sqrt(v.double()).long()
    for _ in range(2):
        s = _torch.where(s * s > v, s - 1, s)
        s = _torch.where((s + 1) * (s + 1) <= v, s + 1, s)
    return s

def _exp_q16(d):
    """exp(d) for d <= 0 (Q16 in, Q16 out). Integer-only:
    exp(d) = 2^(d*log2e) = 2^n * 2^f, poly for 2^f, shift for 2^n."""
    t = _fdiv(d * LOG2E_Q16, ONE)          # Q16 base-2 exponent, <= 0
    n = _fdiv(t, ONE)                      # integer part (floor, <= 0)
    f = t - n * ONE                        # fractional part in [0, ONE)
    p = _fdiv(EXP2_C3_Q16 * f, ONE) + EXP2_C2_Q16
    p = _fdiv(p * f, ONE) + EXP2_C1_Q16
    p = _fdiv(p * f, ONE) + ONE            # 2^f in Q16, in [ONE, 2*ONE]
    sh = _torch.clamp(-n, 0, 62)
    out = p >> sh                          # p > 0, so >> is exact floor /2^sh
    return _torch.where(sh > 40, _torch.zeros_like(out), out)

def _tanh_q16(z):
    """tanh(z), Q16 in/out, via exp of a non-positive argument only."""
    za = _torch.clamp(z.abs(), max=6 * ONE)          # tanh saturates by 6
    e = _exp_q16(-2 * za)                            # e^{-2|z|} in (0, ONE]
    t = _fdiv((ONE - e) * ONE, ONE + e)              # tanh(|z|) in Q16
    return _torch.where(z < 0, -t, t)

# ------------------------------------------------------------ layers -------

def _layernorm(x, gamma_q12, beta_q16):
    mean = _fdiv(x.sum(-1, keepdim=True), x.shape[-1])
    c = x - mean
    c8 = _rshift_round(c, 8)                          # Q8 for the square
    var16 = _fdiv((c8 * c8).sum(-1, keepdim=True), x.shape[-1])  # Q16
    std8 = _torch.clamp(_isqrt(var16), min=1)         # Q8 std
    norm = _fdiv(c * 256, std8)                       # Q16 normalized
    y = _rshift_round(norm * gamma_q12, WF) + beta_q16
    return _torch.clamp(y, -ACT_CLAMP, ACT_CLAMP)

def _gelu(x):
    # gelu_new: 0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))
    xc = _torch.clamp(x, -6 * ONE, 6 * ONE)           # poly region
    x2 = _fdiv(xc * xc, ONE)
    x3 = _fdiv(x2 * xc, ONE)
    u = xc + _fdiv(GELU_C_Q16 * x3, ONE)
    z = _fdiv(SQRT_2_OVER_PI_Q16 * u, ONE)
    g = _rshift_round(xc * (ONE + _tanh_q16(z)), F + 1)
    # outside [-6, 6] gelu(x) = x (right) or 0 (left) to ~1e-9
    y = _torch.where(x > 6 * ONE, x, g)
    y = _torch.where(x < -6 * ONE, _torch.zeros_like(x), y)
    return y

def _softmax_rows_q16(scores):
    """Integer softmax over the last dim. Returns Q16 probabilities."""
    m = scores.max(-1, keepdim=True).values
    e = _exp_q16(scores - m)                          # Q16, max elem -> ONE
    s = e.sum(-1, keepdim=True)
    return _fdiv(e * ONE, s)                          # Q16 probs

def _attention(x, layer, T):
    W = _W
    qkv = _matmul_exact(_rshift_round(x, F - WF), W[f'h.{layer}.attn.c_attn.w'],
                        2 * WF - F, W[f'h.{layer}.attn.c_attn.b'])
    qkv = _torch.clamp(qkv, -ACT_CLAMP, ACT_CLAMP)
    q, k, v = qkv.split(768, dim=-1)
    def heads(t):  # [T,768] -> [12,T,64]
        return t.reshape(T, 12, 64).permute(1, 0, 2)
    q = _torch.clamp(heads(q), -QK_CLAMP, QK_CLAMP)
    k = _torch.clamp(heads(k), -QK_CLAMP, QK_CLAMP)
    v = heads(v)
    q12 = _rshift_round(q, F - WF)
    k12 = _rshift_round(k, F - WF)
    # scores = q k^T / sqrt(64);  Q12*Q12 sum64 -> Q24, /8 -> >>3, to Q16 -> >>8
    s = (q12.double() @ k12.double().transpose(-1, -2)).long()
    s = _rshift_round(s, 2 * WF - F + 3)              # Q16 scores
    mask = _torch.triu(_torch.ones(T, T, dtype=_torch.bool, device=s.device), 1)
    s = _torch.where(mask, _torch.full_like(s, MASK_NEG), s)
    probs = _softmax_rows_q16(s)                      # [12,T,T] Q16
    v12 = _rshift_round(_torch.clamp(v, -ACT_CLAMP, ACT_CLAMP), F - WF)
    o = (probs.double() @ v12.double()).long()        # Q16*Q12 -> Q28
    o = _rshift_round(o, WF)                          # Q16
    o = o.permute(1, 0, 2).reshape(T, 768)
    o = _matmul_exact(_rshift_round(o, F - WF), W[f'h.{layer}.attn.c_proj.w'],
                      2 * WF - F, W[f'h.{layer}.attn.c_proj.b'])
    return _torch.clamp(o, -ACT_CLAMP, ACT_CLAMP), probs

def _mlp(x, layer):
    W = _W
    h = _matmul_exact(_rshift_round(x, F - WF), W[f'h.{layer}.mlp.c_fc.w'],
                      2 * WF - F, W[f'h.{layer}.mlp.c_fc.b'])
    h = _gelu(_torch.clamp(h, -ACT_CLAMP, ACT_CLAMP))
    h = _matmul_exact(_rshift_round(h, F - WF), W[f'h.{layer}.mlp.c_proj.w'],
                      2 * WF - F, W[f'h.{layer}.mlp.c_proj.b'])
    return _torch.clamp(h, -ACT_CLAMP, ACT_CLAMP)

# ------------------------------------------------------------ model --------

def load_model():
    """Convert HF gpt2 float weights to integers ONCE (on CPU, deterministic:
    identical safetensors bytes -> identical ints on every platform), then
    move to the compute device. Prints WEIGHTS_HASH for protocol pinning."""
    global _W, _LNF, _tok, _device, _torch, WEIGHTS_HASH
    if _W is not None:
        return
    import torch
    _torch = torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[int-infer] loading gpt2, integerizing on cpu, running on {_device} ...")
    fm = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = fm.state_dict()
    def q_w(name):   # weight -> Q12, clamp |w|<=16
        w = sd[name].double()
        return torch.clamp(torch.round(w * (1 << WF)), -(16 << WF), 16 << WF).long()
    def q_b(name):   # bias/embedding -> Q16
        b = sd[name].double()
        return torch.clamp(torch.round(b * ONE), -ACT_CLAMP, ACT_CLAMP).long()
    W = {'wte': q_b('transformer.wte.weight'), 'wpe': q_b('transformer.wpe.weight')}
    for l in range(12):
        p = f'transformer.h.{l}.'
        W[f'h.{l}.ln_1.g'] = q_w(p + 'ln_1.weight'); W[f'h.{l}.ln_1.b'] = q_b(p + 'ln_1.bias')
        W[f'h.{l}.ln_2.g'] = q_w(p + 'ln_2.weight'); W[f'h.{l}.ln_2.b'] = q_b(p + 'ln_2.bias')
        for m in ('attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj'):
            W[f'h.{l}.{m}.w'] = q_w(p + m + '.weight')
            W[f'h.{l}.{m}.b'] = q_b(p + m + '.bias')
    # ln_f is converted the same deterministic way but kept OUT of W so
    # WEIGHTS_HASH (the consensus pin over the fingerprint path) is unchanged.
    # It is only used by generate(), whose own determinism is by construction.
    lnf = {'g': q_w('transformer.ln_f.weight'), 'b': q_b('transformer.ln_f.bias')}
    hh = hashlib.sha256()
    for kname in sorted(W):
        hh.update(kname.encode())
        hh.update(W[kname].cpu().numpy().tobytes())
    WEIGHTS_HASH = hh.hexdigest()
    print(f"[int-infer] integer weights hash: {WEIGHTS_HASH}")
    _W = {kk: vv.to(_device) for kk, vv in W.items()}
    _LNF = {kk: vv.to(_device) for kk, vv in lnf.items()}
    del fm, sd
    _tok = GPT2Tokenizer.from_pretrained("gpt2")

def attention_rows(prompt, salt, heads):
    """Run the integer forward pass; return {(layer, head): [Q16 int, ...]}
    -- the last-token attention row of each requested head. Pure integers."""
    load_model()
    torch = _torch
    ids = _tok(f"{salt} {prompt}", return_tensors="pt")["input_ids"][0].to(_device)
    T = ids.shape[0]
    need = {}
    with torch.no_grad():
        x = _W['wte'].index_select(0, ids) + _W['wpe'][:T]
        x = torch.clamp(x, -ACT_CLAMP, ACT_CLAMP)
        want_by_layer = {}
        for (l, h) in heads:
            want_by_layer.setdefault(l, []).append(h)
        for l in range(12):
            a, probs = _attention(_layernorm(x, _W[f'h.{l}.ln_1.g'], _W[f'h.{l}.ln_1.b']), l, T)
            for h in want_by_layer.get(l, []):
                need[(l, h)] = [int(v) for v in probs[h, -1].cpu()]
            x = torch.clamp(x + a, -ACT_CLAMP, ACT_CLAMP)
            x = torch.clamp(x + _mlp(_layernorm(x, _W[f'h.{l}.ln_2.g'], _W[f'h.{l}.ln_2.b']), l),
                            -ACT_CLAMP, ACT_CLAMP)
    return need

# ------------------------------------------------- fingerprint (v4) --------

def quantize_grid_int(q16_row, grid=GRID):
    """Largest-remainder allocation of `grid` units -- pure integer version
    of the v3 quantizer. Input: Q16 integer probabilities."""
    S = sum(q16_row)
    raw = [p * grid for p in q16_row]
    base = [r // S for r in raw]
    rem = [raw[i] - base[i] * S for i in range(len(raw))]
    need = grid - sum(base)
    order = sorted(range(len(raw)), key=lambda i: (rem[i], -i), reverse=True)
    for k in range(need):
        base[order[k % len(base)]] += 1
    return base

def compress_glyphs_int(int_scores):
    """Identical to v3 glyph compression (pure python ints, already exact)."""
    vals = list(int_scores)
    med = sorted(vals)[len(vals) // 2]
    typs = ['R' if v > med else 'G' for v in vals]
    glyphs = []; level = 0
    while len(vals) > 1:
        level += 1; n = len(vals)
        order = sorted(range(n), key=lambda i: (vals[i], -i), reverse=True)
        sv = [vals[i] for i in order]; st = [typs[i] for i in order]
        if n % 2 == 0:
            seq_v, seq_t = sv, st
            pairs = [(i, i + 1) for i in range(0, n, 2)]
        else:
            lv, lt, rv, rt = [], [], [], []
            for i in range(n):
                (lv if i % 2 == 0 else rv).append(sv[i])
                (lt if i % 2 == 0 else rt).append(st[i])
            seq_v = lv + rv[::-1]; seq_t = lt + rt[::-1]
            pairs = [(i, i + 1) for i in range(n - 1)]
        nv, nt = [], []
        for i, j in pairs:
            a, b, ta, tb = seq_v[i], seq_v[j], seq_t[i], seq_t[j]
            if ta == tb and ta != 'X': t = ta
            elif 'X' in (ta, tb):     t = 'R'
            else:                     t = 'X'
            nv.append(a + b); nt.append(t)
        reg = [(v, t) for v, t in zip(nv, nt) if t != 'X']
        gly = [v for v, t in zip(nv, nt) if t == 'X']
        for gv in gly: glyphs.append((level, gv))
        if reg:
            vals = [v for v, _ in reg]; typs = [t for _, t in reg]
        else:
            vals, typs = nv, nt
    return (vals[0] if vals else 0), tuple(glyphs)

def heads_for_salt(salt, layers=12, n_heads=12, k=6):
    import random
    h = hashlib.sha256(salt.encode()).digest()
    rng = random.Random(int.from_bytes(h, 'big'))
    pairs = [(l, hd) for l in range(layers) for hd in range(n_heads)]
    return tuple(sorted(rng.sample(pairs, k)))

def inference_hash(prompt, salt):
    """v4 proof: bit-identical on every device by construction."""
    heads = heads_for_salt(salt)
    rows = attention_rows(prompt, salt, heads)
    parts = []
    for (layer, head) in heads:
        B, gl = compress_glyphs_int(quantize_grid_int(rows[(layer, head)]))
        parts.append((layer, head, B, gl))
    fp = json.dumps(parts, separators=(',', ':'))
    return hashlib.sha256((salt + '|' + fp).encode()).hexdigest()

# ------------------------------------------------- generation (market) -----
# Deterministic text generation for the inference market (docs/USEFUL_WORK.md).
# Same exactness argument as the fingerprint path: every op is integer or an
# exact float64 integer matmul, so generate(prompt, n) is bit-identical on
# every device. That exactness is what makes the market's fraud proofs
# objective: one differing token is a provable lie, no tolerance needed.
#
# Logits bound: hidden Q9 (|q| <= 4096*2^9 = 2^21) x wte Q9 (same bound),
# contraction K = 768: |sum| <= 768 * 2^42 < 2^52 < 2^53 -- every float64
# partial sum exact regardless of accumulation order.

GEN_MAX_CONTEXT = 512      # positions cap (wpe has 1024; stay well inside)

def _forward(ids):
    """Full integer transformer pass; returns final hidden states [T,768] Q16."""
    torch = _torch
    T = ids.shape[0]
    x = _W['wte'].index_select(0, ids) + _W['wpe'][:T]
    x = torch.clamp(x, -ACT_CLAMP, ACT_CLAMP)
    for l in range(12):
        a, _ = _attention(_layernorm(x, _W[f'h.{l}.ln_1.g'], _W[f'h.{l}.ln_1.b']), l, T)
        x = torch.clamp(x + a, -ACT_CLAMP, ACT_CLAMP)
        x = torch.clamp(x + _mlp(_layernorm(x, _W[f'h.{l}.ln_2.g'], _W[f'h.{l}.ln_2.b']), l),
                        -ACT_CLAMP, ACT_CLAMP)
    return x

def generate(prompt, max_new_tokens=64):
    """Greedy (argmax) decode, integer-exact and therefore bit-identical
    everywhere. No KV cache: T <= GEN_MAX_CONTEXT keeps the recompute cheap
    and the code inside the audited exact-arithmetic paths above."""
    load_model()
    torch = _torch
    ids = _tok(prompt, return_tensors="pt")["input_ids"][0].to(_device)
    ids = ids[-(GEN_MAX_CONTEXT - max_new_tokens):]
    out_ids = []
    w9 = _rshift_round(_W['wte'], F - 9)               # Q9 embeddings
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if ids.shape[0] >= GEN_MAX_CONTEXT:
                break
            x = _forward(ids)
            h = _layernorm(x[-1:], _LNF['g'], _LNF['b'])
            h9 = _rshift_round(h, F - 9)               # Q9 hidden
            logits = (h9.double() @ w9.double().t()).long()[0].cpu()
            nxt = int(torch.argmax(logits))            # CPU argmax: first-max
                                                       # tie-break, deterministic
            if nxt == 50256:                           # <|endoftext|>
                break
            ids = torch.cat([ids, torch.tensor([nxt], device=_device)])
            out_ids.append(nxt)
    return _tok.decode(out_ids)
