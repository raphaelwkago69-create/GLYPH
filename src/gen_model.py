"""
==============================================================================
GEN-MODEL -- configurable exact-integer GPT-2 for the verification market
==============================================================================
int_infer.py hard-codes GPT-2 *small* for the consensus lottery. This module
runs ANY GPT-2 size (small/medium/large/xl, 124M -> 1.5B) through the SAME
bit-exact integer arithmetic, so bigger, more coherent models can be served and
cheaply verified by the bisection game (src/verify_game.py).

Why the exact-integer bounds still hold at 1.5B: every GPT-2 variant keeps the
attention head dimension at 64 and shares the LayerNorm/GELU/learned-position
architecture. The only things that change are depth (n_layer) and width
(n_embd) -- both already handled by the chunked (K<=1024) exact matmul in
int_infer. So a larger model is a *parameterization*, not new math, and the
referee's one-step re-execution stays undeniable at any size.

The consensus lottery path in int_infer is left untouched (its WEIGHTS_HASH pin
and the 25-test suite are unaffected); this is a separate weight store for the
generation/answer path only. Each model id gets its own pinned weights hash.
==============================================================================
"""
import hashlib
import int_infer as ii

F, WF, ONE = ii.F, ii.WF, ii.ONE
ACT_CLAMP, QK_CLAMP, MASK_NEG = ii.ACT_CLAMP, ii.QK_CLAMP, ii.MASK_NEG
GEN_MAX_CONTEXT = ii.GEN_MAX_CONTEXT

_CACHE = {}          # name -> loaded Model (converted once per process)


def load(name="gpt2"):
    """Convert an HF GPT-2 variant to integers ONCE (deterministic on CPU) and
    cache it. Returns a Model dict. Valid names: gpt2, gpt2-medium, gpt2-large,
    gpt2-xl."""
    if name in _CACHE:
        return _CACHE[name]
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    ii._torch = torch                      # int_infer's exact ops use this handle
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[gen-model] loading {name}, integerizing on cpu ...")
    fm = GPT2LMHeadModel.from_pretrained(name)
    cfg = {"n_layer": fm.config.n_layer, "n_embd": fm.config.n_embd,
           "n_head": fm.config.n_head}
    hd = cfg["n_embd"] // cfg["n_head"]
    assert hd == 64, f"exact score-scaling assumes head_dim 64, got {hd}"
    sd = fm.state_dict()

    def q_w(nm):
        w = sd[nm].double()
        return torch.clamp(torch.round(w * (1 << WF)), -(16 << WF), 16 << WF).long()

    def q_b(nm):
        b = sd[nm].double()
        return torch.clamp(torch.round(b * ONE), -ACT_CLAMP, ACT_CLAMP).long()

    W = {'wte': q_b('transformer.wte.weight'), 'wpe': q_b('transformer.wpe.weight')}
    for l in range(cfg["n_layer"]):
        p = f'transformer.h.{l}.'
        W[f'h.{l}.ln_1.g'] = q_w(p + 'ln_1.weight'); W[f'h.{l}.ln_1.b'] = q_b(p + 'ln_1.bias')
        W[f'h.{l}.ln_2.g'] = q_w(p + 'ln_2.weight'); W[f'h.{l}.ln_2.b'] = q_b(p + 'ln_2.bias')
        for m in ('attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj'):
            W[f'h.{l}.{m}.w'] = q_w(p + m + '.weight')
            W[f'h.{l}.{m}.b'] = q_b(p + m + '.bias')
    lnf = {'g': q_w('transformer.ln_f.weight'), 'b': q_b('transformer.ln_f.bias')}

    hh = hashlib.sha256()
    for kname in sorted(W):
        hh.update(kname.encode()); hh.update(W[kname].cpu().numpy().tobytes())
    whash = hh.hexdigest()

    tok = GPT2Tokenizer.from_pretrained(name)
    M = {"name": name, "cfg": cfg, "device": dev, "torch": torch, "tok": tok,
         "W": {k: v.to(dev) for k, v in W.items()},
         "LNF": {k: v.to(dev) for k, v in lnf.items()},
         "w9": ii._rshift_round(W['wte'].to(dev), F - 9), "hash": whash}
    del fm, sd
    print(f"[gen-model] {name}: {cfg['n_layer']}L x {cfg['n_embd']}d  "
          f"weights hash {whash[:16]}...")
    _CACHE[name] = M
    return M


# ---- exact layer primitives (reuse int_infer's dimension-agnostic ops) -----

def _attn_block(x, M, l, T):
    t = M["torch"]; W = M["W"]; ne = M["cfg"]["n_embd"]; nh = M["cfg"]["n_head"]
    hd = ne // nh
    qkv = ii._matmul_exact(ii._rshift_round(x, F - WF), W[f'h.{l}.attn.c_attn.w'],
                           2 * WF - F, W[f'h.{l}.attn.c_attn.b'])
    qkv = t.clamp(qkv, -ACT_CLAMP, ACT_CLAMP)
    q, k, v = qkv.split(ne, dim=-1)
    heads = lambda u: u.reshape(T, nh, hd).permute(1, 0, 2)
    q = t.clamp(heads(q), -QK_CLAMP, QK_CLAMP)
    k = t.clamp(heads(k), -QK_CLAMP, QK_CLAMP)
    v = heads(v)
    q12 = ii._rshift_round(q, F - WF); k12 = ii._rshift_round(k, F - WF)
    s = (q12.double() @ k12.double().transpose(-1, -2)).long()
    s = ii._rshift_round(s, 2 * WF - F + 3)            # /sqrt(64) baked in (>>3)
    mask = t.triu(t.ones(T, T, dtype=t.bool, device=s.device), 1)
    s = t.where(mask, t.full_like(s, MASK_NEG), s)
    probs = ii._softmax_rows_q16(s)
    v12 = ii._rshift_round(t.clamp(v, -ACT_CLAMP, ACT_CLAMP), F - WF)
    o = (probs.double() @ v12.double()).long()
    o = ii._rshift_round(o, WF).permute(1, 0, 2).reshape(T, ne)
    o = ii._matmul_exact(ii._rshift_round(o, F - WF), W[f'h.{l}.attn.c_proj.w'],
                         2 * WF - F, W[f'h.{l}.attn.c_proj.b'])
    return t.clamp(o, -ACT_CLAMP, ACT_CLAMP)


def _mlp_block(x, M, l):
    t = M["torch"]; W = M["W"]
    h = ii._matmul_exact(ii._rshift_round(x, F - WF), W[f'h.{l}.mlp.c_fc.w'],
                         2 * WF - F, W[f'h.{l}.mlp.c_fc.b'])
    h = ii._gelu(t.clamp(h, -ACT_CLAMP, ACT_CLAMP))
    h = ii._matmul_exact(ii._rshift_round(h, F - WF), W[f'h.{l}.mlp.c_proj.w'],
                         2 * WF - F, W[f'h.{l}.mlp.c_proj.b'])
    return t.clamp(h, -ACT_CLAMP, ACT_CLAMP)


def embed(ids, M):
    t = M["torch"]; W = M["W"]; T = ids.shape[0]
    x = W['wte'].index_select(0, ids) + W['wpe'][:T]
    return t.clamp(x, -ACT_CLAMP, ACT_CLAMP)


def layer_attn(x, M, l):
    W = M["W"]
    return ii._layernorm(x, W[f'h.{l}.ln_1.g'], W[f'h.{l}.ln_1.b']), x


def emit_logits(x, M):
    """Final layernorm + logits (Q9 hidden @ Q9 embeddings); returns argmax."""
    t = M["torch"]
    h = ii._layernorm(x[-1:], M["LNF"]['g'], M["LNF"]['b'])
    h9 = ii._rshift_round(h, F - 9)
    logits = (h9.double() @ M["w9"].double().t()).long()[0].cpu()
    return int(t.argmax(logits))


def generate_ref(prompt, max_new_tokens, M):
    """Monolithic greedy decode (a full forward per token, NOT the step
    machine) -- an independent reference the committed trace must reproduce."""
    t = M["torch"]; W = M["W"]; nL = M["cfg"]["n_layer"]
    ids = M["tok"](prompt, return_tensors="pt")["input_ids"][0].to(M["device"])
    ids = ids[-(GEN_MAX_CONTEXT - max_new_tokens):]
    out = []
    with t.no_grad():
        for _ in range(max_new_tokens):
            if ids.shape[0] >= GEN_MAX_CONTEXT:
                break
            T = ids.shape[0]
            x = embed(ids, M)
            for l in range(nL):
                x = t.clamp(x + _attn_block(
                    ii._layernorm(x, W[f'h.{l}.ln_1.g'], W[f'h.{l}.ln_1.b']), M, l, T),
                    -ACT_CLAMP, ACT_CLAMP)
                x = t.clamp(x + _mlp_block(
                    ii._layernorm(x, W[f'h.{l}.ln_2.g'], W[f'h.{l}.ln_2.b']), M, l),
                    -ACT_CLAMP, ACT_CLAMP)
            nxt = emit_logits(x, M)
            if nxt == 50256:
                break
            ids = t.cat([ids, t.tensor([nxt], device=M["device"])])
            out.append(nxt)
    return M["tok"].decode(out)
