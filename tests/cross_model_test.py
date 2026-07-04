"""
CROSS-MODEL TEST: prove the fingerprint is model-specific (it MUST be).
Same salt + same prompt through GPT-2 vs DistilGPT2:
  - hashes must be completely unrelated (verification would fail)
If a cheaper model could reproduce GPT-2's hashes, mining could be faked.
"""
import hashlib, json, math, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GRID = 100
N_FP_HEADS = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def quantize_int(scores, grid=GRID):
    raw = [s * grid for s in scores]
    floor = [math.floor(r) for r in raw]
    remainder = grid - sum(floor)
    if remainder > 0:
        fracs = sorted(range(len(raw)), key=lambda i: (raw[i] - floor[i], -i), reverse=True)
        for k in range(remainder):
            floor[fracs[k % len(fracs)]] += 1
    elif remainder < 0:
        fracs = sorted(range(len(raw)), key=lambda i: (raw[i] - floor[i], -i))
        for k in range(-remainder):
            floor[fracs[k % len(fracs)]] -= 1
    return floor

def compress_glyphs_int(int_scores):
    vals = list(int_scores)
    med = sorted(vals)[len(vals) // 2]
    typs = ['R' if v > med else 'G' for v in vals]
    glyphs = []
    level = 0
    while len(vals) > 1:
        level += 1
        n = len(vals)
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
        for gv in gly:
            glyphs.append((level, gv))
        if reg:
            vals = [v for v, _ in reg]; typs = [t for _, t in reg]
        else:
            vals, typs = nv, nt
    B = vals[0] if vals else 0
    return B, tuple(glyphs)

def heads_for_salt(salt, n_layers, n_heads, k=N_FP_HEADS):
    h = hashlib.sha256(salt.encode()).digest()
    rng = random.Random(int.from_bytes(h, 'big'))
    pairs = [(l, hd) for l in range(n_layers) for hd in range(n_heads)]
    return tuple(sorted(rng.sample(pairs, k)))

def fingerprint(model, tok, prompt, salt, n_layers, n_heads):
    heads = heads_for_salt(salt, n_layers, n_heads)
    inputs = tok(f"{salt} {prompt}", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    parts = []
    for (layer, head) in heads:
        row = out.attentions[layer][0, head, -1].cpu().tolist()
        B, gl = compress_glyphs_int(quantize_int(row))
        parts.append((layer, head, B, gl))
    fp = json.dumps(parts, separators=(',', ':'))
    return hashlib.sha256((salt + '|' + fp).encode()).hexdigest(), heads

MODELS = ["gpt2", "distilgpt2"]
CASES = [
    ("salt_alpha_001", "tower for one signal time right window cat memory letter great some"),
    ("salt_beta_002",  "the quick brown dog jumped over the lazy sleeping cat"),
    ("salt_gamma_003", "silver golden copper iron ember frost field ocean forest"),
    ("salt_delta_004", "mathematics is the language of the universe"),
    ("salt_eps_005",   "short sentence"),
]

print(f"Device: {DEVICE}\n")
results = {}
for name in MODELS:
    print(f"Loading {name} ...")
    model = AutoModelForCausalLM.from_pretrained(name, output_attentions=True)
    model.eval(); model = model.to(DEVICE)
    tok = AutoTokenizer.from_pretrained(name)
    cfg = model.config
    n_layers = cfg.n_layer; n_heads = cfg.n_head
    print(f"  {name}: {n_layers} layers x {n_heads} heads, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    hashes = []
    for salt, prompt in CASES:
        hx, heads = fingerprint(model, tok, prompt, salt, n_layers, n_heads)
        hashes.append(hx)
    results[name] = hashes
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

print("\n" + "=" * 74)
print("SAME SALT + SAME PROMPT, TWO MODELS")
print("=" * 74)
matches = 0
for i, (salt, prompt) in enumerate(CASES):
    a, b = results["gpt2"][i], results["distilgpt2"][i]
    same = a == b
    matches += same
    print(f"\n  '{prompt[:45]}'")
    print(f"    gpt2       : {a}")
    print(f"    distilgpt2 : {b}")
    print(f"    match      : {same}")

print("\n" + "=" * 74)
print(f"VERDICT: {matches}/{len(CASES)} cross-model hash matches")
print("  0 matches = CORRECT. A different model cannot forge this model's")
print("  proofs, so the protocol's pinned-model rule is enforceable.")
print("  Any match would mean a cheap model could fake expensive inference.")
print("=" * 74)
