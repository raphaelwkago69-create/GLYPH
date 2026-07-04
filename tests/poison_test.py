"""
POISON TEST: attack the "strangers cross-confirm new models" admission rule.

Adversary submits a RANDOM-WEIGHT 1-layer pygmy transformer as a "new model".
We measure:
  A) Self-agreement  -- does it hash deterministically? (passes admission rule)
  B) Mining speed    -- inferences/sec vs real GPT-2 (cost fraud)
  C) Fingerprint space -- unique fingerprints over 300 prompts (lookup-table
     resurrection if degenerate)
Then the same three measurements for a LEGITIMATE new model (Qwen2.5-0.5B)
to show what an honest admission audit would look for.
"""
import hashlib, json, math, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

GRID = 100
N_FP_HEADS = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}\n")

def quantize_int(scores, grid=GRID):
    raw = [s * grid for s in scores]
    floor = [math.floor(r) for r in raw]
    remainder = grid - sum(floor)
    if remainder > 0:
        fr = sorted(range(len(raw)), key=lambda i: (raw[i] - floor[i], -i), reverse=True)
        for k in range(remainder): floor[fr[k % len(fr)]] += 1
    elif remainder < 0:
        fr = sorted(range(len(raw)), key=lambda i: (raw[i] - floor[i], -i))
        for k in range(-remainder): floor[fr[k % len(fr)]] -= 1
    return floor

def compress_glyphs_int(int_scores):
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

def heads_for_salt(salt, n_layers, n_heads, k=N_FP_HEADS):
    h = hashlib.sha256(salt.encode()).digest()
    rng = random.Random(int.from_bytes(h, 'big'))
    pairs = [(l, hd) for l in range(n_layers) for hd in range(n_heads)]
    k = min(k, len(pairs))
    return tuple(sorted(rng.sample(pairs, k)))

def fingerprint(model, tok, prompt, salt, n_layers, n_heads):
    heads = heads_for_salt(salt, n_layers, n_heads)
    inputs = tok(f"{salt} {prompt}", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    parts = []
    for (layer, head) in heads:
        row = out.attentions[layer][0, head, -1].float().cpu().tolist()
        B, gl = compress_glyphs_int(quantize_int(row))
        parts.append((layer, head, B, gl))
    fp = json.dumps(parts, separators=(',', ':'))
    return hashlib.sha256((salt + '|' + fp).encode()).hexdigest(), fp

VOCAB = """the a is was in on to and of it that for with as at by from or an
be this not but had has they we you all can will one my out if up so big old
new good long great small right came made after back only over take year some
could time very when what how said dog cat sun moon tree bird fish door red
blue dark light cold warm fast slow silver golden copper iron ember frost""".split()

def audit(name, model, tok, n_layers, n_heads, n_space=300):
    print(f"--- AUDIT: {name} ({n_layers}L x {n_heads}H, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params) ---")
    rng = random.Random(42)
    prompts = [" ".join(rng.choices(VOCAB, k=rng.randint(6, 14))) for _ in range(n_space)]

    # A) self-agreement: run 20 prompts twice, do hashes repeat exactly?
    agree = 0
    for p in prompts[:20]:
        h1, _ = fingerprint(model, tok, p, "admission_salt", n_layers, n_heads)
        h2, _ = fingerprint(model, tok, p, "admission_salt", n_layers, n_heads)
        agree += (h1 == h2)
    print(f"  A) self-agreement (Sybil cross-confirm passes?): {agree}/20")

    # B) mining speed
    t0 = time.time(); n_timed = 60
    for p in prompts[:n_timed]:
        fingerprint(model, tok, p, "speed_salt", n_layers, n_heads)
    speed = n_timed / (time.time() - t0)
    print(f"  B) inference speed: {speed:.1f} fingerprints/sec")

    # C) fingerprint space
    fps = set()
    for p in prompts:
        _, fp = fingerprint(model, tok, p, "space_salt", n_layers, n_heads)
        fps.add(fp)
    print(f"  C) fingerprint space: {len(fps)}/{n_space} unique "
          f"({100*len(fps)/n_space:.1f}%)")
    print()
    return agree, speed, len(fps) / n_space

results = {}

# ---- The REAL network model -------------------------------------------------
print("Loading gpt2 (the pinned network model) ...")
m = AutoModelForCausalLM.from_pretrained("gpt2", output_attentions=True)
m.eval(); m = m.to(DEVICE)
t = AutoTokenizer.from_pretrained("gpt2")
results["gpt2"] = audit("gpt2 [REAL NETWORK MODEL]", m, t, m.config.n_layer, m.config.n_head)
del m
if DEVICE.type == 'cuda': torch.cuda.empty_cache()

# ---- The ATTACK: random-weight pygmy model ----------------------------------
print("Constructing attacker's pygmy model (random weights, 1 layer) ...")
torch.manual_seed(666)  # attacker publishes this seed/weights; it's still deterministic
cfg = GPT2Config(n_layer=1, n_head=4, n_embd=64, vocab_size=50257)
cfg._attn_implementation = "eager"  # sdpa kernels can't return attention weights
pyg = GPT2LMHeadModel(cfg)  # RANDOM weights -- never trained, costs nothing
pyg.eval(); pyg = pyg.to(DEVICE)
results["pygmy"] = audit("pygmy [ATTACKER'S FAKE MODEL]", pyg, t, 1, 4)
del pyg
if DEVICE.type == 'cuda': torch.cuda.empty_cache()

# ---- A LEGITIMATE new model: Qwen2.5-0.5B ------------------------------------
print("Loading Qwen/Qwen2.5-0.5B (legitimate new model, ~1GB download) ...")
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", output_attentions=True,
                                          attn_implementation="eager")
qm.eval(); qm = qm.to(DEVICE)
qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
results["qwen"] = audit("Qwen2.5-0.5B [LEGIT NEW MODEL]",
                        qm, qt, qm.config.num_hidden_layers,
                        qm.config.num_attention_heads)

print("=" * 74)
print("VERDICT")
print("=" * 74)
ga, gs, gu = results["gpt2"]; pa, ps, pu = results["pygmy"]; qa, qs, qu = results["qwen"]
print(f"""
  Admission rule 'strangers agree' passed by:  gpt2 {ga}/20, pygmy {pa}/20, qwen {qa}/20
    -> the FAKE model passes admission exactly like real ones.

  Mining speed:  pygmy is {ps/gs:.1f}x faster than gpt2
    -> if admitted, the attacker out-mines the whole gpt2 tier for ~free.

  Fingerprint space:  gpt2 {100*gu:.1f}%  |  pygmy {100*pu:.1f}%  |  qwen {100*qu:.1f}%
    -> if pygmy's space is collapsed, lookup-table attacks return.

  CONCLUSION: determinism-based admission is Sybil-poisonable.
  Model admission must be a governance/whitelist decision (audited,
  versioned protocol upgrade), not a runtime network observation.
""")