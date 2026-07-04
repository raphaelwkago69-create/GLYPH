"""
FULL SYSTEM DUEL: ORIGIN vs CURRENT

ORIGIN  = the complete pipeline as designed in the first notebook session:
          even levels -> overlapping pairs, DROP the middle sum (the border)
          odd levels  -> water-ripple palindrome, sliding pairs
          merged(glyph) collisions -> SUBTRACT
CURRENT = the shipped pipeline:
          even levels -> disjoint pairs, nothing dropped
          odd levels  -> palindrome sliding pairs (survived from origin)
          all merges  -> ADD

Both run on the same integer grid (the determinism layer is orthogonal
to the cascade rules and belongs to both).

Arena:
  1) Noise survival        (higher = better consensus robustness)
  2) Fingerprint space     (uniqueness; low = lookup-table vulnerable)
  3) Irreversibility       (preimage ambiguity: how many distinct inputs
                            share a fingerprint; targeted-search hit rate)
"""
import hashlib, json, math, random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GRID = 100; N_FP_HEADS = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.eval(); model = model.to(DEVICE)
tok = GPT2Tokenizer.from_pretrained("gpt2")

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

def compress(int_scores, system):
    """system: 'current' or 'origin'"""
    vals = list(int_scores)
    med = sorted(vals)[len(vals) // 2]
    typs = ['R' if v > med else 'G' for v in vals]
    glyphs = []; level = 0
    while len(vals) > 1:
        level += 1; n = len(vals)
        order = sorted(range(n), key=lambda i: (vals[i], -i), reverse=True)
        sv = [vals[i] for i in order]; st = [typs[i] for i in order]
        drop_border = False
        if n % 2 == 0:
            if system == "origin" and n >= 4:
                seq_v, seq_t = sv, st
                pairs = [(i, i + 1) for i in range(n - 1)]   # overlapping
                drop_border = True                            # ignore border
            else:
                seq_v, seq_t = sv, st
                pairs = [(i, i + 1) for i in range(0, n, 2)] # disjoint
        else:
            lv, lt, rv, rt = [], [], [], []
            for i in range(n):
                (lv if i % 2 == 0 else rv).append(sv[i])
                (lt if i % 2 == 0 else rt).append(st[i])
            seq_v = lv + rv[::-1]; seq_t = lt + rt[::-1]      # water ripple
            pairs = [(i, i + 1) for i in range(n - 1)]
        nv, nt = [], []
        for i, j in pairs:
            a, b, ta, tb = seq_v[i], seq_v[j], seq_t[i], seq_t[j]
            if ta == tb and ta != 'X':
                t = ta; val = a + b
            elif ta == 'X' and tb == 'X':
                t = 'R'
                val = abs(a - b) if system == "origin" else a + b  # subtract rule
            elif 'X' in (ta, tb):
                t = 'R'; val = a + b
            else:
                t = 'X'; val = a + b
            nv.append(val); nt.append(t)
        if drop_border:
            mid = len(nv) // 2
            del nv[mid]; del nt[mid]
        reg = [(v, t) for v, t in zip(nv, nt) if t != 'X']
        gly = [v for v, t in zip(nv, nt) if t == 'X']
        for gv in gly: glyphs.append((level, gv))
        if reg:
            vals = [v for v, _ in reg]; typs = [t for _, t in reg]
        else:
            vals, typs = nv, nt
    return (vals[0] if vals else 0), tuple(glyphs)

def heads_for_salt(salt, k=N_FP_HEADS):
    h = hashlib.sha256(salt.encode()).digest()
    rng = random.Random(int.from_bytes(h, 'big'))
    return tuple(sorted(rng.sample([(l, hd) for l in range(12) for hd in range(12)], k)))

def get_rows(prompt, salt, heads):
    inputs = tok(f"{salt} {prompt}", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    return [out.attentions[l][0, h, -1].cpu().tolist() for (l, h) in heads]

def fp_from_rows(rows, heads, system):
    parts = []
    for (l, h), row in zip(heads, rows):
        B, gl = compress(quantize_int(row), system)
        parts.append((l, h, B, gl))
    return json.dumps(parts, separators=(',', ':'))

VOCAB = """the a is was in on to and of it that for with as at by from or an
be this not but had has they we you all can will one my out if up so big old
new good long great small right came made after back only over take year some
could time very when what how said dog cat sun moon tree bird fish door red
blue dark light cold warm fast slow silver golden copper iron ember frost""".split()
rng = random.Random(23)
def rand_prompt(): return " ".join(rng.choices(VOCAB, k=rng.randint(6, 14)))

SYSTEMS = ["current", "origin"]
salt = "duel_salt"; heads = heads_for_salt(salt)
print(f"Device: {DEVICE}\n")

# ---- 1) noise survival ------------------------------------------------------
print("=" * 70)
print("ROUND 1: NOISE SURVIVAL (120 trials)")
print("=" * 70)
noise_rng = random.Random(3)
for exp in [-5, -4]:
    eps = 10 ** exp
    ok = {s: 0 for s in SYSTEMS}
    for _ in range(120):
        p = rand_prompt()
        rows = get_rows(p, salt, heads)
        base = {s: fp_from_rows(rows, heads, s) for s in SYSTEMS}
        noisy_rows = []
        for row in rows:
            noisy = [max(1e-12, v + noise_rng.uniform(-eps, eps)) for v in row]
            t = sum(noisy); noisy_rows.append([v / t for v in noisy])
        for s in SYSTEMS:
            if fp_from_rows(noisy_rows, heads, s) == base[s]: ok[s] += 1
    print(f"  noise 1e{exp}:  current {ok['current']:3d}/120    origin {ok['origin']:3d}/120")

# ---- 2) fingerprint space ---------------------------------------------------
print("\n" + "=" * 70)
print("ROUND 2: FINGERPRINT SPACE (300 prompts)")
print("=" * 70)
fps = {s: set() for s in SYSTEMS}
for _ in range(300):
    p = rand_prompt()
    rows = get_rows(p, salt, heads)
    for s in SYSTEMS:
        fps[s].add(fp_from_rows(rows, heads, s))
for s in SYSTEMS:
    print(f"  {s:8s}: {len(fps[s])}/300 unique ({100*len(fps[s])/300:.1f}%)")

# ---- 3) irreversibility -----------------------------------------------------
print("\n" + "=" * 70)
print("ROUND 3: IRREVERSIBILITY (no model needed: pure math on the cascade)")
print("=" * 70)
# 3a) preimage ambiguity: N random integer compositions of GRID into 12 parts;
#     how many distinct inputs collapse onto each fingerprint?
comp_rng = random.Random(77)
def random_composition(total=GRID, parts=12):
    cuts = sorted(comp_rng.sample(range(1, total), parts - 1))
    prev = 0; out = []
    for c in cuts:
        out.append(c - prev); prev = c
    out.append(total - prev)
    comp_rng.shuffle(out)
    return out

N_PRE = 30000
for s in SYSTEMS:
    seen = {}
    comp_rng.seed(77)                       # identical inputs for both systems
    for _ in range(N_PRE):
        row = random_composition()
        key = json.dumps(compress(row, s))
        seen[key] = seen.get(key, 0) + 1
    n_fp = len(seen)
    mx = max(seen.values())
    avg = N_PRE / n_fp
    print(f"  {s:8s}: {N_PRE} inputs -> {n_fp} fingerprints | "
          f"avg preimages/fp {avg:.2f} | largest cluster {mx}")

# 3b) targeted search: given ONE real fingerprint, how often does a random
#     input hit it? (forward-search hardness proxy)
print()
real_rows = get_rows(rand_prompt(), salt, heads)
for s in SYSTEMS:
    target = json.dumps(compress(quantize_int(real_rows[0]), s))
    comp_rng.seed(99)
    hits = sum(1 for _ in range(50000)
               if json.dumps(compress(random_composition(parts=len(real_rows[0])), s)) == target)
    print(f"  {s:8s}: targeted preimage search: {hits}/50000 random inputs "
          f"hit a real fingerprint")

print("""
READING (what to preserve):
  R1 higher  = better consensus safety      (weight: high)
  R2 ~100%   = required, disqualifying if low
  R3 avg preimages HIGHER = harder to invert (privacy/one-wayness better)
     but targeted hits should be ~0 for both (else forgeable)
Pick the system that wins the weighted board; mixed result -> preserve the
winning rule from each round and test the hybrid before adopting anything.
""")