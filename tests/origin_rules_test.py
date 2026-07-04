"""
ORIGIN RULES TEST: resurrect two rules from the original notebook conversation
and duel them against the current champion.

  CHAMP    : current pipeline (disjoint pairs on even, add everywhere)
  SUBTRACT : glyph-glyph collisions SUBTRACT (|a-b|) instead of add
             (the notebook's "unorthodox but it worked" rule)
  BORDER   : even levels use OVERLAPPING pairs and DROP the middle sum
             (the notebook's "find and ignore the border" rule)

Arena: noise survival (1e-5, 1e-4) and fingerprint-space uniqueness (300),
6 salt-selected heads, GPT-2, integer grid 100. No changes to the live system.
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

def compress(int_scores, mode="champ"):
    """mode: champ | subtract | border"""
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
            if mode == "border" and n >= 4:
                # notebook rule: overlapping pairs, drop the middle sum
                seq_v, seq_t = sv, st
                pairs = [(i, i + 1) for i in range(n - 1)]
                drop_border = True
            else:
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
            if ta == tb and ta != 'X':
                t = ta; val = a + b
            elif ta == 'X' and tb == 'X':
                t = 'R'
                val = abs(a - b) if mode == "subtract" else a + b
            elif 'X' in (ta, tb):
                t = 'R'; val = a + b
            else:
                t = 'X'; val = a + b
            nv.append(val); nt.append(t)
        if drop_border:
            mid = len(nv) // 2          # the border = middle overlapping sum
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

def fp_from_rows(rows, heads, mode):
    parts = []
    for (l, h), row in zip(heads, rows):
        B, gl = compress(quantize_int(row), mode)
        parts.append((l, h, B, gl))
    return json.dumps(parts, separators=(',', ':'))

VOCAB = """the a is was in on to and of it that for with as at by from or an
be this not but had has they we you all can will one my out if up so big old
new good long great small right came made after back only over take year some
could time very when what how said dog cat sun moon tree bird fish door red
blue dark light cold warm fast slow silver golden copper iron ember frost""".split()
rng = random.Random(11)
def rand_prompt(): return " ".join(rng.choices(VOCAB, k=rng.randint(6, 14)))

salt = "origin_salt"; heads = heads_for_salt(salt)
MODES = ["champ", "subtract", "border"]
print(f"Device: {DEVICE}   heads: {heads}\n")

print("=" * 70)
print("1) NOISE SURVIVAL (120 trials each)")
print("=" * 70)
noise_rng = random.Random(5)
TRIALS = 120
for exp in [-5, -4]:
    eps = 10 ** exp
    ok = {m: 0 for m in MODES}
    for _ in range(TRIALS):
        p = rand_prompt()
        rows = get_rows(p, salt, heads)
        base = {m: fp_from_rows(rows, heads, m) for m in MODES}
        noisy_rows = []
        for row in rows:
            noisy = [max(1e-12, v + noise_rng.uniform(-eps, eps)) for v in row]
            s = sum(noisy)
            noisy_rows.append([v / s for v in noisy])
        for m in MODES:
            if fp_from_rows(noisy_rows, heads, m) == base[m]:
                ok[m] += 1
    print(f"  noise 1e{exp}:  " + "   ".join(f"{m} {ok[m]:3d}/{TRIALS}" for m in MODES))

print("\n" + "=" * 70)
print("2) FINGERPRINT SPACE (300 prompts)")
print("=" * 70)
fps = {m: set() for m in MODES}
for _ in range(300):
    p = rand_prompt()
    rows = get_rows(p, salt, heads)
    for m in MODES:
        fps[m].add(fp_from_rows(rows, heads, m))
for m in MODES:
    print(f"  {m:9s}: {len(fps[m])}/300 unique ({100*len(fps[m])/300:.1f}%)")

print("""
READING:
  challenger > champ on survival AND ~100% unique  -> notebook rule was better,
                                                      consider protocol v3
  challenger <= champ                              -> current design defended
                                                      (3rd straight win)
""")