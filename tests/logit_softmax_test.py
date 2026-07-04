"""
FOUNDER-SOFTMAX TEST: quantize in the LOGIT domain instead of the
probability domain. Softmax is invertible (log(p) = logit + const), so we
recover logits from the attention rows we already have.

Pipeline A (current):  p -> Hamilton-quantize(p*100) -> glyphs
Pipeline B (proposed): p -> round(log(p)*SCALE), floored -> glyphs

Measured: (1) noise survival at 1e-5 / 1e-4 / 1e-3 injected on p
          (2) fingerprint-space uniqueness over 300 prompts
          (3) boundary-distance profile: how close does each pipeline's
              nearest value sit to a quantization edge? (tail-risk proxy)
"""
import hashlib, json, math, random, time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GRID = 100
LOG_SCALE = 8          # logit grid: round(log(p) * 8)
LOG_FLOOR = -64        # probs below e^-8 all map here (prob floor)
N_FP_HEADS = 6
NOISE_TRIALS = 120
SPACE_N = 300
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

def quantize_logit(scores, scale=LOG_SCALE, floor_val=LOG_FLOOR):
    """FOUNDER-SOFTMAX: quantize in log domain. No sum constraint needed."""
    out = []
    for p in scores:
        if p <= 0:
            out.append(floor_val)
        else:
            out.append(max(floor_val, round(math.log(p) * scale)))
    return out

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

def heads_for_salt(salt, k=N_FP_HEADS):
    h = hashlib.sha256(salt.encode()).digest()
    rng = random.Random(int.from_bytes(h, 'big'))
    pairs = [(l, hd) for l in range(12) for hd in range(12)]
    return tuple(sorted(rng.sample(pairs, k)))

def get_rows(prompt, salt, heads):
    inputs = tok(f"{salt} {prompt}", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    return [out.attentions[l][0, h, -1].cpu().tolist() for (l, h) in heads]

def fp_from_rows(rows, heads, quantizer):
    parts = []
    for (l, h), row in zip(heads, rows):
        B, gl = compress_glyphs_int(quantizer(row))
        parts.append((l, h, B, gl))
    return json.dumps(parts, separators=(',', ':'))

VOCAB = """the a is was in on to and of it that for with as at by from or an
be this not but had has they we you all can will one my out if up so big old
new good long great small right came made after back only over take year some
could time very when what how said dog cat sun moon tree bird fish door red
blue dark light cold warm fast slow silver golden copper iron ember frost""".split()
rng = random.Random(7)
def rand_prompt():
    return " ".join(rng.choices(VOCAB, k=rng.randint(6, 14)))

salt = "logit_test_salt"; heads = heads_for_salt(salt)
print(f"Device: {DEVICE}   heads: {heads}\n")

# ---- 1) NOISE SURVIVAL -------------------------------------------------------
print("=" * 70)
print("1) NOISE SURVIVAL  (A = prob-domain grid, B = logit-domain 'FOUNDER-SOFTMAX')")
print("=" * 70)
noise_rng = random.Random(99)
for exp in [-5, -4, -3]:
    eps = 10 ** exp
    a_ok = b_ok = 0
    for _ in range(NOISE_TRIALS):
        p = rand_prompt()
        rows = get_rows(p, salt, heads)
        base_a = fp_from_rows(rows, heads, quantize_int)
        base_b = fp_from_rows(rows, heads, quantize_logit)
        noisy_rows = []
        for row in rows:
            noisy = [max(1e-12, v + noise_rng.uniform(-eps, eps)) for v in row]
            s = sum(noisy); noisy = [v / s for v in noisy]
            noisy_rows.append(noisy)
        if fp_from_rows(noisy_rows, heads, quantize_int) == base_a: a_ok += 1
        if fp_from_rows(noisy_rows, heads, quantize_logit) == base_b: b_ok += 1
    print(f"  noise 1e{exp}:   A prob-grid {a_ok:3d}/{NOISE_TRIALS}    "
          f"B logit-grid {b_ok:3d}/{NOISE_TRIALS}")

# ---- 2) FINGERPRINT SPACE ----------------------------------------------------
print("\n" + "=" * 70)
print("2) FINGERPRINT SPACE  (uniqueness over 300 prompts; degenerate = dead)")
print("=" * 70)
fps_a, fps_b = set(), set()
for _ in range(SPACE_N):
    p = rand_prompt()
    rows = get_rows(p, salt, heads)
    fps_a.add(fp_from_rows(rows, heads, quantize_int))
    fps_b.add(fp_from_rows(rows, heads, quantize_logit))
print(f"  A prob-grid : {len(fps_a)}/{SPACE_N} unique ({100*len(fps_a)/SPACE_N:.1f}%)")
print(f"  B logit-grid: {len(fps_b)}/{SPACE_N} unique ({100*len(fps_b)/SPACE_N:.1f}%)")

# ---- 3) BOUNDARY-DISTANCE PROFILE --------------------------------------------
print("\n" + "=" * 70)
print("3) BOUNDARY DISTANCE  (how close values sit to a grid edge; smaller = riskier)")
print("    measured as min distance of any value to its rounding boundary,")
print("    in units of the drift the value would need to flip")
print("=" * 70)
min_a, min_b = [], []
for _ in range(60):
    p = rand_prompt()
    rows = get_rows(p, salt, heads)
    for row in rows:
        # A: distance of p*100 fractional parts from 0.5 (Hamilton flip point)
        da = min(abs((v * GRID) % 1 - 0.5) / GRID for v in row)
        min_a.append(da)
        # B: distance of log(p)*8 from its rounding edge, converted back to
        # probability units: d(log)*p/scale approx flip drift
        db = min((abs((math.log(max(v, 1e-12)) * LOG_SCALE) % 1 - 0.5)
                  / LOG_SCALE) * max(v, 1e-12) for v in row)
        min_b.append(db)
import statistics
print(f"  A prob-grid : median flip-drift {statistics.median(min_a):.2e}   "
      f"worst {min(min_a):.2e}")
print(f"  B logit-grid: median flip-drift {statistics.median(min_b):.2e}   "
      f"worst {min(min_b):.2e}")
print("""
READING:
  survival higher + space still ~100% + flip-drift larger  => B wins, adopt it
  survival higher but space collapsed                      => B is a pygmy, reject
  survival equal or lower                                  => keep A, log-domain
                                                              amplified tail noise
""")
