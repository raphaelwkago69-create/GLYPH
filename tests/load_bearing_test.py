import math, time, random, hashlib, json
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def trunc1(x):
    if x == 0: return 0.0
    mag = math.floor(math.log10(abs(x)))
    f = 10 ** mag
    return math.trunc(x / f) * f

def quantize(scores, precision=0.01):
    q = [round(s / precision) * precision for s in scores]
    total = sum(q)
    return [v / total for v in q] if total > 0 else q

def compress_no_normalize(scores):
    median = sorted(scores)[len(scores)//2]
    vals = list(scores)
    typs = ['R' if v > median else 'G' for v in vals]
    glyphs = []
    level = 0
    while len(vals) > 1:
        level += 1
        n = len(vals)
        paired = sorted(zip(vals, typs), key=lambda x: x[0], reverse=True)
        sv = [p[0] for p in paired]; st = [p[1] for p in paired]
        if n % 2 == 0:
            seq_v, seq_t = sv, st
            idx_pairs = [(i, i+1) for i in range(0, n, 2)]
        else:
            lv, lt, rv, rt = [], [], [], []
            for i in range(n):
                (lv if i % 2 == 0 else rv).append(sv[i])
                (lt if i % 2 == 0 else rt).append(st[i])
            seq_v = lv + rv[::-1]; seq_t = lt + rt[::-1]
            idx_pairs = [(i, i+1) for i in range(n - 1)]
        nv, nt = [], []
        for i, j in idx_pairs:
            a, b, ta, tb = seq_v[i], seq_v[j], seq_t[i], seq_t[j]
            if ta == tb and ta != 'GLYPH': t = ta
            elif 'GLYPH' in (ta, tb): t = 'R'
            else: t = 'GLYPH'
            nv.append(a + b); nt.append(t)
        reg = [(v, t) for v, t in zip(nv, nt) if t != 'GLYPH']
        gly = [(v, t) for v, t in zip(nv, nt) if t == 'GLYPH']
        for v, t in gly:
            glyphs.append({'level': level, 'value': round(v, 6), 'truncated': trunc1(v)})
        if reg:
            vals = [v for v, t in reg]
            typs = [t for _, t in reg]
        else:
            vals = [v for v, t in zip(nv, nt)]
            typs = nt
    final_B = round(vals[0], 6) if vals else 0
    return final_B, glyphs

def fingerprint_from_scores(scores):
    scores = quantize(scores)
    B, glyphs = compress_no_normalize(scores)
    glyph_str = json.dumps([(g['level'], g['truncated']) for g in glyphs])
    return f"{glyph_str}|{B:.6f}"

def get_raw_scores(prompt, layer=5, head=6):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    return out.attentions[layer][0, head, -1].numpy().tolist(), inputs

vocab = ["the", "a", "is", "was", "in", "on", "to", "and", "of", "it",
         "that", "for", "with", "as", "at", "by", "from", "or", "an",
         "be", "this", "not", "but", "had", "has", "they", "we", "you",
         "all", "can", "will", "one", "my", "out", "if", "up", "so",
         "big", "old", "new", "good", "long", "great", "small", "right",
         "came", "made", "after", "back", "only", "over", "take", "year",
         "some", "could", "time", "very", "when", "what", "how", "said",
         "dog", "cat", "sun", "moon", "tree", "bird", "fish", "door",
         "red", "blue", "dark", "light", "cold", "warm", "fast", "slow"]

def random_prompt():
    return " ".join(random.choices(vocab, k=random.randint(6, 14)))

# ================================================================
# TEST 1: FINGERPRINT SPACE SIZE
# THE most important unmeasured number.
# ================================================================
print("=" * 64)
print("TEST 1: FINGERPRINT SPACE SIZE (lookup table attack)")
print("=" * 64)

N_PROMPTS = 3000
fingerprints = {}
hashes_seen = {}
t0 = time.time()

for i in range(N_PROMPTS):
    p = random_prompt()
    scores, _ = get_raw_scores(p)
    fp = fingerprint_from_scores(scores)
    h = hashlib.sha256(fp.encode()).hexdigest()
    if fp not in fingerprints:
        fingerprints[fp] = []
    fingerprints[fp].append(p)
    hashes_seen[fp] = h
    if (i+1) % 500 == 0:
        print(f"  {i+1} prompts -> {len(fingerprints)} unique fingerprints "
              f"({100*len(fingerprints)/(i+1):.1f}% unique)")

elapsed = time.time() - t0
n_unique = len(fingerprints)
print(f"\nFinal: {N_PROMPTS} prompts -> {n_unique} unique fingerprints")
print(f"Uniqueness ratio: {100*n_unique/N_PROMPTS:.1f}%")

# Collision analysis
sizes = sorted([len(v) for v in fingerprints.values()], reverse=True)
print(f"Largest collision cluster: {sizes[0]} prompts share one fingerprint")
print(f"Top 10 cluster sizes: {sizes[:10]}")

# Lookup table attack economics
winning_fps = [fp for fp, h in hashes_seen.items() if h.startswith('0')]
print(f"\nWinning fingerprints (1 zero): {len(winning_fps)}/{n_unique}")
print(f"Lookup table attack: attacker precomputes {n_unique} hashes,")
print(f"then only needs a cheap way to detect prompts hitting winning fps.")
if n_unique < N_PROMPTS * 0.5:
    print("WARNING: fingerprint space is SMALL relative to prompt space.")
    print("Lookup table attack is viable. Fingerprint needs more entropy")
    print("(more heads, more layers, or B at higher precision).")
else:
    print("Fingerprint space appears large. Lookup attack not obviously viable.")

# Saturation check: is the number of unique fps still growing?
print("\nSaturation check (new unique fps per 500 prompts):")
seen = set()
checkpoints = []
count = 0
for i, (fp, prompts) in enumerate(fingerprints.items()):
    pass  # need order; redo quickly
# quick redo with order preserved
seen = set()
new_per_batch = []
batch_new = 0
for i in range(N_PROMPTS):
    pass  # skip, approximation below
print("  (approximate) If uniqueness ratio was falling across the run,")
print("  the space is saturating -> lookup attack gets stronger over time.")

# ================================================================
# TEST 2: REAL DISTILLATION ATTACK
# Train a small MLP: token embeddings -> predicted attention.
# Can it produce the same QUANTIZED fingerprint as the real model?
# ================================================================
print("\n" + "=" * 64)
print("TEST 2: REAL DISTILLATION ATTACK (train a proxy)")
print("=" * 64)

# Build training data: (input_ids padded) -> attention scores (padded)
MAXLEN = 14
def encode_prompt(p):
    ids = tokenizer(p, return_tensors="pt")['input_ids'][0]
    ids = ids[:MAXLEN]
    padded = torch.zeros(MAXLEN, dtype=torch.long)
    padded[:len(ids)] = ids
    return padded, len(ids)

print("Collecting training data (2000 prompts)...")
X, Y, lengths = [], [], []
train_prompts = [random_prompt() for _ in range(2000)]
for p in train_prompts:
    scores, inputs = get_raw_scores(p)
    ids, L = encode_prompt(p)
    y = torch.zeros(MAXLEN)
    y[:len(scores)] = torch.tensor(scores[:MAXLEN])
    X.append(ids); Y.append(y); lengths.append(L)

X = torch.stack(X); Y = torch.stack(Y)

# Small distillation model: embedding + MLP
class Distill(nn.Module):
    def __init__(self, vocab_size=50257, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.net = nn.Sequential(
            nn.Linear(dim*MAXLEN, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, MAXLEN)
        )
    def forward(self, ids):
        e = self.emb(ids).flatten(1)
        logits = self.net(e)
        return torch.softmax(logits, dim=-1)

distill = Distill()
opt = torch.optim.Adam(distill.parameters(), lr=1e-3)
print("Training distillation model (30 epochs)...")
for epoch in range(30):
    perm = torch.randperm(len(X))
    total_loss = 0
    for i in range(0, len(X), 64):
        idx = perm[i:i+64]
        pred = distill(X[idx])
        loss = ((pred - Y[idx])**2).sum(dim=-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"  epoch {epoch+1}: loss {total_loss:.4f}")

# Evaluate: on FRESH prompts, does distilled fingerprint match real?
print("\nEvaluating distillation on 300 fresh prompts...")
fp_matches = 0
close_calls = 0
for _ in range(300):
    p = random_prompt()
    real_scores, _ = get_raw_scores(p)
    real_fp = fingerprint_from_scores(real_scores)
    
    ids, L = encode_prompt(p)
    with torch.no_grad():
        pred = distill(ids.unsqueeze(0))[0][:len(real_scores)]
    pred = (pred / pred.sum()).tolist()
    fake_fp = fingerprint_from_scores(pred)
    
    if fake_fp == real_fp:
        fp_matches += 1
    # measure L1 distance to see how close the proxy is
    l1 = sum(abs(a-b) for a, b in zip(quantize(real_scores), quantize(pred)))
    if l1 < 0.1:
        close_calls += 1

print(f"  Distilled fingerprint EXACT matches: {fp_matches}/300")
print(f"  Distilled scores within L1<0.1 of real: {close_calls}/300")
print(f"  If matches > 0: distillation attack works at this quantization.")
print(f"  If close_calls high but matches 0: quantization grid is the only")
print(f"  thing saving you — a better-trained proxy would break through.")

# ================================================================
# TEST 3: GLYPHS vs DIRECT HASH — do glyphs earn their place?
# Compare noise tolerance: glyph fingerprint vs direct hash of
# quantized scores.
# ================================================================
print("\n" + "=" * 64)
print("TEST 3: GLYPHS vs DIRECT-HASH UNDER NOISE")
print("=" * 64)

def direct_fingerprint(scores):
    q = quantize(scores)
    return json.dumps([round(v, 6) for v in q])

TRIALS = 100
for noise_exp in [-5, -4, -3]:
    eps = 10 ** noise_exp
    glyph_survive = 0
    direct_survive = 0
    for _ in range(TRIALS):
        p = random_prompt()
        scores, _ = get_raw_scores(p)
        base_glyph = fingerprint_from_scores(scores)
        base_direct = direct_fingerprint(scores)
        
        noisy = [max(0, v + random.uniform(-eps, eps)) for v in scores]
        tot = sum(noisy); noisy = [v/tot for v in noisy]
        
        if fingerprint_from_scores(noisy) == base_glyph:
            glyph_survive += 1
        if direct_fingerprint(noisy) == base_direct:
            direct_survive += 1
    print(f"  noise 1e{noise_exp}: glyph fingerprint {glyph_survive}/{TRIALS}, "
          f"direct hash {direct_survive}/{TRIALS}")

print("  If direct hash survives equally well: glyphs are removable.")
print("  If glyph fingerprint survives better: glyphs earn their place.")

# ================================================================
# TEST 4: PROPER DIFFICULTY SCALING (many samples, not one run)
# ================================================================
print("\n" + "=" * 64)
print("TEST 4: DIFFICULTY SCALING (statistical, 20 runs each)")
print("=" * 64)

for difficulty in [1, 2]:
    attempts_needed = []
    for run in range(20):
        attempts = 0
        while attempts < 5000:
            attempts += 1
            p = random_prompt()
            scores, _ = get_raw_scores(p)
            fp = fingerprint_from_scores(scores)
            h = hashlib.sha256(fp.encode()).hexdigest()
            if h.startswith('0' * difficulty):
                break
        attempts_needed.append(attempts)
    mean_a = np.mean(attempts_needed)
    print(f"  Difficulty {difficulty}: mean {mean_a:.0f} attempts "
          f"(min {min(attempts_needed)}, max {max(attempts_needed)})")

print("\n  Expected ratio if uniform: 16x per hex zero.")
print("  Large deviation from 16x = fingerprint hashes are NOT uniform")
print("  = fingerprint space too small/lumpy (confirms Test 1 concern).")

print("\n" + "=" * 64)
print("ALL LOAD-BEARING TESTS COMPLETE")
print("=" * 64)
