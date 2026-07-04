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
# NEW: Multi-head fingerprint
# Use 5 diverse heads across different layers
# ================================================================
HEAD_CONFIG = [
    (0, 0), (2, 4), (5, 6), (8, 10), (11, 11)
]

def multi_head_fingerprint(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    
    all_glyphs = []
    all_Bs = []
    for layer, head in HEAD_CONFIG:
        scores = out.attentions[layer][0, head, -1].numpy().tolist()
        scores = quantize(scores)
        B, glyphs = compress_no_normalize(scores)
        all_glyphs.extend([(layer, head, g['level'], g['truncated']) for g in glyphs])
        all_Bs.append(round(B, 6))
    
    fp_str = json.dumps({'g': all_glyphs, 'b': all_Bs})
    h = hashlib.sha256(fp_str.encode()).hexdigest()
    return fp_str, h

def single_head_fingerprint(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    scores = out.attentions[5][0, 6, -1].numpy().tolist()
    scores = quantize(scores)
    B, glyphs = compress_no_normalize(scores)
    fp_str = json.dumps([(g['level'], g['truncated']) for g in glyphs]) + f"|{B:.6f}"
    h = hashlib.sha256(fp_str.encode()).hexdigest()
    return fp_str, h

# ================================================================
# TEST 1: FINGERPRINT SPACE — single vs multi-head
# ================================================================
print("=" * 64)
print("TEST 1: FINGERPRINT SPACE (single vs multi-head)")
print("=" * 64)

N = 3000
single_fps = set()
multi_fps = set()

t0 = time.time()
for i in range(N):
    p = random_prompt()
    sfp, _ = single_head_fingerprint(p)
    mfp, _ = multi_head_fingerprint(p)
    single_fps.add(sfp)
    multi_fps.add(mfp)
    if (i+1) % 500 == 0:
        print(f"  {i+1}: single={len(single_fps)} unique "
              f"({100*len(single_fps)/(i+1):.1f}%), "
              f"multi={len(multi_fps)} unique "
              f"({100*len(multi_fps)/(i+1):.1f}%)")

elapsed = time.time() - t0
print(f"\nFinal ({elapsed:.0f}s):")
print(f"  Single head: {len(single_fps)}/{N} unique ({100*len(single_fps)/N:.1f}%)")
print(f"  Multi head:  {len(multi_fps)}/{N} unique ({100*len(multi_fps)/N:.1f}%)")

# ================================================================
# TEST 2: DISTILLATION — multi-head is harder to distill
# ================================================================
print("\n" + "=" * 64)
print("TEST 2: DISTILLATION ATTACK ON MULTI-HEAD")
print("=" * 64)

MAXLEN = 14
def encode(p):
    ids = tokenizer(p, return_tensors="pt")['input_ids'][0][:MAXLEN]
    padded = torch.zeros(MAXLEN, dtype=torch.long)
    padded[:len(ids)] = ids
    return padded, len(ids)

# Collect training data: predict all 5 heads' scores
print("Collecting multi-head training data (2000 prompts)...")
Xd, Yd = [], []
for _ in range(2000):
    p = random_prompt()
    ids, L = encode(p)
    inputs = tokenizer(p, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    
    all_scores = []
    for layer, head in HEAD_CONFIG:
        s = out.attentions[layer][0, head, -1].numpy().tolist()
        padded = s[:MAXLEN] + [0]*(MAXLEN - len(s))
        all_scores.extend(padded[:MAXLEN])
    
    Xd.append(ids)
    Yd.append(torch.tensor(all_scores))

Xd = torch.stack(Xd); Yd = torch.stack(Yd)
OUT_DIM = len(HEAD_CONFIG) * MAXLEN

class DistillMulti(nn.Module):
    def __init__(self, vocab_size=50257, dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.net = nn.Sequential(
            nn.Linear(dim*MAXLEN, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, OUT_DIM)
        )
    def forward(self, ids):
        e = self.emb(ids).flatten(1)
        return torch.sigmoid(self.net(e))  # sigmoid since scores are 0-1

distill = DistillMulti()
opt = torch.optim.Adam(distill.parameters(), lr=1e-3)
print("Training multi-head distillation model (30 epochs)...")
for epoch in range(30):
    perm = torch.randperm(len(Xd))
    total_loss = 0
    for i in range(0, len(Xd), 64):
        idx = perm[i:i+64]
        pred = distill(Xd[idx])
        loss = ((pred - Yd[idx])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"  epoch {epoch+1}: loss {total_loss:.6f}")

# Evaluate
print("\nEvaluating multi-head distillation on 300 fresh prompts...")
matches = 0
for _ in range(300):
    p = random_prompt()
    real_fp, _ = multi_head_fingerprint(p)
    
    ids, L = encode(p)
    with torch.no_grad():
        pred = distill(ids.unsqueeze(0))[0]
    
    # Split pred into per-head scores and build fingerprint
    all_glyphs = []
    all_Bs = []
    ptr = 0
    inputs_t = tokenizer(p, return_tensors="pt")
    real_len = inputs_t['input_ids'].shape[1]
    for layer, head in HEAD_CONFIG:
        chunk = pred[ptr:ptr+MAXLEN].tolist()[:real_len]
        tot = sum(chunk)
        chunk = [v/tot for v in chunk] if tot > 0 else chunk
        chunk = quantize(chunk)
        B, glyphs = compress_no_normalize(chunk)
        all_glyphs.extend([(layer, head, g['level'], g['truncated']) for g in glyphs])
        all_Bs.append(round(B, 6))
        ptr += MAXLEN
    
    fake_fp = json.dumps({'g': all_glyphs, 'b': all_Bs})
    if fake_fp == real_fp:
        matches += 1

print(f"  Multi-head distilled matches: {matches}/300")
print(f"  (compare to 5/300 with single head)")

# ================================================================
# TEST 3: DIFFICULTY SCALING — multi-head
# ================================================================
print("\n" + "=" * 64)
print("TEST 3: DIFFICULTY SCALING (multi-head, 15 runs each)")
print("=" * 64)

for difficulty in [1, 2]:
    attempts_list = []
    for run in range(15):
        attempts = 0
        while attempts < 5000:
            attempts += 1
            p = random_prompt()
            _, h = multi_head_fingerprint(p)
            if h.startswith('0' * difficulty):
                break
        attempts_list.append(attempts)
    mean_a = np.mean(attempts_list)
    med_a = np.median(attempts_list)
    print(f"  Difficulty {difficulty}: mean {mean_a:.0f}, median {med_a:.0f} "
          f"(min {min(attempts_list)}, max {max(attempts_list)})")

# Expected: ~16x per hex zero if hash distribution is uniform
# Deviation indicates fingerprint-space lumpiness

# ================================================================
# TEST 4: NOISE TOLERANCE — multi-head
# ================================================================
print("\n" + "=" * 64)
print("TEST 4: NOISE TOLERANCE (multi-head)")
print("=" * 64)

for eps_exp in [-5, -4, -3]:
    eps = 10 ** eps_exp
    survived = 0
    TRIALS = 100
    for _ in range(TRIALS):
        p = random_prompt()
        base_fp, base_h = multi_head_fingerprint(p)
        
        # Re-derive with noise
        inputs = tokenizer(p, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        
        all_glyphs = []
        all_Bs = []
        for layer, head in HEAD_CONFIG:
            scores = out.attentions[layer][0, head, -1].numpy().tolist()
            noisy = [max(0, v + random.uniform(-eps, eps)) for v in scores]
            tot = sum(noisy); noisy = [v/tot for v in noisy]
            noisy = quantize(noisy)
            B, glyphs = compress_no_normalize(noisy)
            all_glyphs.extend([(layer, head, g['level'], g['truncated']) for g in glyphs])
            all_Bs.append(round(B, 6))
        
        noisy_fp = json.dumps({'g': all_glyphs, 'b': all_Bs})
        if noisy_fp == base_fp:
            survived += 1
    print(f"  noise 1e{eps_exp}: survived {survived}/{TRIALS}")

print("\n" + "=" * 64)
print("ALL MULTI-HEAD TESTS COMPLETE")
print("=" * 64)
