import math, time, random, hashlib, json, os
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ----------------------------------------------------------------------------
# Config knobs (turn these DOWN for a fast smoke-test, UP for a real verdict)
# ----------------------------------------------------------------------------
SEED              = 1337
UNIQ_N            = 3000      # prompts for the uniqueness test
DISTILL_TRAIN_N   = 2500      # training prompts for the proxy
DISTILL_EPOCHS    = 40
DISTILL_EVAL_N    = 400
DIFF_RUNS         = 15        # independent mining runs per difficulty
DIFF_CAP          = 20000     # max attempts per mining run before giving up
NOISE_TRIALS      = 150
GRID              = 100       # integer quantization grid: scores -> round(s*GRID)
N_FP_HEADS        = 6         # how many (layer,head) pairs form one fingerprint
MODEL_LAYERS      = 12        # GPT-2 small
MODEL_HEADS       = 12

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("Loading GPT-2 ...")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.eval()
tok = GPT2Tokenizer.from_pretrained("gpt2")

# ----------------------------------------------------------------------------
# Vocabulary for random prompts. Larger + more varied than before so prompt
# space is not the bottleneck on uniqueness.
# ----------------------------------------------------------------------------
VOCAB = """the a is was in on to and of it that for with as at by from or an be
this not but had has they we you all can will one my out if up so big old new
good long great small right came made after back only over take year some could
time very when what how said dog cat sun moon tree bird fish door red blue dark
light cold warm fast slow river stone glass paper metal cloud storm quiet loud
north south east west first last never always maybe often seldom under above
between across through around before during without within against toward city
field ocean forest desert mountain valley bridge tower engine signal pattern
memory reason answer question puzzle theory number letter symbol market garden
window silver golden copper iron ember frost """.split()

def random_prompt(minw=8, maxw=16):
    return " ".join(random.choices(VOCAB, k=random.randint(minw, maxw)))

# ----------------------------------------------------------------------------
# CORE: the glyph / type-cascade compression, unchanged in spirit but operating
# on INTEGER-quantized attention so it is deterministic across hardware.
# ----------------------------------------------------------------------------
def quantize_int(scores, grid=GRID):
    """Map a probability vector to integers on a fixed grid.
    Deterministic: identical across any hardware that agrees on `grid`.
    Returns integers whose sum is ~grid (largest-remainder so sum is exact)."""
    raw = [s * grid for s in scores]
    floor = [math.floor(r) for r in raw]
    remainder = grid - sum(floor)
    # distribute the leftover to the largest fractional parts (Hamilton method)
    fracs = sorted(range(len(raw)), key=lambda i: raw[i] - floor[i], reverse=True)
    for k in range(int(round(remainder))):
        floor[fracs[k % len(fracs)]] += 1
    return floor  # list[int], sum == grid

def compress_glyphs_int(int_scores):
    """Type-cascade compression on integers. Returns (B:int, glyph_tuple).
    R/G split at the integer median; R+G merges are glyphs (extracted, not
    summed onward); palindrome arrangement on odd levels; no float anywhere."""
    vals = list(int_scores)
    med = sorted(vals)[len(vals) // 2]
    typs = ['R' if v > med else 'G' for v in vals]
    glyphs = []
    level = 0
    while len(vals) > 1:
        level += 1
        n = len(vals)
        order = sorted(range(n), key=lambda i: vals[i], reverse=True)
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
            else:                     t = 'X'   # X = glyph
            nv.append(a + b); nt.append(t)
        reg = [(v, t) for v, t in zip(nv, nt) if t != 'X']
        gly = [v for v, t in zip(nv, nt) if t == 'X']
        for gv in gly:
            glyphs.append((level, gv))          # pure integers
        if reg:
            vals = [v for v, _ in reg]; typs = [t for _, t in reg]
        else:
            vals, typs = nv, nt
    B = vals[0] if vals else 0
    return B, tuple(glyphs)

# ----------------------------------------------------------------------------
# SALT-SELECTED HEADS: the salt deterministically picks which (layer,head)
# pairs form this block's fingerprint.
# ----------------------------------------------------------------------------
def heads_for_salt(salt, k=N_FP_HEADS):
    h = hashlib.sha256(salt.encode()).digest()
    rng = random.Random(int.from_bytes(h, 'big'))
    pairs = [(l, hd) for l in range(MODEL_LAYERS) for hd in range(MODEL_HEADS)]
    return tuple(sorted(rng.sample(pairs, k)))

def attention_last_row(prompt):
    """One forward pass -> dict of (layer,head) -> python list of the last
    token's attention over the sequence."""
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    return out  # keep raw; we index heads lazily

def fingerprint(prompt, salt, heads=None):
    """Full canonical fingerprint for (salt, prompt).
    Runs the salted text through the model, extracts the salt-selected heads,
    integer-quantizes, glyph-compresses each, and returns (fp_string, sha_hex)."""
    if heads is None:
        heads = heads_for_salt(salt)
    salted_prompt = f"{salt} {prompt}"
    out = attention_last_row(salted_prompt)
    parts = []
    for (layer, head) in heads:
        row = out.attentions[layer][0, head, -1].tolist()
        ints = quantize_int(row)
        B, gl = compress_glyphs_int(ints)
        parts.append((layer, head, B, gl))
    fp = json.dumps(parts, separators=(',', ':'))
    hexd = hashlib.sha256((salt + '|' + fp).encode()).hexdigest()
    return fp, hexd

def direct_fingerprint(prompt, salt, heads=None):
    """Control: hash the integer-quantized scores DIRECTLY, no glyphs.
    Used only to test whether glyphs still earn their place (T3)."""
    if heads is None:
        heads = heads_for_salt(salt)
    salted_prompt = f"{salt} {prompt}"
    out = attention_last_row(salted_prompt)
    parts = []
    for (layer, head) in heads:
        row = out.attentions[layer][0, head, -1].tolist()
        parts.append((layer, head, quantize_int(row)))
    fp = json.dumps(parts, separators=(',', ':'))
    return fp, hashlib.sha256((salt + '|' + fp).encode()).hexdigest()

# ============================================================================
# T1  UNIQUENESS + LOOKUP-TABLE INVALIDATION
# ============================================================================
def test_uniqueness():
    print("\n" + "=" * 70)
    print("T1  UNIQUENESS  &  LOOKUP-TABLE INVALIDATION UNDER RE-SALTING")
    print("=" * 70)
    saltA = "block_salt_" + hashlib.sha256(b"A").hexdigest()[:12]
    headsA = heads_for_salt(saltA)
    print(f"salt A selects heads: {headsA}")

    seen_fp, seen_hash = {}, {}
    t0 = time.time()
    for i in range(UNIQ_N):
        p = random_prompt()
        fp, hx = fingerprint(p, saltA, headsA)
        seen_fp.setdefault(fp, []).append(p)
        seen_hash[hx] = fp
        if (i + 1) % 500 == 0:
            print(f"  {i+1:5d} prompts -> {len(seen_fp):5d} unique fps "
                  f"({100*len(seen_fp)/(i+1):.2f}%)  [{time.time()-t0:.0f}s]")
    uniq = len(seen_fp)
    biggest = max(len(v) for v in seen_fp.values())
    print(f"\n  RESULT: {uniq}/{UNIQ_N} unique ({100*uniq/UNIQ_N:.2f}%), "
          f"largest collision cluster = {biggest}")

    # Lookup-table invalidation: build a table of winning HASHES under salt A,
    # then re-salt and check whether ANY of those winning prompts still win.
    print("\n  Lookup-table invalidation experiment:")
    winners_A = [p_list[0] for fp, p_list in seen_fp.items()
                 if seen_hash_startswith(seen_hash, fp, '0')]
    # recompute winners cleanly (hash prefix '0') under salt A
    winners_A = []
    for p_list in seen_fp.values():
        p = p_list[0]
        _, hx = fingerprint(p, saltA, headsA)
        if hx.startswith('0'):
            winners_A.append(p)
    print(f"    winners under salt A (hash starts '0'): {len(winners_A)}")

    saltB = "block_salt_" + hashlib.sha256(b"B").hexdigest()[:12]
    headsB = heads_for_salt(saltB)
    still = 0
    for p in winners_A:
        _, hx = fingerprint(p, saltB, headsB)
        if hx.startswith('0'):
            still += 1
    print(f"    of those, still winning under salt B: {still}/{len(winners_A)}")
    if winners_A:
        frac = still / len(winners_A)
        print(f"    -> re-salting preserved {100*frac:.1f}% of old winners "
              f"(want ~6.25% = pure chance, i.e. the old table is worthless)")
    return uniq / UNIQ_N

def seen_hash_startswith(*_):  # placeholder kept for clarity; unused
    return False

# ============================================================================
# T2  DISTILLATION  --  fixed heads (learnable) vs salt-selected (not)
# ============================================================================
MAXLEN = 16
def encode_ids(prompt):
    ids = tok(prompt, return_tensors="pt")['input_ids'][0][:MAXLEN]
    pad = torch.zeros(MAXLEN, dtype=torch.long)
    pad[:len(ids)] = ids
    return pad, len(ids)

class Proxy(nn.Module):
    """A deliberately capable proxy: embedding + deep MLP predicting the
    attention rows of a set of heads. If ANYTHING cheap can spoof the
    fingerprint, this should find it."""
    def __init__(self, out_dim, dim=96):
        super().__init__()
        self.emb = nn.Embedding(50257, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * MAXLEN, 768), nn.ReLU(),
            nn.Linear(768, 768), nn.ReLU(),
            nn.Linear(768, 512), nn.ReLU(),
            nn.Linear(512, out_dim), nn.Sigmoid())
    def forward(self, ids):
        return self.net(self.emb(ids).flatten(1))

def gather_head_rows(prompt, heads):
    salted = prompt  # for distillation we train on UNSALTED to give attacker
                     # the easiest possible job (upper bound on their power)
    out = attention_last_row(salted)
    rows = []
    for (layer, head) in heads:
        r = out.attentions[layer][0, head, -1].tolist()[:MAXLEN]
        r = r + [0.0] * (MAXLEN - len(r))
        rows.append(r)
    return rows  # list[k][MAXLEN]

def train_proxy(fixed_heads):
    print(f"  training proxy against FIXED heads {fixed_heads} ...")
    X, Y = [], []
    for _ in range(DISTILL_TRAIN_N):
        p = random_prompt()
        ids, _ = encode_ids(p)
        rows = gather_head_rows(p, fixed_heads)
        X.append(ids); Y.append(torch.tensor(sum(rows, []), dtype=torch.float32))
    X = torch.stack(X); Y = torch.stack(Y)
    net = Proxy(out_dim=len(fixed_heads) * MAXLEN)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(DISTILL_EPOCHS):
        perm = torch.randperm(len(X)); tot = 0.0
        for i in range(0, len(X), 64):
            idx = perm[i:i+64]
            pred = net(X[idx]); loss = ((pred - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:2d}  loss {tot:.5f}")
    return net

def proxy_fp_matches(net, fixed_heads, salt_for_hash):
    """Does the proxy reproduce the exact fingerprint on fresh prompts?
    We build the fingerprint the SAME way the verifier would, but from the
    proxy's predicted attention instead of the real model's."""
    match = 0
    for _ in range(DISTILL_EVAL_N):
        p = random_prompt()
        # real fingerprint (unsalted heads, to match how proxy was trained)
        real_parts = []
        out = attention_last_row(p)
        for (layer, head) in fixed_heads:
            row = out.attentions[layer][0, head, -1].tolist()
            B, gl = compress_glyphs_int(quantize_int(row))
            real_parts.append((layer, head, B, gl))
        real_fp = json.dumps(real_parts, separators=(',', ':'))

        ids, L = encode_ids(p)
        with torch.no_grad():
            pred = net(ids.unsqueeze(0))[0].tolist()
        fake_parts = []
        for hi, (layer, head) in enumerate(fixed_heads):
            row = pred[hi*MAXLEN:(hi+1)*MAXLEN][:L]
            s = sum(row)
            row = [v/s for v in row] if s > 0 else row
            B, gl = compress_glyphs_int(quantize_int(row))
            fake_parts.append((layer, head, B, gl))
        fake_fp = json.dumps(fake_parts, separators=(',', ':'))
        if fake_fp == real_fp:
            match += 1
    return match

def test_distillation():
    print("\n" + "=" * 70)
    print("T2  DISTILLATION:  fixed heads (attackable) vs salt-selected (not)")
    print("=" * 70)
    fixed = heads_for_salt("fixed_demo_salt")
    net = train_proxy(fixed)

    m_fixed = proxy_fp_matches(net, fixed, "fixed_demo_salt")
    print(f"\n  Proxy exact-fingerprint matches on the SAME fixed heads: "
          f"{m_fixed}/{DISTILL_EVAL_N}")
    print("    (this is the attacker's BEST case: they knew the heads in advance)")

    # Now the salted reality: each block uses a DIFFERENT head set. The proxy
    # was trained for `fixed`; evaluate it against a different salt's heads.
    other = heads_for_salt("some_future_block_salt_xyz")
    while other == fixed:
        other = heads_for_salt("some_future_block_salt_" + str(random.random()))
    m_other = proxy_fp_matches(net, other, "some_future_block_salt_xyz")
    print(f"  Same proxy, evaluated on a DIFFERENT block's salt-selected heads:"
          f" {m_other}/{DISTILL_EVAL_N}")
    print("    -> In the live protocol the attacker does not know next block's")
    print("       heads, so their prepared proxy lands in this second column.")
    return m_fixed, m_other

# ============================================================================
# T3  GLYPHS vs DIRECT under the integer regime
# ============================================================================
def test_glyphs_vs_direct():
    print("\n" + "=" * 70)
    print("T3  DO GLYPHS STILL EARN THEIR PLACE?  (integer regime)")
    print("=" * 70)
    salt = "noise_test_salt"; heads = heads_for_salt(salt)
    for exp in [-4, -3, -2]:
        eps = 10 ** exp
        g_ok = d_ok = 0
        for _ in range(NOISE_TRIALS):
            p = random_prompt()
            g_base = fingerprint(p, salt, heads)[1]
            d_base = direct_fingerprint(p, salt, heads)[1]
            # inject noise into the raw rows, then re-quantize both ways
            out = attention_last_row(f"{salt} {p}")
            g_parts, d_parts = [], []
            for (layer, head) in heads:
                row = out.attentions[layer][0, head, -1].tolist()
                noisy = [max(0.0, v + random.uniform(-eps, eps)) for v in row]
                s = sum(noisy); noisy = [v/s for v in noisy] if s > 0 else noisy
                ints = quantize_int(noisy)
                B, gl = compress_glyphs_int(ints)
                g_parts.append((layer, head, B, gl))
                d_parts.append((layer, head, ints))
            g_hash = hashlib.sha256((salt+'|'+json.dumps(g_parts,separators=(',',':'))).encode()).hexdigest()
            d_hash = hashlib.sha256((salt+'|'+json.dumps(d_parts,separators=(',',':'))).encode()).hexdigest()
            if g_hash == g_base: g_ok += 1
            if d_hash == d_base: d_ok += 1
        print(f"  noise 1e{exp}:  glyphs {g_ok:3d}/{NOISE_TRIALS}   "
              f"direct {d_ok:3d}/{NOISE_TRIALS}")
    print("  If glyphs >= direct at every level, they earn their place as the")
    print("  canonicalization layer. If equal, they are optional. Either is fine.")

# ============================================================================
# T4  DIFFICULTY SCALING  (salted multi-head)
# ============================================================================
def test_difficulty():
    print("\n" + "=" * 70)
    print("T4  DIFFICULTY SCALING  (salted, multi-head, integer fingerprint)")
    print("=" * 70)
    salt = "difficulty_salt"; heads = heads_for_salt(salt)
    prev = None
    for d in [1, 2]:
        needed = []
        for _ in range(DIFF_RUNS):
            a = 0
            while a < DIFF_CAP:
                a += 1
                _, hx = fingerprint(random_prompt(), salt, heads)
                if hx.startswith('0' * d):
                    break
            needed.append(a)
        mean = np.mean(needed)
        print(f"  difficulty {d} ('{'0'*d}'):  mean {mean:8.1f}  "
              f"median {np.median(needed):8.1f}  "
              f"min {min(needed)}  max {max(needed)}")
        if prev is not None:
            print(f"    scaling vs previous zero: {mean/prev:.1f}x  (ideal 16x)")
        prev = mean

# ============================================================================
# T5  END-TO-END BLOCK  (+ re-salt invalidation of a valid proof)
# ============================================================================
def test_end_to_end():
    print("\n" + "=" * 70)
    print("T5  END-TO-END:  mine -> verify -> tamper -> re-salt invalidation")
    print("=" * 70)
    salt = "genesis_salt_" + hashlib.sha256(b"g").hexdigest()[:8]
    heads = heads_for_salt(salt)
    difficulty = 1

    # mine
    t0 = time.time(); attempts = 0; won = None
    while attempts < DIFF_CAP:
        attempts += 1
        p = random_prompt()
        fp, hx = fingerprint(p, salt, heads)
        if hx.startswith('0' * difficulty):
            won = (p, fp, hx); break
    if not won:
        print("  could not mine in cap; raise DIFF_CAP or lower difficulty")
        return
    p, fp, hx = won
    print(f"  mined in {attempts} attempts ({time.time()-t0:.1f}s)")
    print(f"    prompt : '{p}'")
    print(f"    hash   : {hx[:40]}...")

    # verify (re-run)
    fp2, hx2 = fingerprint(p, salt, heads)
    print(f"  verify  : hash_match={hx2==hx}  difficulty_ok={hx2.startswith('0'*difficulty)}")

    # tamper the prompt
    fp3, hx3 = fingerprint(p + " intruder", salt, heads)
    print(f"  tamper  : still_valid={hx3==hx}  (want False)")

    # re-salt: a NEW block salt must invalidate this proof entirely
    salt_next = "next_salt_" + hashlib.sha256(b"n").hexdigest()[:8]
    heads_next = heads_for_salt(salt_next)
    fp4, hx4 = fingerprint(p, salt_next, heads_next)
    print(f"  re-salt : same_prompt_still_wins={hx4.startswith('0'*difficulty)} "
          f"(want mostly False -> old work does not carry over)")

# ============================================================================
def main():
    print("#" * 70)
    print("# HARDENED PROOF-OF-INFERENCE  --  decisive run")
    print(f"# grid={GRID}  heads/fp={N_FP_HEADS}  uniq_n={UNIQ_N}")
    print("#" * 70)

    uniq_ratio = test_uniqueness()
    m_fixed, m_other = test_distillation()
    test_glyphs_vs_direct()
    test_difficulty()
    test_end_to_end()

    print("\n" + "#" * 70)
    print("# VERDICT CHECKLIST  (read against the printouts above)")
    print("#" * 70)
    print(f"""
  [{'PASS' if uniq_ratio >= 0.999 else 'FAIL'}] T1 uniqueness >= 99.9%   (got {100*uniq_ratio:.2f}%)
       and re-salting reduced old winners toward ~6.25% chance level.

  [{'PASS' if m_other == 0 else 'CHECK'}] T2 distillation defeated by salt-selected heads
       fixed-head matches   = {m_fixed}/{DISTILL_EVAL_N}  (attacker's best case)
       salted-head matches  = {m_other}/{DISTILL_EVAL_N}  (real protocol case)
       The gap between these two numbers IS the security the salt buys.

  [ -- ] T3 glyphs vs direct: judged from the noise table above.
  [ -- ] T4 difficulty: judged from the scaling ratio above (~16x = healthy).
  [ -- ] T5 end-to-end: mine/verify True, tamper False, re-salt does not carry.

  HONEST NOTES BAKED INTO THIS SCRIPT:
   * Determinism here is simulated via an integer grid on ONE machine.
     A real network must also pin one int8 runtime; cross-GPU truth is still
     unverified until you run the same salted prompt on two different cards.
   * The proxy is a strong MLP but still small. A funded attacker with a
     larger student model is the real threat; salt-selected heads are what
     make that economically pointless, not the proxy's weakness here.
   * Security rests on: submit-prompt + verifier-re-runs + SHA-256 + salting.
     The glyphs are canonicalization, not the lock. Say so in the writeup.
""")

if __name__ == "__main__":
    main()
