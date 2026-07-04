"""
==============================================================================
HARDENED CROSS-HARDWARE DETERMINISM TEST  (v2)
==============================================================================
Differences from real_cross_hardware.py:
  1. Heads are SALT-SELECTED per prompt (the real protocol), not hardcoded.
  2. quantize_int handles negative/zero remainder explicitly (spec-safe).
  3. 100 prompts generated from a salted deterministic RNG, not 10 fixed ones.
  4. Prints per-prompt hashes plus one ULTIMATE HASH for comparison.

Run this EXACT file on every machine. Compare the ULTIMATE HASH line.
==============================================================================
"""
import hashlib, json, math, platform, random, sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GRID         = 100
N_FP_HEADS   = 6
MODEL_LAYERS = 12
MODEL_HEADS  = 12
N_PROMPTS    = 100
RUN_SEED     = "hardened_xhw_v2"   # protocol constant: same on all machines

if torch.cuda.is_available():
    DEVICE = torch.device('cuda'); GPU_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = torch.device('cpu');  GPU_NAME = "CPU only"

print("=" * 70)
print("HARDWARE REPORT")
print("=" * 70)
print(f"  Platform:    {platform.platform()}")
print(f"  Processor:   {platform.processor()}")
print(f"  Python:      {sys.version.split()[0]}")
print(f"  PyTorch:     {torch.__version__}")
print(f"  Device:      {DEVICE}  ({GPU_NAME})")
print("=" * 70)

print("\nLoading GPT-2 ...")
model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
model.eval()
model = model.to(DEVICE)
tok = GPT2Tokenizer.from_pretrained("gpt2")

VOCAB = """the a is was in on to and of it that for with as at by from or an be
this not but had has they we you all can will one my out if up so big old new
good long great small right came made after back only over take year some could
time very when what how said dog cat sun moon tree bird fish door red blue dark
light cold warm fast slow river stone glass paper metal cloud storm quiet loud
north south east west first last never always maybe often seldom under above
between across through around before during without within against toward city
field ocean forest desert mountain valley bridge tower engine signal pattern
memory reason answer question puzzle theory number letter symbol market garden
window silver golden copper iron ember frost""".split()

def quantize_int(scores, grid=GRID):
    raw = [s * grid for s in scores]
    floor = [math.floor(r) for r in raw]
    remainder = grid - sum(floor)
    if remainder > 0:
        fracs = sorted(range(len(raw)), key=lambda i: (raw[i] - floor[i], -i), reverse=True)
        for k in range(remainder):
            floor[fracs[k % len(fracs)]] += 1
    elif remainder < 0:
        # row summed above 1.0 after float multiply: remove from smallest fracs
        fracs = sorted(range(len(raw)), key=lambda i: (raw[i] - floor[i], -i))
        for k in range(-remainder):
            floor[fracs[k % len(fracs)]] -= 1
    return floor  # sum == grid, always

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

def heads_for_salt(salt, k=N_FP_HEADS):
    h = hashlib.sha256(salt.encode()).digest()
    rng = random.Random(int.from_bytes(h, 'big'))
    pairs = [(l, hd) for l in range(MODEL_LAYERS) for hd in range(MODEL_HEADS)]
    return tuple(sorted(rng.sample(pairs, k)))

def fingerprint(prompt, salt):
    heads = heads_for_salt(salt)
    inputs = tok(f"{salt} {prompt}", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    parts = []
    for (layer, head) in heads:
        row = out.attentions[layer][0, head, -1].cpu().tolist()
        B, gl = compress_glyphs_int(quantize_int(row))
        parts.append((layer, head, B, gl))
    fp = json.dumps(parts, separators=(',', ':'))
    return hashlib.sha256((salt + '|' + fp).encode()).hexdigest()

# Deterministic prompt + salt generation (identical on every machine)
gen = random.Random(RUN_SEED)
cases = []
for i in range(N_PROMPTS):
    salt = f"salt_{hashlib.sha256(f'{RUN_SEED}_{i}'.encode()).hexdigest()[:12]}"
    prompt = " ".join(gen.choices(VOCAB, k=gen.randint(6, 16)))
    cases.append((salt, prompt))

print("\n" + "=" * 70)
print(f"RUNNING {N_PROMPTS} SALT-SELECTED-HEAD FINGERPRINTS ON {DEVICE}")
print("=" * 70)

results = []
for i, (salt, prompt) in enumerate(cases):
    hx = fingerprint(prompt, salt)
    results.append(hx)
    if (i + 1) % 10 == 0:
        print(f"  {i+1:3d}/{N_PROMPTS}  last hash: {hx[:24]}...")

ultimate = hashlib.sha256("".join(results).encode()).hexdigest()
print("\n" + "=" * 70)
print("ULTIMATE HASH (compare across machines)")
print("=" * 70)
print(f"  {ultimate}")
print("=" * 70)

# Save full per-prompt hashes for drift localization if ultimates differ
outfile = "hardened_xhw_hashes.json"
with open(outfile, "w") as f:
    json.dump({"device": str(DEVICE), "gpu": GPU_NAME,
               "ultimate": ultimate, "hashes": results}, f, indent=1)
print(f"\nPer-prompt hashes saved to {outfile}")
print("If ultimates differ between machines, diff the two JSON files to find")
print("exactly which prompts diverged, then inspect those heads' raw floats.")
