"""
OVERNIGHT LARGE-MODEL TEST
Part 1: GPU-vs-CPU determinism cross-check on Qwen2.5-0.5B (fp32 both).
Part 2: Full quality suite on Qwen2.5-1.5B (CPU fp32): fingerprint space,
        noise survival, mining speed.
Part 3: (best effort) Qwen2.5-3B mini quality suite; skipped gracefully if
        RAM is insufficient.
Everything logged to overnight_results.txt as it goes.
"""
import hashlib, json, math, random, sys, time, traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GRID = 100; N_FP_HEADS = 6
LOG = open("overnight_results.txt", "a", buffering=1)
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line); LOG.write(line + "\n")

def quantize_int(scores, grid=GRID):
    raw = [s * grid for s in scores]
    floor = [math.floor(r) for r in raw]
    rem = grid - sum(floor)
    if rem > 0:
        fr = sorted(range(len(raw)), key=lambda i: (raw[i] - floor[i], -i), reverse=True)
        for k in range(rem): floor[fr[k % len(fr)]] += 1
    elif rem < 0:
        fr = sorted(range(len(raw)), key=lambda i: (raw[i] - floor[i], -i))
        for k in range(-rem): floor[fr[k % len(fr)]] -= 1
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
    return tuple(sorted(rng.sample(pairs, k)))

def fingerprint(model, tokzr, prompt, salt, n_layers, n_heads, device):
    heads = heads_for_salt(salt, n_layers, n_heads)
    inputs = tokzr(f"{salt} {prompt}", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    parts = []
    for (l, h) in heads:
        row = out.attentions[l][0, h, -1].float().cpu().tolist()
        B, gl = compress_glyphs_int(quantize_int(row))
        parts.append((l, h, B, gl))
    fp = json.dumps(parts, separators=(',', ':'))
    return hashlib.sha256((salt + '|' + fp).encode()).hexdigest(), fp

VOCAB = """the a is was in on to and of it that for with as at by from or an
be this not but had has they we you all can will one my out if up so big old
new good long great small right came made after back only over take year some
could time very when what how said dog cat sun moon tree bird fish door red
blue dark light cold warm fast slow silver golden copper iron ember frost""".split()

def prompts_for(seed, n):
    r = random.Random(seed)
    return [" ".join(r.choices(VOCAB, k=r.randint(6, 14))) for _ in range(n)]

def load(name, device, dtype=torch.float32):
    log(f"loading {name} on {device} ({dtype}) ...")
    m = AutoModelForCausalLM.from_pretrained(name, output_attentions=True,
                                             attn_implementation="eager",
                                             torch_dtype=dtype)
    m.eval(); m = m.to(device)
    t = AutoTokenizer.from_pretrained(name)
    cfg = m.config
    nl = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    nh = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    log(f"  {name}: {nl} layers x {nh} heads, "
        f"{sum(p.numel() for p in m.parameters())/1e9:.2f}B params")
    return m, t, nl, nh

def quality_suite(name, model, tokzr, nl, nh, device, n_space=200, n_noise=50):
    salt = "overnight_salt"
    # speed
    ps = prompts_for(1, 30)
    t0 = time.time()
    for p in ps:
        fingerprint(model, tokzr, p, salt, nl, nh, device)
    speed = 30 / (time.time() - t0)
    log(f"  [{name}] speed: {speed:.2f} fingerprints/sec")
    # space
    fps = set()
    for i, p in enumerate(prompts_for(2, n_space)):
        _, fp = fingerprint(model, tokzr, p, salt, nl, nh, device)
        fps.add(fp)
        if (i + 1) % 50 == 0:
            log(f"  [{name}] space progress {i+1}/{n_space} ({len(fps)} unique)")
    log(f"  [{name}] fingerprint space: {len(fps)}/{n_space} "
        f"({100*len(fps)/n_space:.1f}%)")
    # noise survival at 1e-4
    nrng = random.Random(9); ok = 0
    heads = heads_for_salt(salt, nl, nh)
    for p in prompts_for(3, n_noise):
        inputs = tokzr(f"{salt} {p}", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_attentions=True)
        base_parts, noisy_parts = [], []
        for (l, h) in heads:
            row = out.attentions[l][0, h, -1].float().cpu().tolist()
            B, gl = compress_glyphs_int(quantize_int(row))
            base_parts.append((l, h, B, gl))
            noisy = [max(1e-12, v + nrng.uniform(-1e-4, 1e-4)) for v in row]
            s = sum(noisy); noisy = [v / s for v in noisy]
            B2, gl2 = compress_glyphs_int(quantize_int(noisy))
            noisy_parts.append((l, h, B2, gl2))
        if base_parts == noisy_parts: ok += 1
    log(f"  [{name}] noise 1e-4 survival: {ok}/{n_noise}")

log("=" * 66)
log("OVERNIGHT RUN START")
log("=" * 66)

# ---- PART 1: cross-device determinism, Qwen2.5-0.5B fp32 -------------------
try:
    log("PART 1: Qwen2.5-0.5B GPU-vs-CPU determinism (fp32, 60 prompts)")
    name = "Qwen/Qwen2.5-0.5B"
    prompts = prompts_for(42, 60)
    hashes = {}
    for device in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        m, t, nl, nh = load(name, device)
        hs = []
        for i, p in enumerate(prompts):
            salt = f"xdet_{hashlib.sha256(str(i).encode()).hexdigest()[:8]}"
            hx, _ = fingerprint(m, t, p, salt, nl, nh, device)
            hs.append(hx)
        hashes[device] = hs
        ult = hashlib.sha256("".join(hs).encode()).hexdigest()
        log(f"  [{device}] ultimate hash: {ult}")
        del m
        if device == "cuda": torch.cuda.empty_cache()
    if "cuda" in hashes:
        same = hashes["cuda"] == hashes["cpu"]
        log(f"  PART 1 VERDICT: GPU==CPU bit-for-bit: {same}")
        if not same:
            diverged = [i for i, (a, b) in enumerate(zip(hashes['cuda'], hashes['cpu'])) if a != b]
            log(f"  diverged prompt indices: {diverged}")
except Exception:
    log("PART 1 FAILED:\n" + traceback.format_exc())

# ---- PART 2: quality suite, Qwen2.5-1.5B on CPU -----------------------------
try:
    log("PART 2: Qwen2.5-1.5B quality suite (CPU fp32) -- the 12x scale test")
    m, t, nl, nh = load("Qwen/Qwen2.5-1.5B", "cpu")
    quality_suite("Qwen2.5-1.5B", m, t, nl, nh, "cpu")
    del m
except Exception:
    log("PART 2 FAILED:\n" + traceback.format_exc())

# ---- PART 3: best-effort Qwen2.5-3B mini suite ------------------------------
try:
    log("PART 3 (best effort): Qwen2.5-3B mini suite (CPU fp32, may exceed RAM)")
    m, t, nl, nh = load("Qwen/Qwen2.5-3B", "cpu")
    quality_suite("Qwen2.5-3B", m, t, nl, nh, "cpu", n_space=60, n_noise=20)
    del m
except Exception as e:
    log(f"PART 3 skipped/failed ({type(e).__name__}): likely RAM. Not a problem.")

log("=" * 66)
log("OVERNIGHT RUN COMPLETE")
log("=" * 66)