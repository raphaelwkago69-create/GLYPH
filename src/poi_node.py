"""
==============================================================================
POI-NODE v0.2 -- Proof-of-Inference blockchain node
==============================================================================
Upgrades over poi_chain.py v0.1:
  * Numeric difficulty target (Bitcoin-style hash < target), smooth retargeting
    every RETARGET_INTERVAL blocks toward TARGET_BLOCK_TIME
  * Coinbase rewards + real wallets (ECDSA secp256k1) + signed transactions
  * Balance validation (no overspend, no forged signatures, no double-spend
    within the chain)
  * Vetted MODEL_REGISTRY (whitelist) with protocol version -- models enter by
    upgrade, never by network observation (see poison_test.py for why)
  * Proof bound to the miner: salt = H(prev_hash || miner_address), so a
    broadcast winning prompt cannot be stolen and re-claimed by someone else
  * Fork resolution by cumulative work (most-work chain wins, not longest)

CLI:
  python poi_node.py wallet NAME          create/show wallet NAME
  python poi_node.py mine N [wallet]      mine N blocks, rewards to wallet
  python poi_node.py send FROM TO AMOUNT  queue a signed transaction
  python poi_node.py balance [addr]       balances from chain state
  python poi_node.py verify               full chain verification
  python poi_node.py show                 print chain summary
Files: poi_chain_v2.json (chain), poi_wallets.json (keys), poi_mempool.json
==============================================================================
"""
import hashlib, json, math, os, random, sys, time

# ------------------------------------------------------------ protocol -----
PROTOCOL_VERSION = 2
MODEL_REGISTRY = {
    # vetted by audit (determinism / fingerprint-space / cost benchmarks);
    # additions require a protocol version bump adopted by the whole network
    "gpt2": {"layers": 12, "heads": 12, "tier": 1},
}
ACTIVE_MODEL       = "gpt2"
GRID               = 100
N_FP_HEADS         = 6
BLOCK_REWARD       = 6
TARGET_BLOCK_TIME  = 20          # seconds (demo value)
RETARGET_INTERVAL  = 5           # blocks
MAX_RETARGET_SHIFT = 4.0         # clamp per-retarget factor, like Bitcoin
GENESIS_TARGET     = int("0fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 16)
MAX_ATTEMPTS       = 200000
# POI_PREFIX lets two nodes run from the same folder without clobbering
# each other's files (used by the local two-node test)
_P = os.environ.get("POI_PREFIX", "")
CHAIN_FILE   = _P + "poi_chain_v2.json"
WALLET_FILE  = _P + "poi_wallets.json"
MEMPOOL_FILE = _P + "poi_mempool.json"

# ------------------------------------------------------------ model --------
_model = _tok = _device = None
def load_model():
    global _model, _tok, _device
    if _model is not None:
        return
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[node] loading {ACTIVE_MODEL} on {_device} ...")
    _model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True)
    _model.eval()
    _model = _model.to(_device)
    _tok = GPT2Tokenizer.from_pretrained("gpt2")

# ------------------------------------------------- fingerprint core --------
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

def heads_for_salt(salt, model_id=ACTIVE_MODEL, k=N_FP_HEADS):
    spec = MODEL_REGISTRY[model_id]
    h = hashlib.sha256(salt.encode()).digest()
    rng = random.Random(int.from_bytes(h, 'big'))
    pairs = [(l, hd) for l in range(spec["layers"]) for hd in range(spec["heads"])]
    return tuple(sorted(rng.sample(pairs, k)))

def inference_hash(prompt, salt):
    import torch
    load_model()
    heads = heads_for_salt(salt)
    inputs = _tok(f"{salt} {prompt}", return_tensors="pt").to(_device)
    with torch.no_grad():
        out = _model(**inputs, output_attentions=True)
    parts = []
    for (layer, head) in heads:
        row = out.attentions[layer][0, head, -1].cpu().tolist()
        B, gl = compress_glyphs_int(quantize_int(row))
        parts.append((layer, head, B, gl))
    fp = json.dumps(parts, separators=(',', ':'))
    return hashlib.sha256((salt + '|' + fp).encode()).hexdigest()

# ------------------------------------------------------------ wallets ------
import ecdsa

def load_wallets():
    if os.path.exists(WALLET_FILE):
        with open(WALLET_FILE) as f:
            return json.load(f)
    return {}

def save_wallets(w):
    with open(WALLET_FILE, "w") as f:
        json.dump(w, f, indent=1)

def make_wallet(name):
    wallets = load_wallets()
    if name in wallets:
        return wallets[name]
    sk = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
    vk = sk.get_verifying_key()
    pub = vk.to_string().hex()
    addr = hashlib.sha256(bytes.fromhex(pub)).hexdigest()[:40]
    wallets[name] = {"private": sk.to_string().hex(), "public": pub, "address": addr}
    save_wallets(wallets)
    print(f"[wallet] created '{name}'  address={addr}")
    return wallets[name]

def tx_signing_payload(tx):
    return json.dumps({k: tx[k] for k in ("from", "to", "amount", "nonce", "pubkey")},
                      separators=(',', ':'), sort_keys=True).encode()

def sign_tx(tx, private_hex):
    sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_hex), curve=ecdsa.SECP256k1)
    tx["signature"] = sk.sign_deterministic(tx_signing_payload(tx)).hex()
    return tx

def verify_tx_signature(tx):
    try:
        pub = bytes.fromhex(tx["pubkey"])
        addr = hashlib.sha256(pub).hexdigest()[:40]
        if addr != tx["from"]:
            return False  # pubkey does not own the 'from' address
        vk = ecdsa.VerifyingKey.from_string(pub, curve=ecdsa.SECP256k1)
        return vk.verify(bytes.fromhex(tx["signature"]), tx_signing_payload(tx))
    except Exception:
        return False

# ------------------------------------------------------------ blocks -------
def block_header_hash(block):
    header = json.dumps({k: block[k] for k in
        ("version", "index", "prev_hash", "timestamp", "model_id",
         "transactions", "miner", "prompt", "proof_hash", "target")},
        separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(header.encode()).hexdigest()

def salt_for_block(prev_block, miner_addr):
    # binding the miner address makes winning prompts non-stealable:
    # a relayed proof only verifies for the address that mined it
    return "salt_" + hashlib.sha256(
        (prev_block["block_hash"] + "|" + miner_addr).encode()).hexdigest()[:16]

def genesis_block():
    g = {"version": PROTOCOL_VERSION, "index": 0, "prev_hash": "0" * 64,
         "timestamp": 0, "model_id": ACTIVE_MODEL, "transactions": [],
         "miner": "genesis", "prompt": "genesis", "proof_hash": "0" * 64,
         "target": f"{GENESIS_TARGET:064x}"}
    g["block_hash"] = block_header_hash(g)
    return g

def expected_target(chain, index):
    """Deterministic difficulty schedule: every node computes the same target
    for block `index` from chain history alone."""
    if index <= RETARGET_INTERVAL:
        return GENESIS_TARGET
    last = ((index - 1) // RETARGET_INTERVAL) * RETARGET_INTERVAL
    first = last - RETARGET_INTERVAL
    span = max(1, chain[last]["timestamp"] - chain[first]["timestamp"])
    expected = TARGET_BLOCK_TIME * RETARGET_INTERVAL
    factor = span / expected
    factor = max(1 / MAX_RETARGET_SHIFT, min(MAX_RETARGET_SHIFT, factor))
    prev_target = int(chain[last]["target"], 16)
    new_target = int(prev_target * factor)          # slower than wanted -> easier
    return min(new_target, GENESIS_TARGET)          # never easier than genesis

def block_work(block):
    return (1 << 256) // (int(block["target"], 16) + 1)

def chain_work(chain):
    return sum(block_work(b) for b in chain[1:])

# ------------------------------------------------------------ state --------
def compute_balances(chain, upto=None):
    """Replay the chain into {address: balance}. Returns None if any economic
    rule is violated (bad signature, overspend, wrong coinbase, reused nonce)."""
    bal, nonces = {}, {}
    for block in chain[1:upto]:
        txs = block["transactions"]
        if not txs or txs[0].get("type") != "coinbase":
            return None
        cb = txs[0]
        if cb["amount"] != BLOCK_REWARD or cb["to"] != block["miner"]:
            return None
        bal[cb["to"]] = bal.get(cb["to"], 0) + cb["amount"]
        for tx in txs[1:]:
            if tx.get("type") == "coinbase":
                return None                      # only one coinbase per block
            if not verify_tx_signature(tx):
                return None
            if tx["amount"] <= 0:
                return None
            if nonces.get(tx["from"], -1) >= tx["nonce"]:
                return None                      # replayed / out-of-order nonce
            if bal.get(tx["from"], 0) < tx["amount"]:
                return None                      # overspend
            nonces[tx["from"]] = tx["nonce"]
            bal[tx["from"]] -= tx["amount"]
            bal[tx["to"]] = bal.get(tx["to"], 0) + tx["amount"]
    return bal

# ------------------------------------------------------------ mempool ------
def load_mempool():
    if os.path.exists(MEMPOOL_FILE):
        with open(MEMPOOL_FILE) as f:
            return json.load(f)
    return []

def save_mempool(mp):
    with open(MEMPOOL_FILE, "w") as f:
        json.dump(mp, f, indent=1)

# ------------------------------------------------------------ chain io -----
def load_chain():
    if os.path.exists(CHAIN_FILE):
        with open(CHAIN_FILE) as f:
            return json.load(f)
    return [genesis_block()]

def save_chain(chain):
    with open(CHAIN_FILE, "w") as f:
        json.dump(chain, f, indent=1)

# ------------------------------------------------------------ mining -------
VOCAB = """the a is was in on to and of it that for with as at by from or an
be this not but had has they we you all can will one my out if up so big old
new good long great small right came made after back only over take year some
could time very when what how said dog cat sun moon tree bird fish door red
blue dark light cold warm fast slow river stone glass paper metal cloud storm
field ocean forest desert mountain valley bridge tower engine signal pattern
memory reason answer question puzzle theory number letter symbol market garden
window silver golden copper iron ember frost""".split()

def mine_block(chain, miner_addr, transactions=None, quiet=False):
    prev = chain[-1]
    index = prev["index"] + 1
    target = expected_target(chain, index)
    salt = salt_for_block(prev, miner_addr)
    rng = random.Random()
    t0 = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = " ".join(rng.choices(VOCAB, k=rng.randint(6, 16)))
        hx = inference_hash(prompt, salt)
        if int(hx, 16) < target:
            coinbase = {"type": "coinbase", "to": miner_addr, "amount": BLOCK_REWARD}
            block = {"version": PROTOCOL_VERSION, "index": index,
                     "prev_hash": prev["block_hash"],
                     "timestamp": int(time.time()),
                     "model_id": ACTIVE_MODEL,
                     "transactions": [coinbase] + (transactions or []),
                     "miner": miner_addr, "prompt": prompt, "proof_hash": hx,
                     "target": f"{target:064x}"}
            block["block_hash"] = block_header_hash(block)
            if not quiet:
                dt = time.time() - t0
                print(f"[mine] block {index}: won in {attempt} attempts "
                      f"({dt:.1f}s, {attempt/max(dt,1e-9):.0f} inf/s)  "
                      f"reward {BLOCK_REWARD} -> {miner_addr[:12]}...")
            return block
    raise RuntimeError("MAX_ATTEMPTS exceeded")

# ------------------------------------------------------------ validation ---
def verify_block(block, prev, chain_upto):
    """All consensus rules for one block. chain_upto = chain[:block.index]."""
    errs = []
    if block["version"] > PROTOCOL_VERSION:
        errs.append("unknown protocol version")
    if block["model_id"] not in MODEL_REGISTRY:
        errs.append(f"model '{block['model_id']}' not in vetted registry")
    if block["prev_hash"] != prev["block_hash"]:
        errs.append("broken prev_hash link")
    if block_header_hash(block) != block["block_hash"]:
        errs.append("header hash mismatch")
    want = expected_target(chain_upto, block["index"])
    if int(block["target"], 16) != want:
        errs.append("target does not follow difficulty schedule")
    if int(block["proof_hash"], 16) >= int(block["target"], 16):
        errs.append("proof does not meet target")
    if not errs and block["model_id"] in MODEL_REGISTRY:
        salt = salt_for_block(prev, block["miner"])
        hx = inference_hash(block["prompt"], salt)     # the ONE inference
        if hx != block["proof_hash"]:
            errs.append(f"proof mismatch: recomputed {hx[:16]}...")
    return errs

def verify_chain(chain, verbose=True):
    if block_header_hash(chain[0]) != chain[0]["block_hash"]:
        if verbose: print("[verify] GENESIS CORRUPT")
        return False
    ok = True
    for i in range(1, len(chain)):
        errs = verify_block(chain[i], chain[i - 1], chain[:i])
        if verbose:
            print(f"[verify] block {i}: {'OK' if not errs else 'BAD  <- ' + '; '.join(errs)}")
        if errs:
            ok = False
    if compute_balances(chain) is None:
        if verbose: print("[verify] ECONOMIC RULES VIOLATED (sig/overspend/coinbase)")
        ok = False
    return ok

def resolve_fork(local, remote):
    """Adopt remote iff it is fully valid and has strictly more work."""
    if remote[0] != local[0]:
        return local, "rejected: different genesis"
    if chain_work(remote) <= chain_work(local):
        return local, "kept local: remote has <= work"
    if not verify_chain(remote, verbose=False):
        return local, "rejected: remote chain invalid"
    return remote, "adopted remote: more work and fully valid"

# ------------------------------------------------------------ p2p ----------
def serve(port=9401):
    """Serve this node's chain over HTTP. GET /chain -> full chain JSON,
    GET /status -> height + total work + tip hash."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            chain = load_chain()
            if self.path == "/chain":
                body = json.dumps(chain).encode()
            elif self.path == "/status":
                body = json.dumps({
                    "height": len(chain) - 1,
                    "work": str(chain_work(chain)),
                    "tip": chain[-1]["block_hash"],
                    "model": ACTIVE_MODEL,
                    "protocol": PROTOCOL_VERSION}).encode()
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, fmt, *args):
            print(f"[serve] {self.client_address[0]} {fmt % args}")
    print(f"[serve] node listening on 0.0.0.0:{port}  (GET /chain, /status)")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

def sync(peer_url):
    """Fetch the peer's chain, fully verify it (one inference per block),
    adopt it iff it has more cumulative work. Returns the outcome string."""
    from urllib.request import urlopen
    peer_url = peer_url.rstrip("/")
    with urlopen(peer_url + "/status", timeout=10) as r:
        st = json.load(r)
    print(f"[sync] peer height={st['height']} tip={st['tip'][:16]}... "
          f"model={st['model']}")
    local = load_chain()
    if st["tip"] == local[-1]["block_hash"]:
        print("[sync] already in sync"); return "in-sync"
    with urlopen(peer_url + "/chain", timeout=60) as r:
        remote = json.load(r)
    print(f"[sync] fetched {len(remote)-1} blocks; verifying "
          f"(re-running {len(remote)-1} inferences) ...")
    kept, why = resolve_fork(local, remote)
    print(f"[sync] {why}")
    if kept is remote:
        save_chain(remote)
        print(f"[sync] local chain now height {len(remote)-1}")
    return why

# ------------------------------------------------------------ cli ----------
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    chain = load_chain()
    if cmd == "wallet":
        w = make_wallet(sys.argv[2])
        print(f"[wallet] '{sys.argv[2]}' address: {w['address']}")
    elif cmd == "mine":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        wname = sys.argv[3] if len(sys.argv) > 3 else "miner"
        w = make_wallet(wname)
        mp = load_mempool()
        for _ in range(n):
            block = mine_block(chain, w["address"], transactions=mp)
            chain.append(block)
            save_chain(chain)
            mp = []; save_mempool(mp)
        print(f"[node] height {len(chain)-1}, total work {chain_work(chain):,}")
    elif cmd == "send":
        frm, to, amt = sys.argv[2], sys.argv[3], int(sys.argv[4])
        wallets = load_wallets()
        w = wallets[frm]
        to_addr = wallets[to]["address"] if to in wallets else to
        bal = compute_balances(chain) or {}
        mp = load_mempool()
        used = [t["nonce"] for t in mp if t["from"] == w["address"]]
        chain_nonce = max([t.get("nonce", -1) for b in chain[1:]
                           for t in b["transactions"][1:]
                           if t.get("from") == w["address"]], default=-1)
        tx = {"from": w["address"], "to": to_addr, "amount": amt,
              "nonce": max(used + [chain_nonce]) + 1, "pubkey": w["public"]}
        tx = sign_tx(tx, w["private"])
        mp.append(tx); save_mempool(mp)
        print(f"[tx] queued {amt} from {frm} -> {to} (balance {bal.get(w['address'], 0)})")
    elif cmd == "balance":
        bal = compute_balances(chain)
        if bal is None:
            print("[balance] CHAIN VIOLATES ECONOMIC RULES"); sys.exit(1)
        names = {w["address"]: n for n, w in load_wallets().items()}
        for addr, amount in sorted(bal.items(), key=lambda x: -x[1]):
            print(f"  {names.get(addr, addr[:16] + '...'):>12} : {amount}")
    elif cmd == "verify":
        print(f"[node] verifying {len(chain)-1} blocks ...")
        ok = verify_chain(chain)
        print(f"[node] CHAIN {'VALID' if ok else 'INVALID'}")
        sys.exit(0 if ok else 1)
    elif cmd == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 9401)
    elif cmd == "sync":
        sync(sys.argv[2])
    elif cmd == "show":
        for b in chain:
            ntx = len(b["transactions"])
            print(f"  #{b['index']}  {b['block_hash'][:16]}...  "
                  f"target={b['target'][:8]}  txs={ntx}  miner={str(b['miner'])[:12]}")
        print(f"  total work: {chain_work(chain):,}")
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
