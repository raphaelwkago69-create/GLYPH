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
  python poi_node.py run [wallet] [port]  FULL NODE: serve + mine + gossip
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

# Give every outgoing HTTP request a normal User-Agent. Some seed hosts sit
# behind a CDN/proxy (e.g. Cloudflare) that 403s the default "Python-urllib"
# agent as a suspected bot, which would silently block newcomers from joining.
import urllib.request as _urlreq
_opener = _urlreq.build_opener()
_opener.addheaders = [("User-Agent", "glyph-node/0.3")]
_urlreq.install_opener(_opener)

# ------------------------------------------------------------ protocol -----
# v4: proof switched to the integer-only engine (int_infer.py). Fingerprints
# are exact integer arithmetic end to end, so the proof hash is bit-identical
# on every device BY CONSTRUCTION -- the v3 cross-hardware boundary flip
# (mainnet block 1693) is impossible in this protocol.
# v5: OPTIMISTIC INFERENCE MARKET (docs/USEFUL_WORK.md). New transaction
# types job/result/challenge/verdict turn the network into a decentralized
# inference service: users escrow fees for prompts, servers post real model
# outputs bonded by stake, disputes are settled by consensus re-running the
# ONE disputed generation (exact integer decode -> a lie is objective).
# CONSENSUS CHANGE: v4 nodes reject v5 blocks; the live network must adopt
# v5 for the market to run on mainnet. Until then this is a testnet protocol.
PROTOCOL_VERSION = 5
# --- market parameters ---
JOB_STAKE_MULT     = 10     # server bonds fee*this to post a result
CHALLENGE_WINDOW   = 30     # blocks a result can be disputed (~10 min)
JOB_EXPIRY         = 90     # blocks before an unserved job refunds its poster
JOB_MAX_TOKENS     = 64     # generation length (protocol constant: verdicts
                            # re-run generate(prompt, JOB_MAX_TOKENS))
JOB_MAX_OUTPUT     = 4000   # chars of output allowed on-chain (data availability)
JOB_MAX_PROMPT     = 1000
MODEL_REGISTRY = {
    # vetted by audit (determinism / fingerprint-space / cost benchmarks);
    # additions require a protocol version bump adopted by the whole network.
    # int_weights_hash pins the deterministic float->integer weight
    # conversion: a node whose converted weights hash differently is running
    # different weights and must not mine on them.
    "gpt2": {"layers": 12, "heads": 12, "tier": 1,
             "int_weights_hash": "842a00bc8f09c1e6eb870e750deaa49159dd45fb9ab860fc8f40bef6878029ac"},
}
ACTIVE_MODEL       = "gpt2"
GRID               = 100
N_FP_HEADS         = 6
# All amounts are integers in the smallest unit: 1 GLY = 100 units.
# (Divisibility, like Bitcoin's satoshis; also makes the halving sum clean.)
UNITS_PER_GLY      = 100
INITIAL_REWARD     = 700         # units = 7.00 GLY; halves per era below
HALVING_INTERVAL   = 1_500_000   # blocks per era (~347 days at 20s blocks);
                                 # 700>>k summed over eras -> 20.91M GLY cap

def block_reward(height):
    """Coinbase reward in units at a given height. 700 -> 350 -> 175 -> ...
    -> 0 after era 10; total supply converges to ~20,910,000 GLY."""
    return INITIAL_REWARD >> (height // HALVING_INTERVAL)

def fmt_gly(units):
    return f"{units / UNITS_PER_GLY:.2f}"
TARGET_BLOCK_TIME  = 20          # seconds
RETARGET_INTERVAL  = 5           # blocks
MAX_RETARGET_SHIFT = 4.0         # clamp per-retarget factor, like Bitcoin
# Timestamp consensus rules (Bitcoin's median-time-past + future limit,
# scaled to 20s blocks). Timestamps feed difficulty retargeting, so without
# these a miner could time-warp the schedule with fabricated clock values.
TIMESTAMP_WINDOW   = 11          # blocks in the median-time-past window
MAX_FUTURE_DRIFT   = 600         # seconds a timestamp may run ahead of a
                                 # validator's clock (Bitcoin uses 2h @ 10min)
MAX_PEERS          = 64          # peer-table cap (anti peer-list poisoning)
# Verification-DoS defenses (node policy, not consensus): a chain that fails
# full verification cost us real inference, so the submitting IP is banned;
# and POST /chain is rate-limited per IP so one host cannot keep the inbox
# churning even with valid-looking (higher-claimed-work) junk.
BAN_SECONDS        = 3600        # how long an IP that fed us an invalid chain is banned
POST_MIN_INTERVAL  = 10          # seconds between accepted POSTs from one IP
# Checkpoints (node policy): {height: block_hash} pins. A remote chain that
# disagrees with a pinned hash is rejected without verification, whatever its
# claimed work. This is Bitcoin's assumevalid/checkpoint defense: a majority
# attacker with rented GPUs can extend the tip, but can never rewrite history
# below a checkpoint on nodes that pin it. Set POI_CHECKPOINTS="h:hash,h:hash".
def _load_checkpoints():
    cps = {}
    for part in os.environ.get("POI_CHECKPOINTS", "").split(","):
        if ":" in part:
            h, hx = part.split(":", 1)
            cps[int(h.strip())] = hx.strip()
    return cps
CHECKPOINTS = _load_checkpoints()
# Known public nodes tried automatically before mining on a fresh chain.
# Priority: POI_SEEDS env var > live SEEDS.txt on GitHub > baked-in fallback.
# The GitHub fetch means seed URLs can be updated for all future nodes by
# editing one file in the repo — no code re-download needed.
SEEDS_URL = ("https://raw.githubusercontent.com/raphaelwkago69-create/"
             "GLYPH/main/SEEDS.txt")
def _live_seeds():
    try:
        from urllib.request import urlopen
        with urlopen(SEEDS_URL, timeout=10) as r:
            lines = r.read(65536).decode().splitlines()
        return [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    except Exception:
        return []
def get_seed_nodes():
    env = [s for s in os.environ.get("POI_SEEDS", "").split(",") if s]
    return env or _live_seeds() or [
        # baked-in fallback (SEEDS.txt on GitHub is the live list)
        "https://glyph.surfacedplus.com",
    ]
GENESIS_TARGET     = int("0fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", 16)
MAX_ATTEMPTS       = 200000
# Gossip: mining runs in chunks of this many attempts; between chunks the
# node checks peers/inbox so it never grinds long on a stale tip.
GOSSIP_CHUNK       = int(os.environ.get("POI_CHUNK", "150"))
MAX_SYNC_BYTES     = 256 * 1024 * 1024   # refuse to download/accept chains above this
# POI_PREFIX lets two nodes run from the same folder without clobbering
# each other's files (used by the local two-node test)
_P = os.environ.get("POI_PREFIX", "")
CHAIN_FILE   = _P + "poi_chain_v2.json"
WALLET_FILE  = _P + "poi_wallets.json"
MEMPOOL_FILE = _P + "poi_mempool.json"
PEERS_FILE   = _P + "poi_peers.json"
INBOX_FILE   = _P + "poi_inbox.json"
BANS_FILE    = _P + "poi_bans.json"

# ------------------------------------------- model + fingerprint (v4) ------
# The float model and float fingerprint path are GONE from consensus.
# int_infer.py runs GPT-2 in exact integer arithmetic; heads_for_salt, the
# GRID quantizer and glyph compression live there (same GRID / N_FP_HEADS
# protocol constants as above).
import int_infer

heads_for_salt = int_infer.heads_for_salt   # re-export for tools/tests

def load_model():
    int_infer.load_model()
    want = MODEL_REGISTRY[ACTIVE_MODEL]["int_weights_hash"]
    if int_infer.WEIGHTS_HASH != want:
        raise RuntimeError(
            "integer weight conversion mismatch: got "
            f"{int_infer.WEIGHTS_HASH}, protocol pins {want} -- "
            "your gpt2 download is not the vetted one")

def inference_hash(prompt, salt):
    load_model()
    return int_infer.inference_hash(prompt, salt)

def job_output(prompt):
    """The market's canonical answer for a prompt: deterministic integer
    greedy decode. Every honest node computes the same string, so a result
    that differs from this is provably fraudulent."""
    load_model()
    return int_infer.generate(prompt, JOB_MAX_TOKENS)

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

# signed fields per transaction type; transfers keep the original v4 key set
# so their signatures remain valid unchanged
_TX_SIGNED_KEYS = {
    "transfer":  ("from", "to", "amount", "nonce", "pubkey"),
    "job":       ("type", "from", "prompt", "fee", "nonce", "pubkey"),
    "result":    ("type", "from", "job_id", "output", "nonce", "pubkey"),
    "challenge": ("type", "from", "job_id", "nonce", "pubkey"),
}

def tx_signing_payload(tx):
    keys = _TX_SIGNED_KEYS[tx.get("type", "transfer")]
    return json.dumps({k: tx[k] for k in keys},
                      separators=(',', ':'), sort_keys=True).encode()

def job_id_of(job_tx):
    """A job's identity is the hash of its signed content (nonce makes it
    unique per poster)."""
    return hashlib.sha256(tx_signing_payload(job_tx)).hexdigest()[:24]

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

def median_time_past(chain_upto):
    """Median timestamp of the last TIMESTAMP_WINDOW blocks (Bitcoin's MTP).
    A new block's timestamp may not be before this value, so no miner can
    rewind the clock to game difficulty retargeting."""
    ts = sorted(b["timestamp"] for b in chain_upto[-TIMESTAMP_WINDOW:])
    return ts[len(ts) // 2]

def block_work(block):
    return (1 << 256) // (int(block["target"], 16) + 1)

def chain_work(chain):
    return sum(block_work(b) for b in chain[1:])

# ------------------------------------------------------------ state --------
def compute_state(chain, upto=None):
    """Replay the chain into (balances, jobs). Returns (None, None) if any
    economic rule is violated (bad signature, overspend, wrong coinbase,
    reused nonce, or an invalid market transition).

    Market lifecycle (docs/USEFUL_WORK.md):
      job:       poster escrows `fee`                        -> status open
      result:    server posts output, bonds fee*STAKE_MULT   -> status served
      (window passes unchallenged): server paid fee+stake    -> status paid
      (JOB_EXPIRY with no result):  poster refunded          -> status expired
      challenge: challenger bonds `fee`                      -> status challenged
      verdict:   miner-inserted; verify_block re-ran the generation, so the
                 `honest` field here is trusted to be consensus-checked.
                 honest server:      stake+fee+bond -> server
                 honest challenger:  bond + stake/2 -> challenger,
                                     stake - stake/2 burned, fee -> poster"""
    bal, nonces, jobs = {}, {}, {}
    def credit(a, v): bal[a] = bal.get(a, 0) + v
    def spend(a, v):
        if v <= 0 or bal.get(a, 0) < v: return False
        bal[a] -= v; return True
    for block in chain[1:upto]:
        h = block["index"]
        # settlements that fall due at this height, before this block's txs
        for j in jobs.values():
            if j["status"] == "open" and h >= j["posted"] + JOB_EXPIRY:
                credit(j["poster"], j["fee"]); j["status"] = "expired"
            elif j["status"] == "served" and h > j["served"] + CHALLENGE_WINDOW:
                credit(j["server"], j["stake"] + j["fee"]); j["status"] = "paid"
        txs = block["transactions"]
        if not txs or txs[0].get("type") != "coinbase":
            return None, None
        cb = txs[0]
        if cb["amount"] != block_reward(h) or cb["to"] != block["miner"]:
            return None, None
        credit(cb["to"], cb["amount"])
        for tx in txs[1:]:
            t = tx.get("type", "transfer")
            if t == "coinbase":
                return None, None                # only one coinbase per block
            if t == "verdict":
                # unsigned, miner-inserted; correctness of `honest` is checked
                # in verify_block by re-running the generation
                j = jobs.get(tx.get("job_id"))
                if j is None or j["status"] != "challenged":
                    return None, None
                if tx.get("honest") == "server":
                    credit(j["server"], j["stake"] + j["fee"] + j["bond"])
                elif tx.get("honest") == "challenger":
                    half = j["stake"] // 2       # other half burned
                    credit(j["challenger"], j["bond"] + half)
                    credit(j["poster"], j["fee"])
                else:
                    return None, None
                j["status"] = "closed:" + tx["honest"]
                continue
            if not verify_tx_signature(tx):
                return None, None
            if nonces.get(tx["from"], -1) >= tx["nonce"]:
                return None, None                # replayed / out-of-order nonce
            nonces[tx["from"]] = tx["nonce"]
            if t == "transfer":
                if tx["amount"] <= 0 or not spend(tx["from"], tx["amount"]):
                    return None, None
                credit(tx["to"], tx["amount"])
            elif t == "job":
                if not (0 < len(tx["prompt"]) <= JOB_MAX_PROMPT):
                    return None, None
                if not spend(tx["from"], tx["fee"]):
                    return None, None            # fee escrowed
                jobs[job_id_of(tx)] = {"poster": tx["from"], "prompt": tx["prompt"],
                                       "fee": tx["fee"], "posted": h,
                                       "status": "open"}
            elif t == "result":
                j = jobs.get(tx["job_id"])
                if j is None or j["status"] != "open":
                    return None, None
                if len(tx["output"]) > JOB_MAX_OUTPUT:
                    return None, None
                stake = j["fee"] * JOB_STAKE_MULT
                if not spend(tx["from"], stake):
                    return None, None            # stake bonded
                j.update(server=tx["from"], output=tx["output"], stake=stake,
                         served=h, status="served")
            elif t == "challenge":
                j = jobs.get(tx["job_id"])
                if j is None or j["status"] != "served" \
                        or h > j["served"] + CHALLENGE_WINDOW:
                    return None, None
                if not spend(tx["from"], j["fee"]):
                    return None, None            # bond = the job fee
                j.update(challenger=tx["from"], bond=j["fee"], status="challenged")
            else:
                return None, None                # unknown transaction type
    return bal, jobs

def compute_balances(chain, upto=None):
    return compute_state(chain, upto)[0]

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
    # atomic: the serving process reads this file concurrently; write-then-
    # rename means it can never observe a half-written chain
    tmp = CHAIN_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(chain, f, indent=1)
    os.replace(tmp, CHAIN_FILE)

# ------------------------------------------------------------ mining -------
VOCAB = """the a is was in on to and of it that for with as at by from or an
be this not but had has they we you all can will one my out if up so big old
new good long great small right came made after back only over take year some
could time very when what how said dog cat sun moon tree bird fish door red
blue dark light cold warm fast slow river stone glass paper metal cloud storm
field ocean forest desert mountain valley bridge tower engine signal pattern
memory reason answer question puzzle theory number letter symbol market garden
window silver golden copper iron ember frost""".split()

def mine_block(chain, miner_addr, transactions=None, quiet=False, max_attempts=None):
    """Mine one block. With max_attempts set (gossip chunk mode), returns
    None when the budget runs out instead of raising, so the caller can
    check peers and resume on a possibly-new tip."""
    prev = chain[-1]
    index = prev["index"] + 1
    target = expected_target(chain, index)
    salt = salt_for_block(prev, miner_addr)
    rng = random.Random()
    t0 = time.time()
    budget = max_attempts or MAX_ATTEMPTS
    for attempt in range(1, budget + 1):
        prompt = " ".join(rng.choices(VOCAB, k=rng.randint(6, 16)))
        hx = inference_hash(prompt, salt)
        if int(hx, 16) < target:
            coinbase = {"type": "coinbase", "to": miner_addr, "amount": block_reward(index)}
            block = {"version": PROTOCOL_VERSION, "index": index,
                     "prev_hash": prev["block_hash"],
                     # never below MTP even if our clock is skewed backwards
                     "timestamp": max(int(time.time()), median_time_past(chain)),
                     "model_id": ACTIVE_MODEL,
                     "transactions": [coinbase] + (transactions or []),
                     "miner": miner_addr, "prompt": prompt, "proof_hash": hx,
                     "target": f"{target:064x}"}
            block["block_hash"] = block_header_hash(block)
            if not quiet:
                dt = time.time() - t0
                print(f"[mine] block {index}: won in {attempt} attempts "
                      f"({dt:.1f}s, {attempt/max(dt,1e-9):.0f} inf/s)  "
                      f"reward {fmt_gly(block_reward(index))} GLY -> {miner_addr[:12]}...")
            return block
    if max_attempts is not None:
        return None
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
    if block["timestamp"] < median_time_past(chain_upto):
        errs.append("timestamp before median-time-past")
    if block["timestamp"] > int(time.time()) + MAX_FUTURE_DRIFT:
        errs.append("timestamp too far in the future")
    if int(block["proof_hash"], 16) >= int(block["target"], 16):
        errs.append("proof does not meet target")
    if not errs and block["model_id"] in MODEL_REGISTRY:
        salt = salt_for_block(prev, block["miner"])
        hx = inference_hash(block["prompt"], salt)     # the ONE inference
        if hx != block["proof_hash"]:
            errs.append(f"proof mismatch: recomputed {hx[:16]}...")
    # market fraud proofs: a verdict is only valid if OUR OWN re-run of the
    # disputed generation agrees with it. This is the single place consensus
    # pays for a job inference, and only when someone posted a challenge.
    if not errs:
        verdicts = [t for t in block["transactions"][1:]
                    if t.get("type") == "verdict"]
        if verdicts:
            _, jobs = compute_state(chain_upto)        # cheap replay, no inference
            if jobs is None:
                errs.append("verdict in block but prior state invalid")
            else:
                for v in verdicts:
                    j = jobs.get(v.get("job_id"))
                    if j is None or j["status"] != "challenged":
                        errs.append("verdict for non-challenged job")
                        continue
                    truth = "server" if job_output(j["prompt"]) == j["output"] \
                            else "challenger"
                    if v.get("honest") != truth:
                        errs.append(f"verdict lies: recomputation says {truth} "
                                    f"was honest for job {v['job_id'][:8]}")
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
            if not verbose:
                return False   # bail on first bad block: don't burn inference
                               # verifying the rest of a hostile chain
    if compute_balances(chain) is None:
        if verbose: print("[verify] ECONOMIC RULES VIOLATED (sig/overspend/coinbase)")
        ok = False
    return ok

# ------------------------------------------------------------ market -------
def next_nonce(chain, addr):
    """Next unused nonce for addr across chain + mempool."""
    used = [t["nonce"] for t in load_mempool() if t.get("from") == addr]
    chain_used = [t.get("nonce", -1) for b in chain[1:]
                  for t in b["transactions"][1:] if t.get("from") == addr]
    return max(used + chain_used + [-1]) + 1

def market_txs_for_miner(chain):
    """Verdict txs the next block should carry: one per challenged job.
    Costs the miner one generation per open dispute -- the only time the
    market makes a miner run inference."""
    _, jobs = compute_state(chain)
    if jobs is None:
        return []
    out = []
    for jid, j in jobs.items():
        if j["status"] == "challenged":
            truth = "server" if job_output(j["prompt"]) == j["output"] \
                    else "challenger"
            out.append({"type": "verdict", "job_id": jid, "honest": truth})
    return out

def resolve_fork(local, remote):
    """Adopt remote iff it is fully valid and has strictly more work.
    Blocks identical to already-validated local ones are not re-verified
    (no inference re-runs for the common prefix), so a slow node keeping
    up with a fast chain only pays for the delta."""
    if remote[0] != local[0]:
        return local, "rejected: different genesis"
    if chain_work(remote) <= chain_work(local):
        return local, "kept local: remote has <= work"
    # checkpoint gate BEFORE any inference: a chain that rewrites history
    # below a pin is hostile by definition, however much work it claims
    for h, hx in CHECKPOINTS.items():
        if h < len(remote) and remote[h]["block_hash"] != hx:
            return local, f"rejected: violates checkpoint at height {h}"
        if h >= len(remote):
            return local, f"rejected: shorter than checkpoint height {h}"
    common = 0
    for i in range(1, min(len(local), len(remote))):
        if remote[i] != local[i]:
            break
        common = i
    for j in range(common + 1, len(remote)):
        if verify_block(remote[j], remote[j - 1], remote[:j]):
            return local, "rejected: remote chain invalid"
    if compute_balances(remote) is None:   # full replay, cheap (no inference)
        return local, "rejected: remote chain invalid"
    return remote, (f"adopted remote: more work and fully valid "
                    f"(verified {len(remote) - 1 - common} new blocks)")

# ------------------------------------------------------------ p2p ----------
def load_peers():
    if os.path.exists(PEERS_FILE):
        with open(PEERS_FILE) as f:
            return json.load(f)
    return []

def add_peers(urls):
    """Capped peer table (Bitcoin caps its addrman for the same reason):
    a hostile peer exchanging a giant or junk address list must not be able
    to flood us out of our known-good peers or bloat the table."""
    peers = load_peers()
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip().rstrip("/")
        if (u.startswith("http://") or u.startswith("https://")) \
                and len(u) <= 256 and u not in peers and len(peers) < MAX_PEERS:
            peers.append(u)
    with open(PEERS_FILE, "w") as f:
        json.dump(peers, f, indent=1)
    return peers

def load_bans():
    if os.path.exists(BANS_FILE):
        with open(BANS_FILE) as f:
            bans = json.load(f)
        now = time.time()
        return {ip: t for ip, t in bans.items() if t > now}
    return {}

def is_banned(ip):
    return ip in load_bans()

def ban_ip(ip, reason=""):
    """An invalid chain costs us real inference to discover; make its source
    pay for it with a timeout. Bitcoin bans misbehaving peers the same way."""
    bans = load_bans()
    bans[ip] = time.time() + BAN_SECONDS
    with open(BANS_FILE, "w") as f:
        json.dump(bans, f, indent=1)
    print(f"[ban] {ip} for {BAN_SECONDS}s ({reason})")

def inbox_put(remote_chain, source="?"):
    """Keep at most one pending remote chain: the highest-claimed-work one.
    The source IP rides along so the verifier can ban whoever fed us junk."""
    if os.path.exists(INBOX_FILE):
        with open(INBOX_FILE) as f:
            pending = json.load(f)["chain"]
        if chain_work(remote_chain) <= chain_work(pending):
            return False
    tmp = INBOX_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"source": source, "chain": remote_chain}, f)
    os.replace(tmp, INBOX_FILE)   # atomic: mining process reads concurrently
    return True

def inbox_take():
    """Returns (source_ip, chain) or None."""
    if not os.path.exists(INBOX_FILE):
        return None
    with open(INBOX_FILE) as f:
        pending = json.load(f)
    os.remove(INBOX_FILE)
    return pending["source"], pending["chain"]

def serve(port=9401):
    """Serve this node's chain over HTTP.
    GET  /chain  -> full chain JSON        GET /status -> height/work/tip
    GET  /peers  -> known peer URLs
    POST /chain  -> submit a higher-work chain; queued to the inbox and
                    verified by the mining loop (never inline, so a hostile
                    submission can't hijack the GPU serving thread)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    _POST_GATE = threading.Semaphore(2)   # max concurrent chain submissions
    _last_post = {}                       # ip -> time of last accepted POST
    _lp_lock = threading.Lock()
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
            elif self.path == "/peers":
                body = json.dumps(load_peers()).encode()
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            if self.path != "/chain":
                self.send_response(404); self.end_headers(); return
            ip = self.client_address[0]
            if is_banned(ip):
                self.send_response(403); self.end_headers(); return
            with _lp_lock:   # per-IP rate limit: cheap gate before any parsing
                now = time.time()
                if now - _last_post.get(ip, 0) < POST_MIN_INTERVAL:
                    self.send_response(429); self.end_headers(); return
                _last_post[ip] = now
            length = int(self.headers.get("Content-Length") or 0)
            # a legitimately-better chain is about the size of ours; refuse
            # to even parse submissions wildly larger than what we hold
            # (anti resource-exhaustion: parsing is the pre-inference cost)
            try:
                local_size = os.path.getsize(CHAIN_FILE)
            except OSError:
                local_size = 1 << 20
            post_cap = min(MAX_SYNC_BYTES, 4 * local_size + (1 << 20))
            if not 0 < length <= post_cap:
                self.send_response(413); self.end_headers(); return
            if not _POST_GATE.acquire(blocking=False):
                self.send_response(503); self.end_headers(); return
            try:
                remote = json.loads(self.rfile.read(length))
                local = load_chain()
                # cheap gates only -- no inference in the serving thread
                if remote[0] != local[0]:
                    verdict, code = "rejected: different genesis", 400
                elif chain_work(remote) <= chain_work(local):
                    verdict, code = "rejected: not more work", 200
                elif inbox_put(remote, source=ip):
                    verdict, code = "queued for verification", 202
                else:
                    verdict, code = "queued already has higher-work candidate", 200
            except Exception as e:
                verdict, code = f"malformed: {e}", 400
            finally:
                _POST_GATE.release()
            body = json.dumps({"result": verdict}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, fmt, *args):
            print(f"[serve] {self.client_address[0]} {fmt % args}")
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[serve] node listening on 0.0.0.0:{port}  "
          f"(GET /chain /status /peers, POST /chain)")
    return srv

def push_chain(peer_url, chain):
    """Announce our chain to a peer (gossip push). Best-effort."""
    from urllib.request import urlopen, Request
    body = json.dumps(chain).encode()
    req = Request(peer_url.rstrip("/") + "/chain", data=body,
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=30) as r:
        return json.load(r).get("result", "")

def sync(peer_url):
    """Fetch the peer's chain, fully verify it (one inference per block),
    adopt it iff it has more cumulative work. Returns the outcome string."""
    from urllib.request import urlopen
    peer_url = peer_url.rstrip("/")
    with urlopen(peer_url + "/status", timeout=10) as r:
        st = json.load(r)
    print(f"[sync] peer height={st['height']} tip={st['tip'][:16]}... "
          f"model={st['model']}")
    add_peers([peer_url])
    try:  # peer-list exchange (Bitcoin's addr gossip, pull flavor)
        with urlopen(peer_url + "/peers", timeout=10) as r:
            add_peers(json.load(r))
    except Exception:
        pass
    local = load_chain()
    if st["tip"] == local[-1]["block_hash"]:
        print("[sync] already in sync"); return "in-sync"
    with urlopen(peer_url + "/chain", timeout=60) as r:
        raw = r.read(MAX_SYNC_BYTES + 1)
    if len(raw) > MAX_SYNC_BYTES:
        print("[sync] rejected: peer chain exceeds size cap"); return "too-big"
    remote = json.loads(raw)
    print(f"[sync] fetched {len(remote)-1} blocks; verifying "
          f"(re-running {len(remote)-1} inferences) ...")
    kept, why = resolve_fork(local, remote)
    print(f"[sync] {why}")
    if kept is remote:
        save_chain(remote)
        print(f"[sync] local chain now height {len(remote)-1}")
    return why

def gossip_run(wallet_name, port=9401):
    """Full node: mine + gossip, with serving in a SEPARATE PROCESS.
    Same split Bitcoin arrived at (bitcoind vs. mining software): the miner
    grinds inference without ever starving the HTTP endpoints, and the
    server answers peers instantly no matter how hard mining is running.
    The two talk only through files on disk (chain file, inbox file), both
    written atomically. Mining runs in GOSSIP_CHUNK-attempt slices; between
    slices it (a) adopts any verified higher-work chain from the POST inbox,
    (b) polls peers' /status and pulls if someone is ahead, so it never
    mines long on a stale tip. Winning a block pushes to every known peer."""
    import atexit, subprocess
    w = make_wallet(wallet_name)
    srv_proc = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(__file__), "serve", str(port)])
    atexit.register(srv_proc.terminate)
    print(f"[run] server process started (pid {srv_proc.pid})")
    peers = add_peers(get_seed_nodes())
    chain = load_chain()
    if len(chain) == 1:
        for p in peers:
            try:
                sync(p); chain = load_chain()
                if len(chain) > 1: break
            except Exception as e:
                print(f"[run] seed {p} unreachable ({e})")
        if len(chain) == 1 and os.environ.get("POI_NEW_CHAIN") != "1":
            print("[run] no seed reachable; refusing to start a new isolated "
                  "chain (set POI_NEW_CHAIN=1 to do that deliberately)")
            sys.exit(1)
    last_poll = 0.0
    print(f"[run] mining to '{wallet_name}' ({w['address'][:12]}...), "
          f"gossiping with {len(load_peers())} peer(s)")
    while True:
        # 1. adopt anything a peer pushed to us (verify OUR side, one thread)
        taken = inbox_take()
        if taken is not None:
            source, pending = taken
            kept, why = resolve_fork(chain, pending)
            print(f"[run] inbox: {why}")
            if kept is pending:
                chain = pending; save_chain(chain)
            elif "invalid" in why or "checkpoint" in why:
                # this submission burned our inference (or tried to rewrite
                # pinned history) -- its source doesn't get another shot soon
                ban_ip(source, why)
        # 2. every ~20s, ask peers if someone is ahead; pull if so
        if time.time() - last_poll > 20:
            last_poll = time.time()
            from urllib.request import urlopen
            for p in list(load_peers()):
                try:
                    with urlopen(p + "/status", timeout=5) as r:
                        st = json.load(r)
                    if int(st["work"]) > chain_work(chain):
                        sync(p); chain = load_chain()
                except Exception:
                    pass
        # 3. mine a slice on the current tip
        block = mine_block(chain, w["address"],
                           transactions=load_mempool() + market_txs_for_miner(chain),
                           max_attempts=GOSSIP_CHUNK)
        if block is not None:
            chain.append(block); save_chain(chain)
            # Remove ONLY the txs we actually included -- not the whole pool.
            # A `send` can write a new tx into the mempool after mine_block
            # took its snapshot; blindly clearing would drop that tx (it was
            # never mined). Re-read now and keep anything not in this block.
            included = block["transactions"][1:]   # skip coinbase
            save_mempool([t for t in load_mempool() if t not in included])
            for p in list(load_peers()):
                try:
                    print(f"[run] push -> {p}: {push_chain(p, chain)}")
                except Exception:
                    pass

# ------------------------------------------------------------ cli ----------
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    chain = load_chain()
    if cmd == "wallet":
        w = make_wallet(sys.argv[2])
        print(f"[wallet] '{sys.argv[2]}' address: {w['address']}")
    elif cmd == "mine":
        if len(chain) == 1:
            # Fresh chain: try to join the real network before mining alone.
            print("[node] fresh chain detected — trying seed nodes ...")
            for seed in get_seed_nodes():
                try:
                    sync(seed)
                    chain = load_chain()
                    if len(chain) > 1:
                        break
                except Exception as e:
                    print(f"[node] seed {seed} unreachable ({e})")
            if len(chain) == 1 and os.environ.get("POI_NEW_CHAIN") != "1":
                print("=" * 70)
                print("WARNING: no seed node reachable. Mining now would start a")
                print("NEW ISOLATED CHAIN, not the real Glyph network. If that is")
                print("really what you want, set POI_NEW_CHAIN=1 and rerun.")
                print("Otherwise: poi_node.py sync http://<known-node>:9401 first.")
                print("=" * 70)
                sys.exit(1)
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        wname = sys.argv[3] if len(sys.argv) > 3 else "miner"
        w = make_wallet(wname)
        mp = load_mempool()
        for _ in range(n):
            block = mine_block(chain, w["address"],
                               transactions=mp + market_txs_for_miner(chain))
            chain.append(block)
            save_chain(chain)
            mp = []; save_mempool(mp)
        print(f"[node] height {len(chain)-1}, total work {chain_work(chain):,}")
    elif cmd == "send":
        # amount is given in GLY (decimals allowed), stored in units
        frm, to = sys.argv[2], sys.argv[3]
        amt = round(float(sys.argv[4]) * UNITS_PER_GLY)
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
        print(f"[tx] queued {fmt_gly(amt)} GLY from {frm} -> {to} "
              f"(balance {fmt_gly(bal.get(w['address'], 0))})")
    elif cmd == "job":
        # job WALLET "prompt text" FEE_GLY  -- pay the network for an answer
        wname, prompt = sys.argv[2], sys.argv[3]
        fee = round(float(sys.argv[4]) * UNITS_PER_GLY)
        w = make_wallet(wname)
        tx = {"type": "job", "from": w["address"], "prompt": prompt, "fee": fee,
              "nonce": next_nonce(chain, w["address"]), "pubkey": w["public"]}
        tx = sign_tx(tx, w["private"])
        mp = load_mempool(); mp.append(tx); save_mempool(mp)
        print(f"[job] queued id={job_id_of(tx)}  fee={fmt_gly(fee)} GLY  "
              f"prompt={prompt[:50]!r}")
    elif cmd == "answer":
        # answer WALLET  -- serve every open job (runs real inference), bond stake
        w = make_wallet(sys.argv[2])
        _, jobs = compute_state(chain)
        if jobs is None:
            print("[answer] chain state invalid"); sys.exit(1)
        mp = load_mempool()
        claimed = {t["job_id"] for t in mp if t.get("type") == "result"}
        served = 0
        for jid, j in jobs.items():
            if j["status"] != "open" or jid in claimed:
                continue
            out = job_output(j["prompt"])
            tx = {"type": "result", "from": w["address"], "job_id": jid,
                  "output": out, "nonce": next_nonce(chain, w["address"]),
                  "pubkey": w["public"]}
            mp.append(sign_tx(tx, w["private"])); save_mempool(mp)
            served += 1
            print(f"[answer] job {jid[:8]} fee={fmt_gly(j['fee'])} "
                  f"stake={fmt_gly(j['fee'] * JOB_STAKE_MULT)} -> {out[:60]!r}")
        print(f"[answer] served {served} job(s)")
    elif cmd == "watch":
        # watch WALLET  -- re-run served jobs in their window, challenge lies
        w = make_wallet(sys.argv[2])
        _, jobs = compute_state(chain)
        if jobs is None:
            print("[watch] chain state invalid"); sys.exit(1)
        mp = load_mempool()
        disputed = {t["job_id"] for t in mp if t.get("type") == "challenge"}
        for jid, j in jobs.items():
            if j["status"] != "served" or jid in disputed:
                continue
            if job_output(j["prompt"]) == j["output"]:
                print(f"[watch] job {jid[:8]} honest")
                continue
            tx = {"type": "challenge", "from": w["address"], "job_id": jid,
                  "nonce": next_nonce(chain, w["address"]), "pubkey": w["public"]}
            mp.append(sign_tx(tx, w["private"])); save_mempool(mp)
            print(f"[watch] job {jid[:8]} FRAUD -- challenge queued "
                  f"(bond {fmt_gly(j['fee'])}, win {fmt_gly(j['stake'] // 2)})")
    elif cmd == "jobs":
        _, jobs = compute_state(chain)
        if jobs is None:
            print("[jobs] chain state invalid"); sys.exit(1)
        for jid, j in jobs.items():
            print(f"  {jid[:12]}  {j['status']:<18} fee={fmt_gly(j['fee'])} "
                  f"prompt={j['prompt'][:40]!r}"
                  + (f" output={j['output'][:40]!r}" if "output" in j else ""))
        if not jobs:
            print("  (no jobs on chain)")
    elif cmd == "balance":
        bal = compute_balances(chain)
        if bal is None:
            print("[balance] CHAIN VIOLATES ECONOMIC RULES"); sys.exit(1)
        names = {w["address"]: n for n, w in load_wallets().items()}
        for addr, amount in sorted(bal.items(), key=lambda x: -x[1]):
            print(f"  {names.get(addr, addr[:16] + '...'):>12} : {fmt_gly(amount)} GLY")
    elif cmd == "verify":
        print(f"[node] verifying {len(chain)-1} blocks ...")
        ok = verify_chain(chain)
        print(f"[node] CHAIN {'VALID' if ok else 'INVALID'}")
        sys.exit(0 if ok else 1)
    elif cmd == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 9401).serve_forever()
    elif cmd == "sync":
        sync(sys.argv[2])
    elif cmd == "run":
        wname = sys.argv[2] if len(sys.argv) > 2 else "miner"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 9401
        gossip_run(wname, port)
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
