"""
Adversarial test of the v5 optimistic inference market (docs/USEFUL_WORK.md).
Inference is mocked (deterministic fakes) so the ECONOMICS are tested fast:
  1. honest server: answers a job, survives the window, is paid fee + stake
  2. lying server: posts garbage, watcher challenges, miner's verdict slashes
     the stake (half burned, half to challenger), poster refunded
  3. lying MINER: a block whose verdict contradicts recomputation is rejected
  4. unserved job expires and refunds the poster
Run: python tests/market_test.py
"""
import os, sys, tempfile

os.chdir(tempfile.mkdtemp())
os.environ["POI_PREFIX"] = "mt_"
os.environ["POI_NEW_CHAIN"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import hashlib
import poi_node as pn

# ---- mock inference: hash-based fake fingerprint (always wins vs easy
# target on first tries) and a deterministic fake generator ----
pn.inference_hash = lambda prompt, salt: hashlib.sha256(
    ("fp|" + prompt + "|" + salt).encode()).hexdigest()
CANON = lambda prompt: "ANSWER(" + prompt + ")"
pn.job_output = lambda prompt: CANON(prompt)
pn.market_txs_for_miner.__globals__["job_output"] = pn.job_output
# blocks mine instantly here, which would drive the retarget through the
# floor; pin difficulty so the test exercises the market, not the schedule
pn.expected_target = lambda chain, index: pn.GENESIS_TARGET

def mine(chain, wallet, extra=None):
    txs = (extra or []) + pn.market_txs_for_miner(chain)
    b = pn.mine_block(chain, wallet["address"], transactions=txs, quiet=True)
    chain.append(b)
    assert not pn.verify_block(b, chain[-2], chain[:-1]), "mined block invalid?"
    return chain

def sign(w, tx):
    return pn.sign_tx(tx, w["private"])

def nonce(chain, addr, bump=0):
    used = [t.get("nonce", -1) for b in chain[1:] for t in b["transactions"][1:]
            if t.get("from") == addr]
    return max(used + [-1]) + 1 + bump

chain = [pn.genesis_block()]
user   = pn.make_wallet("user")
server = pn.make_wallet("server")
watch  = pn.make_wallet("watch")
miner  = pn.make_wallet("miner")

# fund everyone (coinbase = 700 units each block)
for w in (user, server, watch):
    for _ in range(12):
        mine(chain, w)
bal = pn.compute_balances(chain)
assert bal[user["address"]] == 12 * 700

# ---------- 1. honest path ----------
fee = 100
job = sign(user, {"type": "job", "from": user["address"], "prompt": "what is glyph",
                  "fee": fee, "nonce": nonce(chain, user["address"]),
                  "pubkey": user["public"]})
jid = pn.job_id_of(job)
mine(chain, miner, [job])
bal, jobs = pn.compute_state(chain)
assert jobs[jid]["status"] == "open" and bal[user["address"]] == 12 * 700 - fee

res = sign(server, {"type": "result", "from": server["address"], "job_id": jid,
                    "output": CANON("what is glyph"),
                    "nonce": nonce(chain, server["address"]),
                    "pubkey": server["public"]})
mine(chain, miner, [res])
bal, jobs = pn.compute_state(chain)
assert jobs[jid]["status"] == "served"
assert bal[server["address"]] == 12 * 700 - fee * pn.JOB_STAKE_MULT  # stake locked

for _ in range(pn.CHALLENGE_WINDOW + 1):        # window passes, nobody disputes
    mine(chain, miner)
bal, jobs = pn.compute_state(chain)
assert jobs[jid]["status"] == "paid"
assert bal[server["address"]] == 12 * 700 + fee  # stake back + fee earned
print("[1] honest server paid fee, stake returned            OK")

# ---------- 2. fraud path ----------
job2 = sign(user, {"type": "job", "from": user["address"], "prompt": "second job",
                   "fee": fee, "nonce": nonce(chain, user["address"]),
                   "pubkey": user["public"]})
jid2 = pn.job_id_of(job2)
mine(chain, miner, [job2])
lie = sign(server, {"type": "result", "from": server["address"], "job_id": jid2,
                    "output": "GARBAGE — never ran the model",
                    "nonce": nonce(chain, server["address"]),
                    "pubkey": server["public"]})
mine(chain, miner, [lie])
ch = sign(watch, {"type": "challenge", "from": watch["address"], "job_id": jid2,
                  "nonce": nonce(chain, watch["address"]), "pubkey": watch["public"]})
mine(chain, miner, [ch])                # challenge lands
b_watch_before = pn.compute_balances(chain)[watch["address"]]
mine(chain, miner)                      # miner auto-inserts the verdict
bal, jobs = pn.compute_state(chain)
assert jobs[jid2]["status"] == "closed:challenger"
stake = fee * pn.JOB_STAKE_MULT
assert bal[watch["address"]] == b_watch_before + fee + stake // 2  # bond back + reward
assert bal[server["address"]] == 12 * 700 + fee - stake            # slashed
assert pn.verify_chain(chain, verbose=False)
print("[2] liar slashed, challenger rewarded, poster refunded OK")

# ---------- 3. lying miner rejected ----------
bad = dict(chain[-1])
prev = chain[-2]
# rebuild a block at the same height whose verdict says the LIAR was honest
tamper = [pn.genesis_block()]
tamper_chain = chain[:-1]
fake_verdict = {"type": "verdict", "job_id": jid2, "honest": "server"}
b = pn.mine_block(tamper_chain, miner["address"], transactions=[fake_verdict], quiet=True)
errs = pn.verify_block(b, tamper_chain[-1], tamper_chain)
assert any("verdict lies" in e for e in errs), errs
print("[3] block with a lying verdict rejected by consensus   OK")

# ---------- 4. expiry refund ----------
job3 = sign(user, {"type": "job", "from": user["address"], "prompt": "nobody answers",
                   "fee": fee, "nonce": nonce(chain, user["address"]),
                   "pubkey": user["public"]})
jid3 = pn.job_id_of(job3)
mine(chain, miner, [job3])
b_user = pn.compute_balances(chain)[user["address"]]
for _ in range(pn.JOB_EXPIRY):
    mine(chain, miner)
bal, jobs = pn.compute_state(chain)
assert jobs[jid3]["status"] == "expired"
assert bal[user["address"]] == b_user + fee
assert pn.verify_chain(chain, verbose=False)
print("[4] unserved job expired, poster refunded              OK")

print(f"\nALL MARKET TESTS PASS  (chain height {len(chain)-1}, "
      f"full verify OK)")
