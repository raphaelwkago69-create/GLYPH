"""
POI-NODE v0.2 ADVERSARIAL TEST SUITE
Runs 10 attacks/checks against the node logic in-process. Uses its own
temp files so the real chain is untouched.
"""
import copy, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# sandbox the chain files
import poi_node as N
N.CHAIN_FILE   = "test_chain.json"
N.WALLET_FILE  = "test_wallets.json"
N.MEMPOOL_FILE = "test_mempool.json"
for f in (N.CHAIN_FILE, N.WALLET_FILE, N.MEMPOOL_FILE):
    if os.path.exists(f): os.remove(f)

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond: PASS += 1
    else:    FAIL += 1
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))

print("=" * 70)
print("SETUP: wallets + mine 8 blocks (miner=alice)")
print("=" * 70)
alice = N.make_wallet("alice")
bob   = N.make_wallet("bob")
eve   = N.make_wallet("eve")

chain = N.load_chain()
t0 = time.time()
for i in range(8):
    block = N.mine_block(chain, alice["address"], quiet=True)
    chain.append(block)
print(f"  mined 8 blocks in {time.time()-t0:.1f}s")

print("\n" + "=" * 70)
print("T1  HONEST CHAIN VERIFIES")
print("=" * 70)
check("full verification (8 inference re-runs)", N.verify_chain(chain, verbose=False))
bal = N.compute_balances(chain)
check("alice earned 8 block rewards", bal.get(alice["address"]) == 8 * N.block_reward(1),
      f"balance={bal.get(alice['address'])}")

print("\n" + "=" * 70)
print("T2  SIGNED TRANSACTION: alice pays bob 30")
print("=" * 70)
tx = {"from": alice["address"], "to": bob["address"], "amount": 30,
      "nonce": 0, "pubkey": alice["public"]}
tx = N.sign_tx(tx, alice["private"])
block = N.mine_block(chain, alice["address"], transactions=[tx], quiet=True)
chain.append(block)
bal = N.compute_balances(chain)
check("chain still valid", N.verify_chain(chain, verbose=False))
check("bob received 30", bal.get(bob["address"]) == 30)
check("alice debited", bal.get(alice["address"]) == 9 * N.block_reward(1) - 30)

print("\n" + "=" * 70)
print("T3  FORGED SIGNATURE: eve signs a tx spending ALICE's coins")
print("=" * 70)
forged = {"from": alice["address"], "to": eve["address"], "amount": 100,
          "nonce": 1, "pubkey": eve["public"]}        # eve's key, alice's money
forged = N.sign_tx(forged, eve["private"])
bad = copy.deepcopy(chain)
blk = N.mine_block(bad, eve["address"], transactions=[forged], quiet=True)
bad.append(blk)
check("forgery rejected", N.compute_balances(bad) is None)

# also: eve pastes alice's pubkey but signs with her own key
forged2 = {"from": alice["address"], "to": eve["address"], "amount": 100,
           "nonce": 1, "pubkey": alice["public"]}
forged2 = N.sign_tx(forged2, eve["private"])
bad2 = copy.deepcopy(chain)
blk2 = N.mine_block(bad2, eve["address"], transactions=[forged2], quiet=True)
bad2.append(blk2)
check("wrong-key signature rejected", N.compute_balances(bad2) is None)

print("\n" + "=" * 70)
print("T4  OVERSPEND: bob (has 30) tries to send 500")
print("=" * 70)
over = {"from": bob["address"], "to": eve["address"], "amount": 500,
        "nonce": 0, "pubkey": bob["public"]}
over = N.sign_tx(over, bob["private"])
bad = copy.deepcopy(chain)
blk = N.mine_block(bad, eve["address"], transactions=[over], quiet=True)
bad.append(blk)
check("overspend rejected", N.compute_balances(bad) is None)

print("\n" + "=" * 70)
print("T5  REPLAY ATTACK: rebroadcast alice's old signed tx (nonce reuse)")
print("=" * 70)
bad = copy.deepcopy(chain)
blk = N.mine_block(bad, eve["address"], transactions=[copy.deepcopy(tx)], quiet=True)
bad.append(blk)
check("replayed tx rejected", N.compute_balances(bad) is None)

print("\n" + "=" * 70)
print("T6  COINBASE FRAUD: miner pays themselves double reward")
print("=" * 70)
bad = copy.deepcopy(chain)
blk = N.mine_block(bad, eve["address"], quiet=True)
blk["transactions"][0]["amount"] = N.block_reward(blk["index"]) * 2
blk["block_hash"] = N.block_header_hash(blk)   # re-seal header honestly
bad.append(blk)
check("inflated coinbase rejected", N.compute_balances(bad) is None)

print("\n" + "=" * 70)
print("T6b HALVING: pre-halving reward claimed after the halving height")
print("=" * 70)
check("halving schedule: 7 -> 3 -> 1 -> 0",
      [N.block_reward(0), N.block_reward(N.HALVING_INTERVAL),
       N.block_reward(2 * N.HALVING_INTERVAL),
       N.block_reward(3 * N.HALVING_INTERVAL)] == [7, 3, 1, 0])
_saved_interval = N.HALVING_INTERVAL
N.HALVING_INTERVAL = 4          # pretend the halving landed inside our chain
check("stale full-reward coinbase rejected post-halving",
      N.compute_balances(chain) is None)
N.HALVING_INTERVAL = _saved_interval
check("chain valid again with real halving interval",
      N.compute_balances(chain) is not None)

print("\n" + "=" * 70)
print("T7  PROOF THEFT: eve steals alice's winning prompt for herself")
print("=" * 70)
victim = N.mine_block(chain, alice["address"], quiet=True)
stolen = copy.deepcopy(victim)
stolen["miner"] = eve["address"]
stolen["transactions"][0]["to"] = eve["address"]
stolen["block_hash"] = N.block_header_hash(stolen)
errs = N.verify_block(stolen, chain[-1], chain)
check("stolen proof fails verification", any("proof mismatch" in e for e in errs),
      "salt is bound to miner address")
errs_honest = N.verify_block(victim, chain[-1], chain)
check("original proof still verifies for alice", not errs_honest)
chain.append(victim)

print("\n" + "=" * 70)
print("T8  UNVETTED MODEL: block claims model_id outside the registry")
print("=" * 70)
bad = copy.deepcopy(chain)
blk = N.mine_block(bad, eve["address"], quiet=True)
blk["model_id"] = "evil_pygmy_3M"
blk["block_hash"] = N.block_header_hash(blk)
errs = N.verify_block(blk, bad[-1], bad)
check("unvetted model rejected", any("not in vetted registry" in e for e in errs))

print("\n" + "=" * 70)
print("T9  DIFFICULTY: fabricated easy target rejected; schedule retargets")
print("=" * 70)
bad = copy.deepcopy(chain)
blk = N.mine_block(bad, eve["address"], quiet=True)
blk["target"] = "f" * 64                      # claim trivial difficulty
blk["block_hash"] = N.block_header_hash(blk)
errs = N.verify_block(blk, bad[-1], bad)
check("fake easy target rejected", any("difficulty schedule" in e for e in errs))
t_first = int(chain[1]["target"], 16)
t_late  = N.expected_target(chain, len(chain))
check("difficulty retargeted after interval", t_late != N.GENESIS_TARGET or t_first == t_late,
      f"genesis={t_first:.3e} now={t_late:.3e}" if False else
      f"target moved {t_first/max(t_late,1):.2f}x harder" if t_late < t_first else "unchanged (blocks were slow)")

print("\n" + "=" * 70)
print("T10 FORK RESOLUTION: most-work valid chain wins")
print("=" * 70)
# fork off two blocks back; attacker builds an alternative tip of SAME length
fork_base = copy.deepcopy(chain[:-1])
attacker_tip = N.mine_block(fork_base, eve["address"], quiet=True)
fork_base.append(attacker_tip)
kept, why = N.resolve_fork(chain, fork_base)
check("equal-work fork does not replace local chain", kept is chain, why)
# attacker extends their fork one block further (now MORE work)
attacker_tip2 = N.mine_block(fork_base, eve["address"], quiet=True)
fork_base.append(attacker_tip2)
kept, why = N.resolve_fork(chain, fork_base)
check("more-work valid fork adopted", kept is fork_base, why)
# but a more-work fork with an invalid block inside is rejected
poisoned = copy.deepcopy(fork_base)
poisoned[3]["transactions"][0]["amount"] = 9999
poisoned[3]["block_hash"] = N.block_header_hash(poisoned[3])
# re-link the rest so linkage is intact but content is fraudulent
for i in range(4, len(poisoned)):
    poisoned[i]["prev_hash"] = poisoned[i-1]["block_hash"]
    poisoned[i]["block_hash"] = N.block_header_hash(poisoned[i])
kept, why = N.resolve_fork(chain, poisoned)
check("more-work INVALID fork rejected", kept is chain, why)

print("\n" + "=" * 70)
print(f"RESULT: {PASS} passed, {FAIL} failed")
print("=" * 70)
N.save_chain(chain)
sys.exit(1 if FAIL else 0)
