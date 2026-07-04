"""
GOSSIP TEST (v0.3): two full nodes on one machine converge to one chain.

  A (port 9497) starts a fresh chain and runs serve+mine+gossip.
  B (port 9498) is pointed at A as its only seed: it must sync, verify,
    then mine in competition. B's wins reach A by push; A's wins reach B
    by status-poll pull.

PASS requires, within the time budget:
  1. both nodes report the SAME tip hash,
  2. the shared chain grew well past where A was when B joined,
  3. the shared chain contains blocks mined by BOTH wallets.

Run from an empty scratch directory (writes nodeA_*/nodeB_* files).
"""
import json, os, subprocess, sys, time
from urllib.request import urlopen

NODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "poi_node.py")
PY = sys.executable
A_PORT, B_PORT = 9497, 9498
BUDGET_S = 600

def env_for(prefix, seeds, new_chain=False):
    e = dict(os.environ)
    e["POI_PREFIX"] = prefix
    e["POI_SEEDS"] = seeds
    e["POI_CHUNK"] = "60"          # small slices -> frequent gossip on test rig
    if new_chain:
        e["POI_NEW_CHAIN"] = "1"
    return e

def status(port):
    with urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as r:
        return json.load(r)

def chain(port):
    with urlopen(f"http://127.0.0.1:{port}/chain", timeout=10) as r:
        return json.load(r)

def wait_for(cond, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            v = cond()
            if v: return v
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(what)

procs = []
try:
    print("[gossip-test] starting node A (fresh chain) ...")
    a = subprocess.Popen([PY, NODE, "run", "alice", str(A_PORT)],
                         env=env_for("nodeA_", "http://127.0.0.1:1", new_chain=True))
    procs.append(a)
    wait_for(lambda: status(A_PORT)["height"] >= 2, 240, "A never mined 2 blocks")
    a_height_at_join = status(A_PORT)["height"]
    print(f"[gossip-test] A alive at height {a_height_at_join}; starting B ...")

    b = subprocess.Popen([PY, NODE, "run", "bob", str(B_PORT)],
                         env=env_for("nodeB_", f"http://127.0.0.1:{A_PORT}"))
    procs.append(b)

    def converged():
        sa, sb = status(A_PORT), status(B_PORT)
        if sa["tip"] != sb["tip"]:
            return None
        if sa["height"] < a_height_at_join + 4:
            return None
        c = chain(A_PORT)
        miners = {blk["miner"] for blk in c[1:]}
        return c if len(miners) >= 2 else None

    c = wait_for(converged, BUDGET_S, "nodes never converged with both mining")
    miners = [blk["miner"][:8] for blk in c[1:]]
    counts = {m: miners.count(m) for m in set(miners)}
    print(f"[gossip-test] CONVERGED at height {len(c)-1}, tip shared.")
    print(f"[gossip-test] blocks per miner: {counts}")
    print("[gossip-test] PASS: one chain, two miners, gossip works")
finally:
    for p in procs:
        p.terminate()
    time.sleep(2)
    for p in procs:
        if p.poll() is None:
            p.kill()
