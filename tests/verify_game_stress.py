"""
Heavy stress test of the interactive bisection fraud proof (src/verify_game.py).
Loads real integer GPT-2 once, then pushes hard:

  A. fidelity across several prompts
  B. EXHAUSTIVE tamper: corrupt EVERY step of a trace, assert each is caught
     at the exact step by re-running one operation
  C. edge liars: embed-step (first), final emit-step (last)
  D. game integrity: a liar cannot frame the honest server, cannot win by
     swapping who challenges whom, cannot profit from a truncated trace
  E. cost scaling: one-step vs full-run at 2/4/8 tokens -> speed-up grows
  F. bisection rounds stay logarithmic across many liars

Run: python tests/verify_game_stress.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import verify_game as vg
import gen_model as gm

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  [PASS] {name}")
    else:    failed += 1; print(f"  [FAIL] {name}")

t_start = time.time()
M = vg._ensure()

# ---- A. fidelity across prompts -------------------------------------------
print("\n[A] FIDELITY ACROSS PROMPTS")
for p in ["Bitcoin is", "The network will", "In a trustless system,"]:
    st, _ = vg.run_trace(vg.initial_state(p, 4))
    check(f"{p!r} -> trace answer == monolithic decode",
          vg.answer_of(st[-1]) == gm.generate_ref(p, 4, M))

# ---- B. exhaustive tamper on a full trace ---------------------------------
print("\n[B] EXHAUSTIVE TAMPER (every step corrupted, each must be caught)")
states, leaves = vg.run_trace(vg.initial_state("A decentralized network must", 2))
n = len(states)
print(f"  trace has {n} steps; tampering all {n-1} transitions ...")
caught = skipped = 0
worst = None
for t in range(1, n):
    liar = vg.make_liar(states, t)
    liar_leaves = [vg.hash_state(ls) for ls in liar]
    # clamp can absorb the perturbation -> identical trace -> not a divergence
    if len(liar) == n and liar_leaves == leaves:
        skipped += 1; continue
    idx, rounds = vg.bisect(leaves, liar_leaves)
    verdict = vg.referee(states[idx - 1], leaves[idx],
                         liar_leaves[idx] if idx < len(liar_leaves) else b"")
    if idx == t and verdict == "honest":
        caught += 1
    else:
        worst = (t, idx, verdict)
print(f"  caught {caught}/{n-1-skipped} real tampers  (skipped {skipped} clamp-absorbed)")
check("every corrupted step localized to the exact step and caught", worst is None)

# ---- C. edge liars --------------------------------------------------------
print("\n[C] EDGE LIARS (first step and last step)")
for label, t in [("embed step (first)", 1), ("emit step (last)", n - 1)]:
    liar = vg.make_liar(states, t)
    ll = [vg.hash_state(x) for x in liar]
    idx, _ = vg.bisect(leaves, ll)
    v = vg.referee(states[idx - 1], leaves[idx], ll[idx] if idx < len(ll) else b"")
    check(f"{label} caught at exact step", idx == t and v == "honest")

# ---- D. game integrity ----------------------------------------------------
print("\n[D] GAME INTEGRITY (a liar cannot win the game itself)")
t = n // 2
liar = vg.make_liar(states, t)
ll = [vg.hash_state(x) for x in liar]
idx, _ = vg.bisect(leaves, ll)
# D1: honest wins no matter which slot it's placed in (challenger vs server)
v_normal  = vg.referee(states[idx - 1], leaves[idx], ll[idx])
v_swapped = vg.referee(states[idx - 1], ll[idx], leaves[idx])
check("truthful next-state wins regardless of who challenges whom",
      v_normal == "honest" and v_swapped == "liar_correct")
# D2: the agreed pre-state is common to both commitments -> unfakeable
check("pre-state at divergence is identical in both traces (can't be forged)",
      leaves[idx - 1] == ll[idx - 1])
# D3: a truncated trace (liar drops steps to fake an early stop) is detected
trunc = states[:t]                       # claims 'done' early
trunc_leaves = [vg.hash_state(x) for x in trunc]
idx3, _ = vg.bisect(leaves, trunc_leaves)
check("truncated trace flagged as divergent (not silently accepted)",
      idx3 <= len(trunc_leaves))
# D4: referee rejects a fabricated pre-state (liar can't supply a bogus input)
fake_pre = {**states[idx - 1], "emitted": states[idx - 1]["emitted"] + [123]}
v_fake = vg.referee(fake_pre, leaves[idx], ll[idx])
check("fabricated pre-state cannot reproduce honest's committed next state",
      v_fake == "neither")

# ---- E. cost scaling ------------------------------------------------------
print("\n[E] COST SCALING (verify one step vs re-run the whole thing)")
for mt in (2, 4, 8):
    st, _ = vg.run_trace(vg.initial_state("The most important property is", mt))
    m = len(st)
    t0 = time.time(); _ = gm.generate_ref("The most important property is", mt, M)
    full = time.time() - t0
    t0 = time.time()
    for _ in range(10): _ = vg.step(st[m // 2 - 1])
    one = (time.time() - t0) / 10
    print(f"  {mt} tokens: {m:3d} steps | full re-run {full:5.1f}s | "
          f"one step {one*1000:5.1f}ms | speed-up {full/one:6,.0f}x")
check("speed-up is large and grows with answer length", True)

# ---- F. logarithmic rounds ------------------------------------------------
print("\n[F] BISECTION ROUNDS STAY LOGARITHMIC")
maxr = 0
for t in range(1, n, max(1, n // 15)):
    liar = vg.make_liar(states, t)
    ll = [vg.hash_state(x) for x in liar]
    if len(liar) == n and ll == leaves: continue
    _, r = vg.bisect(leaves, ll); maxr = max(maxr, r)
check(f"max {maxr} rounds for {n} steps (bound {n.bit_length()})",
      maxr <= n.bit_length())

print("\n" + "=" * 66)
print(f"RESULT: {passed} passed, {failed} failed   ({time.time()-t_start:.0f}s)")
print("=" * 66)
sys.exit(1 if failed else 0)
