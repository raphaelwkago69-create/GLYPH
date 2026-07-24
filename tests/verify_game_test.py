"""
Adversarial test of the interactive bisection fraud proof (src/verify_game.py).
Loads the REAL integer GPT-2 once, then checks the invariants that make big
models cheaply verifiable:

  1. fidelity     : the committed step-trace's answer == int_infer.generate()
  2. honest path  : two honest traces agree -> no dispute, 0 steps re-run
  3. fraud caught : for many tamper positions (incl. edges), bisection localizes
                    the FIRST divergent step and the referee catches the liar by
                    re-executing exactly ONE step
  4. determinism  : re-running one step reproduces the committed next state bit
                    for bit (the property float models lack)

Run: python tests/verify_game_test.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import verify_game as vg
import gen_model as gm

PROMPT = "A decentralized network must"
MAXTOK = 6

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; print(f"  [FAIL] {name}")

print("building honest committed trace with real integer GPT-2 ...")
s0 = vg.initial_state(PROMPT, MAXTOK)
states, leaves = vg.run_trace(s0)
n = len(states)
print(f"  trace: {n} steps, answer = {vg.answer_of(states[-1])!r}")

# 1. fidelity ---------------------------------------------------------------
print("\n[1] TRACE FIDELITY (committed trace reproduces the real answer)")
real = gm.generate_ref(PROMPT, MAXTOK, vg._ensure())
check("trace answer equals monolithic decode", vg.answer_of(states[-1]) == real)

# 2. honest vs honest -------------------------------------------------------
print("\n[2] HONEST PATH (no dispute, nothing re-run)")
states2, leaves2 = vg.run_trace(vg.initial_state(PROMPT, MAXTOK))
check("independent honest re-run gives identical commitment", leaves == leaves2)
check("merkle roots match", vg.merkle_root(leaves) == vg.merkle_root(leaves2))

# 4. one-step determinism ---------------------------------------------------
print("\n[4] STEP DETERMINISM (referee's one-step re-exec is bit-exact)")
det_ok = all(vg.hash_state(vg.step(states[i])) == leaves[i + 1]
             for i in range(0, n - 1, max(1, (n - 1) // 20)))
check("re-executing any sampled step reproduces the committed next state", det_ok)

# 3. fraud caught at many tamper positions ----------------------------------
print("\n[3] FRAUD PROOF (liar caught by re-running ONE step)")
# sample spread of positions plus explicit edges (first, second, last, an emit)
emit_steps = [i for i in range(1, n) if states[i - 1]["phase"] == "emit"
              or (states[i]["phase"] == "emit")]
sample = sorted(set(
    [1, 2, n - 1, n // 2]
    + list(range(3, n - 1, max(1, (n - 1) // 12)))
    + emit_steps[:3]))
sample = [t for t in sample if 1 <= t <= n - 1]

all_caught = True
max_rounds = 0
for t in sample:
    liar = vg.make_liar(states, t)
    liar_leaves = [vg.hash_state(ls) for ls in liar]
    if liar_leaves == leaves[:len(liar_leaves)] and len(liar) == n:
        # perturbation collided with the honest state (clamp) -> not a real
        # divergence; skip this position rather than assert on a non-fraud
        continue
    idx, rounds = vg.bisect(leaves, liar_leaves)
    max_rounds = max(max_rounds, rounds)
    verdict = vg.referee(states[idx - 1], leaves[idx],
                         liar_leaves[idx] if idx < len(liar_leaves) else b"")
    ok = (idx == t) and (verdict == "honest")
    if not ok:
        all_caught = False
        print(f"    [!] tamper@{t}: localized {idx}, verdict {verdict}")
check(f"every tampered trace caught at the exact step ({len(sample)} positions)",
      all_caught)
check(f"bisection cost is logarithmic (max {max_rounds} rounds for {n} steps)",
      max_rounds <= (n).bit_length())

print("\n" + "=" * 66)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 66)
sys.exit(1 if failed else 0)
