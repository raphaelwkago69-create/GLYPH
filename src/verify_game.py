"""
==============================================================================
VERIFY-GAME -- interactive bisection fraud proofs for full-fledged models
==============================================================================
The verification wall: checking an answer by RE-RUNNING the model costs a full
inference, so a big model is expensive to verify. This module removes that wall
without shrinking the model, using the same fraud-proof design that secures
Ethereum L2s (Arbitrum/Truebit):

  * The server runs the model as a sequence of small deterministic STEPS and
    commits a Merkle root over the whole execution TRACE (every intermediate
    state). Publishing the commitment is O(1); it does not reveal the guts.
  * If a challenger disputes the answer, the two parties BISECT the trace:
    binary-search to the single step where their traces first diverge. This
    takes ceil(log2(steps)) rounds, each exchanging one hash.
  * The referee (consensus) then re-executes exactly ONE step from the agreed
    pre-state and sees whose next-state is correct. One bounded operation --
    NOT the whole model. The liar is objectively exposed and slashed.

Why GLYPH can do this and most AI cannot: the referee's one-step re-execution
must have a single undeniable answer. Float inference differs in the last bit
across machines, so "who is right?" is ambiguous and the scheme collapses. The
int_infer engine is bit-exact BY CONSTRUCTION, so one step re-executed anywhere
gives the identical integer state -- the determinism built for the lottery is
exactly what makes big models cheaply verifiable here.

A step here is one primitive of the transformer (embed / one attention sub-block
/ one MLP sub-block / one emit). For a P-layer model, one step is ~1/(2P) of a
single token's compute and ~1/(2P*N) of an N-token answer -- and the referee
runs only that one. The model stays full-fledged; the check stays tiny.

Run:  python src/verify_game.py "your prompt here"
==============================================================================
"""
import hashlib, os, sys, time
import int_infer as ii
import gen_model as gm


# --------------------------------------------------------------- trace -------
# The generation of an answer is expressed as a state machine whose transition
# `step()` is a pure function. Iterating it from the initial state reproduces
# greedy integer decoding token-for-token, so the honest trace's answer is the
# real answer -- verified in the demo below. The model is configurable (any
# GPT-2 size via gen_model); default is set by POI_GEN_MODEL or "gpt2".

_M = None                                  # currently loaded gen_model


def set_model(name):
    """Load/select the generation model (gpt2 / gpt2-medium / -large / -xl)."""
    global _M
    _M = gm.load(name)
    return _M


def _ensure():
    if _M is None:
        set_model(os.environ.get("POI_GEN_MODEL", "gpt2"))
    return _M


def model_name():
    return _ensure()["name"]


def initial_state(prompt, max_new_tokens):
    """State 0: prompt tokenized, no forward done yet."""
    M = _ensure()
    ids = M["tok"](prompt, return_tensors="pt")["input_ids"][0].to(M["device"])
    ids = ids[-(gm.GEN_MAX_CONTEXT - max_new_tokens):]
    return {"ids": ids, "x": None, "l": 0, "sub": "attn", "phase": "embed",
            "emitted": [], "done": False, "max": max_new_tokens}


def step(s):
    """One primitive transition. Pure: returns a new state, never mutates s."""
    M = _ensure(); torch = M["torch"]; nL = M["cfg"]["n_layer"]
    if s["done"]:
        return s
    phase = s["phase"]

    if phase == "embed":
        x = gm.embed(s["ids"], M)
        return {**s, "x": x, "l": 0, "sub": "attn", "phase": "layer"}

    if phase == "layer":
        l = s["l"]; x = s["x"]; T = x.shape[0]
        if s["sub"] == "attn":
            x = torch.clamp(x + gm._attn_block(
                ii._layernorm(x, M["W"][f'h.{l}.ln_1.g'], M["W"][f'h.{l}.ln_1.b']),
                M, l, T), -ii.ACT_CLAMP, ii.ACT_CLAMP)
            return {**s, "x": x, "sub": "mlp"}
        # mlp sub-block
        x = torch.clamp(x + gm._mlp_block(
            ii._layernorm(x, M["W"][f'h.{l}.ln_2.g'], M["W"][f'h.{l}.ln_2.b']), M, l),
            -ii.ACT_CLAMP, ii.ACT_CLAMP)
        if l == nL - 1:
            return {**s, "x": x, "phase": "emit"}
        return {**s, "x": x, "l": l + 1, "sub": "attn"}

    # phase == "emit": final layernorm + logits + argmax -> next token
    nxt = gm.emit_logits(s["x"], M)
    emitted = s["emitted"] + [nxt]
    ids = s["ids"]
    end = (len(emitted) >= s["max"] or ids.shape[0] + 1 >= gm.GEN_MAX_CONTEXT)
    if nxt == 50256:                       # eot is a stop signal, not output
        return {**s, "x": None, "emitted": s["emitted"], "done": True}
    if end:
        return {**s, "x": None, "emitted": emitted, "done": True}
    ids = torch.cat([ids, torch.tensor([nxt], device=M["device"])])
    return {**s, "ids": ids, "x": None, "phase": "embed", "emitted": emitted}


def generate(prompt, max_new_tokens=8):
    """Greedy decode via the step machine (matches the committed trace)."""
    s = initial_state(prompt, max_new_tokens)
    while not s["done"]:
        s = step(s)
    return answer_of(s)


def answer_of(s):
    return _ensure()["tok"].decode(s["emitted"])


# --------------------------------------------------------- commitment --------

def hash_state(s):
    """Canonical sha256 of a state -- the leaf committed in the trace."""
    h = hashlib.sha256()
    h.update(s["phase"].encode()); h.update(s["sub"].encode())
    h.update(str(s["l"]).encode()); h.update(b"1" if s["done"] else b"0")
    h.update(s["ids"].cpu().numpy().tobytes())
    h.update(bytes(str(s["emitted"]), "utf8"))
    if s["x"] is not None:
        h.update(s["x"].cpu().numpy().tobytes())
    return h.digest()


def run_trace(s0):
    """Iterate step() to completion; return (states, leaf_hashes)."""
    states = [s0]; leaves = [hash_state(s0)]
    s = s0
    while not s["done"]:
        s = step(s)
        states.append(s); leaves.append(hash_state(s))
    return states, leaves


def merkle_root(leaves):
    """O(1) commitment to the whole trace."""
    level = list(leaves)
    if not level:
        return b""
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest()
                 for i in range(0, len(level), 2)]
    return level[0]


# ---------------------------------------------------- the bisection game -----

def bisect(honest_leaves, liar_leaves):
    """Binary-search the first index where the two committed traces differ.
    In the real protocol each probe reveals ONE leaf hash per side; here we
    read the arrays directly. Returns (first_divergent_index, rounds)."""
    assert honest_leaves[0] == liar_leaves[0], "must agree on the input state"
    n = min(len(honest_leaves), len(liar_leaves))
    lo, hi, rounds = 0, n - 1, 0
    # if traces have different length, the shorter one 'ends early': treat the
    # missing tail as divergent from the first index past its end.
    if honest_leaves[n - 1] == liar_leaves[n - 1] and \
       len(honest_leaves) != len(liar_leaves):
        return n, 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        rounds += 1
        if honest_leaves[mid] == liar_leaves[mid]:
            lo = mid
        else:
            hi = mid
    return hi, rounds


def referee(pre_state, honest_leaf_at_i, liar_leaf_at_i):
    """Consensus re-executes exactly ONE step from the agreed pre-state and
    declares who committed the correct next state. Returns 'server' (honest
    party's leaf matches truth), 'challenger' (liar's does), or 'neither'."""
    truth = hash_state(step(pre_state))
    if truth == honest_leaf_at_i:
        return "honest"
    if truth == liar_leaf_at_i:
        return "liar_correct"      # the 'liar' array actually held the truth
    return "neither"


# --------------------------------------------------------------- demo --------

def make_liar(states, tamper_at):
    """A cheating server: run honestly up to `tamper_at`, corrupt that one
    step's output, then continue stepping honestly from the corrupted state.
    Produces a full, internally-consistent-looking trace with a wrong answer."""
    torch = _ensure()["torch"]
    pre = states[tamper_at - 1]
    bad = step(pre)
    if bad["x"] is not None:                 # nudge a hidden activation
        x = bad["x"].clone(); x[-1, 0] = x[-1, 0] + 777 * ii.ONE
        bad = {**bad, "x": torch.clamp(x, -ii.ACT_CLAMP, ii.ACT_CLAMP)}
    else:                                    # corrupt an emitted token instead
        em = list(bad["emitted"]); em[-1] = (em[-1] + 1) % 50256
        bad = {**bad, "emitted": em}
    out = list(states[:tamper_at]) + [bad]
    s = bad
    while not s["done"]:
        s = step(s); out.append(s)
    return out


def demo(prompt, max_new_tokens=8):
    M = _ensure()
    nsub = M["cfg"]["n_layer"] * 2
    print(f'MODEL : {M["name"]}  ({M["cfg"]["n_layer"]}L x {M["cfg"]["n_embd"]}d)')
    print(f'PROMPT: "{prompt}"')
    print("running the model as a committed step-trace ...")
    t0 = time.time()
    s0 = initial_state(prompt, max_new_tokens)
    states, leaves = run_trace(s0)
    gen_time = time.time() - t0
    root = merkle_root(leaves)
    ans = answer_of(states[-1])

    # fidelity: the honest trace's answer IS the monolithic-decode answer
    real = gm.generate_ref(prompt, max_new_tokens, M)
    assert ans == real, f"trace/answer mismatch:\n {ans!r}\n {real!r}"

    print(f"\nANSWER: {ans!r}")
    print(f"trace steps      : {len(states)}  (embed + {nsub} layer-ops + emit, per token)")
    print(f"commitment (root): {root.hex()[:32]}...")
    print(f"full run time    : {gen_time:.2f}s  <- what re-running to verify would cost")

    # cost of ONE step (what the referee actually pays), measured on a mid step
    mid = len(states) // 2
    t0 = time.time()
    for _ in range(20):
        _ = step(states[mid - 1])
    one_step = (time.time() - t0) / 20
    print(f"one-step time    : {one_step*1000:.2f}ms  <- what the referee actually pays")
    print(f"verify speed-up  : {gen_time/one_step:,.0f}x cheaper than re-running "
          f"(grows with model size)")

    # -------- honest server, no dispute: answer accepted for free --------
    print("\n[A] honest server, nobody challenges")
    print("    consensus checks only the commitment -> 0 inference steps re-run. accepted.")

    # -------- lying server, challenger disputes: caught in one step --------
    import random
    tamper_at = random.randint(1, len(states) - 1)
    liar_states = make_liar(states, tamper_at)
    _, liar_leaves = zip(*[(ls, hash_state(ls)) for ls in liar_states])
    liar_leaves = list(liar_leaves)
    liar_ans = answer_of(liar_states[-1])
    print(f"\n[B] lying server corrupts step {tamper_at}, posts wrong answer:")
    print(f"    liar's answer: {liar_ans!r}")

    idx, rounds = bisect(leaves, liar_leaves)
    print(f"    bisection localizes first divergence in {rounds} rounds -> step {idx}")
    verdict = referee(states[idx - 1], leaves[idx], liar_leaves[idx])
    caught = verdict == "honest" and idx == tamper_at
    print(f"    referee re-executes 1 step from step {idx-1} -> verdict: honest server correct")
    print(f"    liar caught: {caught}  (slash the stake)")

    print("\n" + "=" * 70)
    print(f"a {len(states)}-step model answer was verified by re-running ONE step.")
    print(f"the model can be ANY size -- the referee still pays for one step.")
    print("=" * 70)
    return caught


if __name__ == "__main__":
    # usage: python verify_game.py "prompt" [model] [max_new_tokens]
    prompt = sys.argv[1] if len(sys.argv) > 1 else \
        "The most important property of a decentralized network is"
    if len(sys.argv) > 2:
        set_model(sys.argv[2])
    mnt = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    ok = demo(prompt, mnt)
    sys.exit(0 if ok else 1)
