# Cheap verification of full-fledged models: interactive bisection fraud proofs

Status: IMPLEMENTED and tested (src/verify_game.py, src/gen_model.py;
tests/verify_game_test.py, tests/verify_game_stress.py). Demonstrated on real
GPT-2 small / large / xl (124M -> 1.5B).

## The problem

Verifying an inference answer by RE-RUNNING the model costs a full inference.
So a big model is expensive to verify, which is the wall behind the
"useful work" and "who watches the servers" open problems: an honest node
cannot cheaply police a mythos-class model, because checking = running it again.

## The idea (same design that secures Ethereum L2s: Arbitrum / Truebit)

Do not re-run the whole model to check it. Find the single step where a liar
lied, and check only that.

1. **Commit.** The server runs the model as a sequence of small deterministic
   STEPS (embed / one attention sub-block / one MLP sub-block / one emit) and
   publishes one Merkle root over the whole execution trace. O(1) to publish,
   reveals nothing.
2. **Optimistic accept.** No challenge in the window -> consensus re-runs ZERO
   inference. The answer is accepted on the commitment alone.
3. **Bisect.** A challenger who disputes the answer plays a binary search over
   the trace: each round reveals one committed leaf per side until the FIRST
   divergent step is isolated. ceil(log2(steps)) rounds (~10 for a 741-step
   answer).
4. **Referee.** Consensus re-executes exactly ONE step from the agreed
   pre-state and sees whose next-state is correct. One bounded operation, not
   the model. The liar is objectively slashed.

Cost to the verifier: ~log2(steps) hash checks + ONE step, regardless of model
size. The model stays full-fledged; the check stays tiny.

## Why GLYPH can do this and most AI cannot

The referee's one-step re-execution must have a single undeniable answer. Float
inference differs in the last bit across machines, so "who is right?" is
ambiguous and the whole scheme collapses. The v4 integer engine is bit-exact BY
CONSTRUCTION, so one step re-executed anywhere yields the identical integer
state. The determinism built for the lottery is exactly what makes big models
cheaply verifiable here. (Test [4] checks this property directly.)

## Measured (real GPT-2, CPU)

| model        | layers | trace steps (10 tok) | one-step | verify speed-up |
|--------------|--------|----------------------|----------|-----------------|
| gpt2 (124M)  | 12     | 209 (8 tok)          | ~0.11s   | 41x             |
| gpt2-large   | 36     | 741                  | ~0.16s   | 154x            |

The speed-up GROWS with model depth (one step is ~1/(2*n_layer) of a token's
work) and with answer length, while the referee's cost stays ~constant. On a
100+ layer model the advantage is hundreds-to-thousands x. gpt2-xl (1.5B, 48
layers) is the ceiling of the current exact engine without re-porting a new
architecture; every GPT-2 variant keeps head_dim=64 so the proven integer
bounds hold unchanged (src/gen_model.py).

## Tested invariants

- fidelity: the committed step-trace reproduces an independent monolithic decode
- honest path: two honest runs commit identically; nothing is re-run
- fraud: EVERY corrupted step (exhaustive, 52/52 on a 53-step trace) is
  localized to the exact step and caught by one re-execution
- game integrity: a liar cannot frame the honest server, cannot win by swapping
  who challenges whom, cannot profit from a truncated trace, cannot fake the
  pre-state
- rounds stay logarithmic

## Honest limits / trust model

- Optimistic, not zero-trust: needs >=1 honest challenger in the window (the
  same assumption as optimistic rollups). Verification is now cheap for any
  model size; liveness of a watcher is the remaining assumption.
- Interactive: a dispute takes ceil(log2(steps)) rounds; both parties must be
  live in the challenge window.
- The server stores the full trace to answer challenges (it only reveals one
  slice). Data-availability cost.
- zkML remains the endgame that removes interactivity and the watcher
  assumption; the same integer/quantized execution is also what makes a future
  zk proof cheap. This design is the buildable-today step, not the final one.

## CLI

```
python src/verify_game.py "your prompt" [model] [max_new_tokens]
# model in: gpt2 (default), gpt2-medium, gpt2-large, gpt2-xl
```
Prints the answer, its commitment, the honest-path (0 steps), and a simulated
liar caught by one re-executed step, with cost numbers.
