# Protocol v4: the block-1693 incident and the integer fix

Receipts for the v3→v4 protocol reset (2026-07-05).

## What happened

Live v3 mainnet block 1693 (prompt `they blue this will copper right bird`)
re-computed to a different proof hash on CPU than on the GPU that mined it:

- GPU: `0010713ed807712706ff...` — meets target, valid
- CPU: `8a81809096769c3d838b...` — fails target, invalid

Confirmed on two independent CPU stacks. A full CPU re-verification of all
2,423 blocks then live found exactly ONE divergent block: 1693. That is a
0.041% rate — almost exactly the ~4-in-100,000 quantization-boundary-flip
rate predicted in whitepaper §5, limitation 1. Because blocks are chained,
a CPU node correctly rejected 1693 and therefore could never sync past it:
the "any hardware can verify" claim was false in production.

## The fix (v4)

`src/int_infer.py` — GPT-2 in exact integer arithmetic. Fixed-point int64
activations; integer layernorm/softmax/GELU from hardcoded integer constants
(no libm anywhere in consensus); matmuls in float64 where every partial sum
is a bounded exact integer (< 2^53), which IEEE-754 guarantees is exact in
any accumulation order on any chip. Integer weight conversion pinned by
SHA-256 in the model registry:
`842a00bc8f09c1e6eb870e750deaa49159dd45fb9ab860fc8f40bef6878029ac`

## Verification runs (all reproducible)

1. Block 1693 under v4: GPU, GPU-venv-with-CUDA-disabled, and a separate
   Python 3.14 CPU-torch stack all produce
   `bcad52c8fca7cd0dae662b7a7525c2f997d861d02c9734cfb779b7aae83bdf82`.
2. All 2,450 v3 mainnet (prompt, salt) pairs re-run under v4, GPU vs CPU:
   **2,450 / 2,450 identical, zero divergences.**
3. 100-prompt receipt (`tests/int_cross_hardware.py`), GPU and CPU:
   `e82749818d566719fd311d171ab2f277697c71887d68b263027072422035937c`
4. Integer attention vs float attention (sanity that it is still real
   inference): max abs difference 0.005 in probability units, worst
   correlation 0.998 across 36 head/prompt pairs.
5. Full 21-test adversarial consensus suite: 21/21 pass on the v4 engine.
6. Speed: GTX 1650 ~8.4 inf/s, i3 CPU ~4.5 inf/s — the integer engine
   narrows the GPU/CPU gap versus the float pipeline, which is good for
   verification decentralization.

The v3 chain (2,450 blocks) is archived by the founder node; v4 restarted
from a fresh genesis. All v3 balances were wiped — the founder's included.
Fair launch, again, on an engine where the flip that killed v3 is
impossible by construction.

Research written and developed by Claude Fable 5.
