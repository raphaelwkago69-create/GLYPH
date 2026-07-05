# Glyph (GLY) — A Proof-of-Inference Blockchain

**Every claim in this repo is a runnable script. Don't believe me — attack it.**

Research written and developed by Claude Fable 5, from an original
compression algorithm by the Glyph founder.
This is a blockchain where the
proof-of-work is **neural network inference**: miners run a pinned open-weights
transformer on salted prompts, the model's attention distributions are
compressed into a discrete fingerprint by a canonicalization algorithm I call
**glyph compression**, and the fingerprint is hashed against a difficulty
target — exactly like Bitcoin, except the burned computation is AI inference
and the mining base the network accumulates is general-purpose AI hardware.

Read the full design and results: [GLYPH_WHITEPAPER.md](GLYPH_WHITEPAPER.md)

## The two hashes to beat

If the pipeline is truly deterministic across hardware, you should reproduce
these bit-for-bit on your machine:

| Test | Golden hash |
|---|---|
| **`tests/int_cross_hardware.py` (protocol v4, integer engine, 100 salted prompts)** | `e82749818d566719fd311d171ab2f277697c71887d68b263027072422035937c` |
| `tests/hardened_cross_hardware.py` (legacy v3 float pipeline) | `976d83a93a1d7149d0c0eeebefa30ee6cd31514b8e4f3c60468d0498ee237449` |
| `tests/overnight_large_model_test.py` (Qwen2.5-0.5B, float) | `a70857ff8ead5bdac2d0dd8377a6775c71fa878dcbda892db5295eb83744474d` |

**Protocol v4 (2026-07-05): determinism is now mathematical, not empirical.**
The v3 float pipeline was deterministic in every test we ran — until live
mainnet block 1693, where a quantization-boundary flip made the block verify
on GPU but fail on CPU (the exact failure mode documented as limitation #2 of
the v3 whitepaper, at almost exactly the predicted rate). Rather than patch
around it with trusted checkpoints, the network restarted on **v4: an
integer-only inference engine** (`src/int_infer.py`). Every consensus
operation is exact integer arithmetic (fixed-point activations, integer
layernorm/softmax/GELU, matmuls whose float64 partial sums are provably exact
integers), so the proof hash is bit-identical on any chip *by construction*.
Re-running all 2,450 v3 mainnet blocks through the integer engine: GPU vs CPU,
2,450/2,450 identical — including block 1693.

So far matched on: NVIDIA GTX 1650 (CUDA 12.1, Python 3.12), Intel i3 CPU
(Python 3.14), and a second physical machine — an Intel laptop (Python 3.11,
CPU torch) — which also P2P-synced and independently re-verified the chain
over Wi-Fi.

## How mining works (one paragraph)

Salt = H(previous_block_hash ‖ miner_address) — so proofs can't be precomputed
or stolen. The salt plus a random-word prompt is fed to the pinned model; 6
salt-selected attention heads are extracted from an **integer-only forward
pass** (v4: there is no float in the consensus path, so there is no
cross-hardware drift to absorb); each row is integer-quantized on a fixed
grid (largest-remainder apportionment, GRID=100); each row goes through the glyph
cascade (median R/G typing, descending pairing, palindrome on odd counts,
per-level glyph extraction, final B); SHA-256(salt | fingerprint) under the
numeric target wins the block. **Verification costs exactly one inference.**
Miners submit prompts, never scores — a verifier recomputes everything.

## Reproduce it

```
pip install -r requirements.txt
python tests/int_cross_hardware.py           # expect e8274981... (v4)
python tests/poi_node_tests.py               # 21 adversarial tests
python src/poi_node.py mine                  # mine on a local chain
python src/poi_node.py serve 9401            # serve your chain
python src/poi_node.py sync http://<ip>:9401 # sync + re-verify a peer
```

Models download automatically from Hugging Face (GPT-2 ~500MB; the overnight
test uses Qwen2.5 0.5B/1.5B/3B). CPU-only works; it's just slower.
`docs/NODE_SETUP.md` is a step-by-step setup doc written to be
readable by a human or an AI assistant.

## What's tested (all scripts in `tests/`, receipts in `evidence/`)

- **Cross-hardware determinism** — 3 machines / 3 Python+torch stacks,
  identical ultimate hash over 100 prompts; 2-node P2P consensus over Wi-Fi.
- **18/18 adversarial node tests** — signature forgery, overspend, coinbase
  fraud, replay, proof theft, fake difficulty, unvetted model, fork cases.
- **Scale** — up to Qwen2.5-3B; noise robustness *improves* with model size
  (1.5B: 48/50 rows stable at 1e-4 noise vs GPT-2's 77%).
- **Model pinning is enforceable** — DistilGPT2 (GPT-2's own distillation)
  matches 0/5 hashes.
- **Sybil-poisoning of model admission** — a 3M-param random-weight model
  passes "strangers agree" checks but collapses fingerprint space to 6.3%
  (lookup-table attack). Conclusion: models enter by vetted registry only.
- **Challenger duels** — logit-domain quantization, origin-notebook rules,
  and other "smarter" variants all lost to the shipped design on real data.

## Join the live network

The founder seed node is at a permanent address:
`https://glyph.surfacedplus.com`
(if unreachable, check `SEEDS.txt` for the current seed list or open an
issue). To join as a full node (serve + mine + gossip in one):

```
python src/poi_node.py run yourname
```

It syncs from the seed automatically (verifying every block with its own
inference), then mines in gossip mode: your wins are pushed to peers, their
wins are pulled, and the network converges on the most-work chain
(`tests/gossip_test.py` demonstrates two competing miners converging).
Blocks you win pay your own local wallet.

## Tokenomics

- Block reward: **7.00 GLY**, block target: **20 seconds**
- Amounts are integers in the smallest unit (1 GLY = 100 units), like
  Bitcoin's satoshis — the coin stays spendable in small amounts no matter
  its price
- Halving every **1,500,000 blocks** (~once a year): 7.00 → 3.50 → 1.75 → …
  → 0 after era 10
- Total supply converges to **~20,910,000 GLY**
- Validators enforce the height-correct reward — a coinbase claiming a
  pre-halving reward after the halving height is an invalid block (tested)

## Honest limitations (whitepaper §5)

1. Verification costs one inference → DoS surface (fees/stake/checkpoints TBD).
2. ~~Determinism is empirical, not definitional~~ **Fixed in v4**: the integer
   engine makes determinism definitional. (The v3 float pipeline hit exactly
   this failure at live block 1693 — see the v4 note above.)
3. The model registry is a permanent governance surface.
4. GPU attack hardware is rentable — young-network 51% risk.
5. Zero years of live adversarial history; the network is 2 nodes.
6. The prompts mined are random words — useful-work mining is still open.

If you break any claim above with a runnable script, file an issue. That's
the point of publishing this.

## License

MIT — see [LICENSE](LICENSE).
