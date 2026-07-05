# Glyph: A Proof-of-Inference Blockchain

**Draft v0.2 — July 2026 · protocol v3 · the network described here is live**
Research written and developed by Claude Fable 5, from an original
compression algorithm by the Glyph founder.

---

## Abstract

We present a proof-of-work blockchain in which the work is neural network
inference. Miners run a pinned, open-weights transformer model on salted
random prompts; the model's internal attention distributions are compressed
by a novel canonicalization algorithm ("glyph compression") into a discrete
fingerprint, which is hashed. A block is won when the hash meets the
difficulty target, exactly as in Bitcoin. Verification requires a single
inference run. We show empirically that the scheme is deterministic across
heterogeneous hardware and software stacks, resistant to lookup-table,
distillation, and score-forgery attacks, and that model admission must be
governed by a vetted whitelist rather than network observation. All results
in this paper are reproducible with the published test scripts.

This is not only a proposal: a public mainnet running the protocol is live.
Its first day included two physical machines on different internet
connections (one on cellular data) converging on one chain via gossip, the
first block mined by non-founder hardware — a consumer laptop CPU — and the
first peer-to-peer transaction. Instructions for joining the network with
one command are in the repository README.

---

## 1. Motivation

Bitcoin's security rests on provably burned computation, but the computation
itself is arithmetically meaningless, and the hardware it fosters (SHA-256
ASICs) is useless for anything else. We ask: can the burned computation be
AI inference, so that the network's accumulated mining base is
general-purpose AI hardware, while retaining Bitcoin's verification
asymmetry and objectivity?

The central obstacle is that inference produces floating-point outputs that
differ subtly across hardware, threatening consensus. This paper's core
contribution is an empirically validated pipeline that makes transformer
attention deterministic enough to hash.

## 2. The Mechanism

### 2.1 Attention fingerprints

For a pinned model M (weights hash fixed by protocol), input text produces,
at each attention head, a probability distribution over tokens (softmax
output, summing to 1). These distributions are a rich, input-sensitive,
model-specific signal that can only be obtained by actually running M.

### 2.2 Integer quantization (determinism layer)

Each attention row is mapped to integers on a fixed grid (GRID = 100) using
largest-remainder (Hamilton) apportionment, so the integers sum exactly to
the grid. All subsequent computation is integer-only. Hardware float drift
below the grid resolution is absorbed. (Measured drift between NVIDIA CUDA
and Intel CPU begins at the 7th decimal place; the grid absorbs it with a
wide margin — see §4.1.)

### 2.3 Glyph compression (canonicalization layer)

The quantized row is reduced by an iterated pairwise-merge cascade. Let
v₁ … vₙ be the integers produced by §2.2, and let m be their median. Each
value is assigned a **type**:

- **R** if vᵢ > m (values carrying above-median mass),
- **G** otherwise.

(The letters originate in the algorithm's first formulation, which was
drawn with red and green tokens; we retain them.) The row is now a typed
multiset, and the cascade proceeds in levels. At each level:

1. **Order.** Values are sorted descending (ties broken by position, so the
   order is total and deterministic).
2. **Pair.** An even-length sequence is partitioned into adjacent disjoint
   pairs. An odd-length sequence is first rearranged into a palindrome-like
   interleaving and paired with a sliding window (a+b, b+c, …), so that no
   element is orphaned; interior elements participate in two pairs. Odd
   levels therefore produce n−1 outputs and restore even parity.
3. **Merge.** Each pair (a, b) yields a + b, typed by the algebra
   R·R → R, G·G → G, R·G → **X** (a *glyph*), X·X → R, X·regular → R.
   A glyph is thus a **boundary artifact**: it records that above-median
   and below-median mass collided at this level, and how much.
4. **Extract.** Glyphs leave the stream, recorded as (level, value) pairs;
   the remaining typed values continue to the next level. The cascade
   terminates when one value B remains.

The head fingerprint is the pair (B, glyph chain). Because glyphs carry the
extracted boundary mass, B + Σ(glyph values) recovers the row total — the
cascade relocates information rather than discarding it, but the *placement*
of mass across the original row is destroyed. Empirically the glyph form
survives noise better than hashing the quantized integers directly (§4.4):
small perturbations move mass within a type class far more often than they
move the R/G boundary itself. Its role in the blockchain is
canonicalization, not security.

Glyph compression is intentionally one-way: the fingerprint is exact for
comparison but cannot be inverted to recover the attention distribution,
and a fortiori cannot be searched backwards for prompts that would produce
a target fingerprint. (The algorithm predates this blockchain: it was
originally designed as an irreversible-compression privacy primitive, where
the compressed form is safe to transmit precisely because reconstruction is
impossible without locally held state.) Mining therefore cannot be inverted;
it can only be performed forward, one inference at a time. Alternative
cascade rules from the original formulation (subtractive glyph collisions;
border-sum elimination) were tested against the present rules and found
equivalent, not superior; the present rules stand (origin_rules_test.py).

### 2.4 Salting and head selection

Each block's salt is derived from the previous block hash and the miner's
address: salt = H(prev_hash ‖ miner_addr). The salt is prepended to the
prompt before inference, and its hash seeds a deterministic RNG that selects
k = 6 of the model's attention heads (of 144 in GPT-2) to form the block
fingerprint. Consequences:

- Precomputed lookup tables die every block (§4.2).
- A distillation proxy cannot know which heads to imitate (§4.3).
- A winning prompt is bound to its miner: relayed proofs cannot be stolen,
  because a different miner address yields a different salt and hash.

### 2.5 Mining and verification

Mining: generate a random prompt, run M on salt+prompt, extract the
salt-selected heads, quantize, glyph-compress, hash with SHA-256; win if
hash < target. Verification: one inference run reproducing the same hash.
The submitted object is the **prompt**, never scores — the verifier
re-derives everything, so forged scores cannot enter the system.

### 2.6 Difficulty, rewards, forks

Difficulty is a numeric target (hash < target) recomputed deterministically
from chain timestamps every 5 blocks toward a target block time, clamped to
4× per adjustment (Bitcoin-style). All amounts are integers in the smallest
unit (1 GLY = 100 units, satoshi-style, preserving small payments at any
coin price). Blocks carry a coinbase reward of 7.00 GLY, halving every
1,500,000 blocks (roughly yearly at the 20-second block target):
7.00 → 3.50 → 1.75 → … → 0 after the tenth era, so total supply converges
to ~20,910,000 GLY — validators enforce the height-correct reward, so a
stale reward claim is an invalid block. Blocks also carry ECDSA-signed
transactions with per-sender nonces (replay protection). Fork choice is
most-cumulative-work among fully valid chains.

### 2.7 Model governance

Models are admitted only via a vetted registry changed by versioned protocol
upgrade. §4.5 demonstrates why runtime admission ("strangers cross-confirm
a new model") is Sybil-poisonable: determinism is free, and a random-weight
3M-parameter model passes cross-confirmation while collapsing the
fingerprint space to 6.3% (resurrecting lookup-table attacks) and mining ~6×
faster than the honest model. Admission audits must check: (a) provenance
(known public release), (b) fingerprint-space size, (c) honest cost
benchmarks. Multiple vetted models can coexist as difficulty tiers on one
chain; splitting models across separate chains fragments security and is
rejected.

Because the fingerprint requires access to internal attention states, only
open-weights models can participate. The network is structurally restricted
to open AI.

### 2.8 Network protocol

Nodes run a combined serve/mine/gossip loop. Mining proceeds in bounded
attempt slices; between slices the node (a) adopts any pending higher-work
chain submitted by peers, and (b) polls known peers' status, pulling and
verifying any chain claiming more cumulative work. Winning a block pushes
the full chain to all known peers, so NAT-bound nodes (which cannot be
dialed) still propagate their wins over their outbound connections. Peer
lists are exchanged on every sync (Bitcoin's addr gossip, pull flavor).

Verification asymmetry is protected structurally: submitted chains pass
cheap gates first (size cap, genesis match, strictly-more claimed work) and
are queued to an inbox; expensive inference re-verification happens only in
the single mining thread, never in the network-facing thread, so a hostile
submission cannot commandeer the node's GPU. Re-verification is incremental:
blocks byte-identical to already-validated local history are not re-run —
a node re-verifies only the delta beyond the common prefix, making slow CPU
nodes viable followers of a fast chain (they pay the deep history once).

Bootstrap uses a repository-hosted seed list fetched at join time, so seed
addresses can be updated for all future nodes without shipping code. A seed
can only introduce peers, never falsify state: newcomers re-verify every
block by local inference regardless of who served it. A node with no
reachable seed refuses to mine a fresh chain unless explicitly forced,
preventing accidental isolated networks.

## 3. Threat Model and Defenses

| Attack | Defense | Result (tested) |
|---|---|---|
| Forged scores | Prompt-submission; verifier re-runs | 0 matches / 130k+ attempts |
| Lookup table | Per-block salt | old winners → 5.7% ≈ chance (6.25%) |
| Distillation proxy | Salt-selected heads | 0/400 exact matches |
| Proof theft | Miner-bound salt | stolen proof rejected; original verifies |
| Signature forgery / overspend / replay / coinbase inflation | Standard validation | all rejected (18/18 suite) |
| Fake difficulty | Deterministic target schedule | rejected |
| Unvetted model | Registry check | rejected |
| Invalid more-work fork | Full re-verification before adoption | rejected |
| Sybil model admission | Vetted registry (no runtime admission) | poison demonstrated, §4.5 |

## 4. Experimental Results

All scripts and raw outputs available alongside this document. Model: GPT-2
small (124M) unless stated. Hardware: NVIDIA GTX 1650 (CUDA), Intel i3
(CPU), and an independent Intel laptop (CPU, integrated graphics).

### 4.1 Cross-hardware determinism
100 prompts, salt-selected heads per prompt. Identical ultimate hash
`976d83a9…ee237449` across: (a) GTX 1650, Python 3.12, torch 2.5.1+cu121;
(b) same machine CPU-only, Python 3.14, torch 2.12.1+cpu; (c) an unrelated
Intel laptop, Python 3.11, CPU torch. Three processors, three Python
versions, three torch builds — bit-for-bit agreement (600 head-fingerprints,
zero divergence).

### 4.2 Lookup-table invalidation
Winning prompts under salt A retained winner status under salt B at 5.7%,
statistically indistinguishable from the 6.25% chance rate.

### 4.3 Distillation
An MLP proxy trained on 2,500 prompts against fixed, known heads achieved
some exact fingerprint matches; the same proxy against a different salt's
head selection: 0/400. Additionally, DistilGPT2 — a model distilled from
GPT-2 itself — reproduced 0/5 of GPT-2's fingerprint hashes.

### 4.4 Canonicalization ablations
Glyph fingerprints survive injected noise better than direct hashes of the
quantized integers (112/150 vs 98/150 at 1e-4). A logit-domain quantization
variant was tested and rejected (0/120 at 1e-4 vs 92/120 for the
probability-domain grid): log-scaling amplifies perturbations of small
probabilities. Note 1e-4 injected noise is ~10³ larger than measured
hardware drift; at measured drift levels survival is 100% (§4.1).

### 4.5 Admission poisoning
A randomly initialized 1-layer, 3M-parameter transformer passes
determinism-based cross-confirmation 20/20 while exhibiting a collapsed
fingerprint space (19 unique fingerprints over 300 prompts, 6.3%) and ~6×
mining speed. A legitimate new model (Qwen2.5-0.5B, 24 layers, different
tokenizer) ran through the unmodified pipeline with 100% fingerprint
uniqueness — the mechanism is model-agnostic; admission is the part that
cannot be automated.

### 4.6 Scale
The pipeline was run unmodified on Qwen2.5-0.5B, 1.5B and 3B (24/28/36
layers; up to 25× GPT-2's parameters). Fingerprint uniqueness was 100% at
every scale. GPU-vs-CPU determinism at 0.5B (24 layers, double GPT-2's
depth of accumulated float error) was bit-for-bit. Noise survival at 1e-4
*improved* with scale (GPT-2: 77%; 1.5B: 96%; 3B: 85%), consistent with
larger models producing sharper attention distributions that sit further
from quantization boundaries. Separately, boundary-flip frequency was
measured directly: 4 per 100,000 random rows at 1e-7 (measured-drift-scale)
perturbation, i.e. ~0.02% of 6-head proofs; per protocol an ambiguous proof
is simply invalid and costs only its miner.

### 4.7 Live network
A public mainnet running protocol v3 is live (genesis
`86c5e8db29b3…`). Its predecessor chain, run for one full day as a dress
rehearsal, demonstrated every network property end to end:

- **Cross-ISP consensus.** Two physical machines — an NVIDIA GTX 1650
  desktop and a consumer Intel laptop on cellular data — gossiped through a
  public tunnel and converged on one chain for hours, unattended.
- **Non-founder mining.** The laptop's CPU, at roughly 1/20th the desktop's
  inference rate, won multiple blocks; each was pushed to the network,
  independently re-verified by the recipient, and adopted.
- **First transaction.** A signed transfer (founder → second wallet,
  21 GLY) was mined into a block and validated by the receiving node.
- **Onboarding.** A fresh clone joined with one command: it fetched the
  seed list, synced the full chain, re-verified every block by local
  inference ("adopted remote: more work and fully valid"), and mined.
- **Adversarial suite.** 21 tests covering §3 (including halving
  enforcement) pass in full, and a two-miner convergence test
  (gossip_test.py) shows competing miners resolving to a single chain
  containing blocks from both.

## 5. Honest Limitations and Open Problems

1. **Determinism is empirical, not proven — RESOLVED in protocol v4.**
   This limitation was real and it fired: live v3 mainnet block 1693 landed
   on a quantization boundary and verified on GPU while failing on CPU
   (1 block in 2,423 — almost exactly the ~4-in-100k boundary-flip rate this
   section predicted). Because blocks are chained, one divergent block made
   the whole suffix unverifiable on CPU. The fix was the endgame this
   section named: protocol v4 replaces the float forward pass with an
   **integer-only engine** (`src/int_infer.py`) — fixed-point int64
   activations, integer layernorm/softmax/GELU built from hardcoded integer
   constants (no libm), and matrix multiplies whose float64 partial sums
   are provably exact integers (bounded < 2^53, so IEEE-754 makes every
   operation exact in any order on any chip). Determinism is now
   definitional: re-running all 2,450 v3 blocks GPU-vs-CPU under v4 gives
   2,450/2,450 identical hashes, including block 1693. The integer weight
   conversion is pinned by hash in the model registry.
2. **Verification-cost DoS.** Verifying costs one inference (seconds on
   CPU), vastly more than Bitcoin's microsecond hash check. Spam of invalid
   proofs is a real surface; mitigations (verification fees, peer scoring,
   proof-of-stake gating for submission) are future work.
3. **Distillation economics are informal.** The salt-selected-heads defense
   is empirically strong against small proxies; a formal argument that
   full-coverage distillation costs ≥ honest mining is future work.
4. **The work is not yet useful.** Prompts are random; the burned compute
   proves inference occurred but produces no useful output. Binding mining
   to real tasks without enabling precomputation is an open research
   problem (for the entire field).
5. **Small network.** Model scale has been tested to 3B parameters across
   three architectures (§4.6) with results improving at scale, and the live
   network has demonstrated multi-node, multi-ISP consensus (§4.7) — but at
   a scale of two cooperating machines. Adversarial live networks, thousands
   of peers, and years of uptime are untested.
6. **No fee market yet.** Transactions are currently free and miner income
   is coinbase-only, which ends after the final halving era. A fee market is
   required before then, and doubles as the natural anti-spam mechanism for
   limitation 2. It is the first planned protocol upgrade.

## 6. Conclusion

Proof-of-inference with glyph-canonicalized attention fingerprints achieves
Bitcoin's structure — objective, asymmetric, precomputation-resistant work —
while directing the burned computation through general-purpose AI hardware
and restricting participation to open-weights models. The network is live;
joining it is one command. Every security claim above is backed by a
runnable script rather than an argument from authority. We invite attack.

---

*Reproduction package: hardened_cross_hardware.py (§4.1), hardened_poi.py
(§4.2–4.4), cross_model_test.py (§4.3), logit_softmax_test.py (§4.4),
poison_test.py (§4.5), overnight_large_model_test.py (§4.6),
poi_node.py + poi_node_tests.py + gossip_test.py (§2.8, §3, §4.7).
Live seed list: SEEDS.txt in the repository.*
