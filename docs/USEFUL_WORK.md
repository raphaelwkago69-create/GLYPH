# Useful Work: an optimistic inference market on top of the Glyph lottery

Status: IMPLEMENTED as protocol v5 (src/poi_node.py + int_infer.generate).
Economics tested in tests/market_test.py (mocked inference). CONSENSUS
CHANGE: v4 nodes reject v5 blocks — testnet until the network adopts it.

CLI: `job WALLET "prompt" FEE` · `answer WALLET` (serve jobs, bond stake) ·
`watch WALLET` (auto-challenge fraud) · `jobs` (market state). Miners insert
verdicts automatically; consensus checks them by re-running the generation.

This addresses the whitepaper's open problem
"the burned compute proves inference occurred but produces no useful output."

## Why the naive fix is impossible

You cannot simply mine on user prompts. The lottery's security depends on
prompts being unpredictable (salted, random): the miner must run inference
*forward* on each fresh attempt. A real user prompt is known in advance, so
answers can be precomputed, cached, and shared — the work stops being burned
per-block and the difficulty target stops measuring anything. Binding mining
to real tasks without enabling precomputation is open for the entire field.

## Design: separate the two jobs

Keep the existing inference lottery untouched as the *security* layer (it
orders blocks and resists rewriting). Add a second, parallel layer — an
**optimistic inference market** — where usefulness lives:

1. **Request.** A user posts a `job` transaction: prompt (or its hash for
   privacy), model id from the registry, and a fee in GLY, escrowed on-chain.
2. **Serve.** Any node may claim the job by posting a `result` transaction:
   the model's actual output plus a bonded **stake** (e.g. 10x the fee).
3. **Optimistic acceptance.** The result is *not* verified by consensus.
   After a challenge window of W blocks with no dispute, the escrowed fee
   pays the server and the stake unlocks.
4. **Challenge.** During the window, anyone may post a `challenge` (with its
   own smaller bond). Only then do miners re-run the single disputed
   inference — deterministically, thanks to the v4 integer engine — as a
   consensus rule for including the `verdict` in a block. Liar loses:
   a wrong server forfeits its stake (half burned, half to the challenger);
   a wrong challenger forfeits its bond to the server.

## Why this fits Glyph unusually well

- The v4 integer-only engine makes the fraud proof **exact**: one bit of
  output difference is an objective, machine-checkable lie. Optimistic
  schemes on float inference need fuzzy tolerance; Glyph doesn't.
- Verification cost stays out of the hot path: consensus re-runs an
  inference only for *disputed* jobs, expected to be rare because lying is
  -EV against a 10x stake and a public, deterministic re-check.
- The security budget still comes from the lottery. The market can fail
  (nobody serves a job, fee refunds after expiry) without touching consensus.

## Trust model, stated honestly

This is not zero-trust like the lottery. It assumes **at least one honest
watcher** re-runs served jobs looking for challenge profit. That is the same
assumption as optimistic rollups, and it is the practical frontier today;
the fully trustless alternative (zkML — a succinct proof that the model ran
correctly, checkable in milliseconds) is not yet cheap enough for GPT-2-class
work but slots into the same job/result transaction shape later: replace the
challenge window with a proof field, keep everything else.

## New transaction types (sketch)

```
{type:"job",      id, prompt_hash, model_id, fee, expiry, from, sig}
{type:"result",   job_id, output, output_hash, stake, from, sig}
{type:"challenge",job_id, bond, from, sig}
{type:"verdict",  job_id, honest:"server"|"challenger"}   # miner-inserted,
                                                          # consensus-checked
```

Consensus additions: escrow/stake accounting in `compute_balances`; a
`verdict` is valid in a block only if the miner's recomputed inference
matches (or refutes) `output_hash` — this is the one place the market
touches `verify_block`, and only for disputed jobs.

## Parameters to fight about later

Challenge window W (long enough for a CPU watcher to re-run the job),
stake multiple (10x fee?), challenger bond, output size cap (data
availability: outputs live in blocks, so cap tokens per job).

## Build order

1. Job/result/escrow txs, no challenges (trusted beta on testnet).
2. Challenge + verdict consensus rule (the actual fraud proof).
3. Watcher mode: `poi_node.py watch` — re-run recent results, auto-challenge.
4. Fee market ties in here naturally (jobs bid for block space).
