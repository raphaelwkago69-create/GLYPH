# Running a Glyph node

This guide is written to be followed by a person **or** pasted to an AI
assistant with terminal access ("set up a Glyph node using this document").

## What you need
- Python 3.10+ on Windows, Linux or macOS
- ~2 GB disk (GPT-2 downloads automatically on first run, ~500 MB)
- Any hardware. A GPU mines faster; a plain CPU works and can win blocks.

## Setup
```
git clone https://github.com/raphaelwkago69-create/GLYPH.git
cd GLYPH
pip install -r requirements.txt
```
(No git? Download the repo ZIP from GitHub and extract it instead.)

CPU-only machines can install the smaller torch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Join the network — one command
```
python src/poi_node.py run YOURNAME
```
This single command does everything, in order:
1. Creates a wallet named YOURNAME (keys in `poi_wallets.json` — **back this
   file up; it IS your coins; never share or publish it**).
2. Fetches the current seed nodes (from `SEEDS.txt` on GitHub).
3. Downloads the chain and re-verifies EVERY block with your own local
   inference before trusting it. This is the point of the system: you audit
   the network, you don't believe it. Expect minutes on GPU, an hour+ on
   CPU for a long chain. It only happens once — afterwards you verify just
   the new blocks as they arrive.
4. Mines continuously in gossip mode: blocks you win are pushed to peers
   (reward: 7.00 GLY each), blocks others win are pulled and verified.

Leave it running. Stopping is safe at any time — the chain saves to disk
after every block; restart with the same command to continue.

## Expected output milestones
- `[sync] adopted remote: more work and fully valid` — you are on the real
  network.
- `[mine] block N: won in ...` — you just earned 7.00 GLY.
- A warning about "NEW ISOLATED CHAIN" means no seed was reachable — do NOT
  force-mine; check your connection or the seed list, otherwise you would be
  mining a private chain worth nothing.

## Optional: verify the science first
```
python tests/hardened_cross_hardware.py
```
must print ultimate hash
`976d83a93a1d7149d0c0eeebefa30ee6cd31514b8e4f3c60468d0498ee237449` —
bit-for-bit identical on every machine ever tested. If yours differs,
please open an issue with the hash and your hardware; you may have found
the first diverging silicon.
```
python tests/poi_node_tests.py
```
runs the 21-test adversarial suite locally.

## Optional: become a seed node
If your node is reachable at a public address (e.g. a Cloudflare tunnel:
`cloudflared tunnel --url http://localhost:9401`), open a pull request
adding your URL to `SEEDS.txt`. Seeds only introduce newcomers — they can
never falsify state (everyone re-verifies), so this is safe to open up.

## Useful commands
```
python src/poi_node.py balance          # all balances, in GLY
python src/poi_node.py show             # chain summary
python src/poi_node.py verify           # full local re-verification
python src/poi_node.py send FROM TO 1.5 # queue a signed payment (GLY)
```
