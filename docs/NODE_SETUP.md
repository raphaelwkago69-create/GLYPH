# Instructions for the Intel laptop (hand this to Gemini)

ATTENTION AI ASSISTANT ON THIS MACHINE: the user is testing a proof-of-inference
blockchain across two physical computers. This machine is the VERIFIER node.
The mining machine (NVIDIA desktop) is on the same Wi-Fi network.

## Files to copy onto this laptop (from the desktop's Downloads folder)
- `poi_node.py`
- `hardened_cross_hardware.py`

## Setup
1. Ensure Python 3.10+ is installed.
2. `pip install torch transformers ecdsa` (CPU torch is fine and smaller:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu` then
   `pip install transformers ecdsa`)
3. First run downloads GPT-2 (~500MB) from HuggingFace automatically.

## TEST 1 — cross-hardware determinism (do this first)
```
python hardened_cross_hardware.py
```
Wait for the ULTIMATE HASH line. It MUST equal:
```
976d83a93a1d7149d0c0eeebefa30ee6cd31514b8e4f3c60468d0498ee237449
```
- MATCH: cross-machine determinism proven on this Intel silicon. Report success.
- NO MATCH: report the hash you got, and diff `hardened_xhw_hashes.json`
  against the desktop's copy to find which prompt indices diverged.

## TEST 2 — P2P sync (blockchain consensus between the two machines)
On the DESKTOP, someone must first run:
```
python poi_node.py serve 9401
```
and find the desktop's LAN IP with `ipconfig` (IPv4 Address, e.g. 192.168.x.x).
Windows Firewall may prompt to allow Python on private networks — allow it.

Then on THIS laptop:
```
python poi_node.py sync http://<DESKTOP_IP>:9401
python poi_node.py verify
```
Expected: it fetches the chain, re-runs one GPT-2 inference per block on this
laptop's CPU, prints `adopted remote: more work and fully valid`, and
`verify` prints CHAIN VALID.

That result means: this Intel laptop independently re-derived, from raw model
inference, every proof the NVIDIA machine mined — cross-hardware blockchain
consensus. Report the output of both commands.

## If anything fails
Report the exact error text. Do not modify the scripts — determinism testing
requires both machines run byte-identical code.
