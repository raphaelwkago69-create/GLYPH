"""
Protocol v4 cross-hardware determinism receipt.

Runs 100 salted prompts through the integer-only engine (src/int_infer.py)
and prints one ultimate hash. Because every consensus operation is exact
integer arithmetic, this hash is identical on every device BY CONSTRUCTION
-- if your machine prints anything else, your model weights differ from the
vetted ones (the engine also pins their conversion hash).

Expected ultimate hash:
  e82749818d566719fd311d171ab2f277697c71887d68b263027072422035937c

Research written and developed by Claude Fable 5.
"""
import hashlib, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import int_infer as I

PROMPTS = [f"determinism receipt prompt {i:03d} glyph" for i in range(100)]

h = hashlib.sha256()
for i, p in enumerate(PROMPTS):
    salt = "v4_receipt_" + hashlib.sha256(p.encode()).hexdigest()[:12]
    h.update(I.inference_hash(p, salt).encode())
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/100 ...")
print("integer weights hash:", I.WEIGHTS_HASH)
print("ULTIMATE HASH:", h.hexdigest())
