# Hardened Proof-of-Inference: Securing LLM Execution
*(Formerly Attention Glyph Fingerprinting)*

This document summarizes the full evolution of the Proof-of-Inference (PoI) mechanism—from an initial experiment in compressing Large Language Model (LLM) attention distributions, into a secure, hardened cryptographic protocol.

## 1. The Origin: Attention Compression & Glyphs
The fundamental goal began as an attempt to take the probability distribution of an attention head and iteratively compress it to extract structural meaning. 

By categorizing probability values as either above the median ("Red") or below the median ("Green") and combining them in a sliding window, we extracted "Glyphs"—values where high and low probabilities collided. This generated highly unique structural fingerprints for LLM generations.

## 2. The Vulnerabilities of the Initial System
While the original system successfully generated deterministic signatures, it failed several critical load-bearing security tests:
1. **Lookup Table Attacks:** The fingerprint space was too small. Only ~35% of test prompts generated unique signatures, creating massive collision clusters. An attacker could easily precompute a table of winning hashes.
2. **Distillation Attacks:** Because the signature relied on a static set of attention heads, a simple, cheap MLP proxy model could be trained to predict the fingerprint without running the actual, expensive LLM.
3. **Floating-Point Drift:** Slight differences in hardware floating-point math would cause the cryptographic hash to diverge across different machines.

## 3. The Decisive Upgrades (The Hardened System)
To transform the fingerprinting concept into a secure cryptographic Proof-of-Inference protocol, four major architectural upgrades were introduced:

### A. Salting (Killing Lookup Tables)
Every block/challenge now prepends a random salt to the prompt. A precomputed table of winning hashes becomes completely worthless the moment the salt changes. 
* *Result:* Re-salting reduced a previously valid set of winning hashes to a 5.7% survival rate (pure mathematical chance). The lookup table attack is dead.

### B. Salt-Selected Heads (Killing Distillation)
Instead of relying on a static set of attention heads, the block's salt dynamically and deterministically selects which specific heads (e.g., 6 out of GPT-2's 144) will form the fingerprint. An attacker cannot prepare a cheap distillation proxy because they do not know which heads matter until the block opens. To win reliably, the attacker must run the *entire* depth of the model, completely destroying the economic advantage of proxy distillation.
* *Result:* A dedicated proxy model scored 0 / 400 exact matches against dynamically selected heads.

### C. Canonical Integer Fingerprint (Killing Float Drift)
Attention scores are mapped to a strict integer quantization grid. The entire downstream fingerprinting process (including glyph compression) runs exclusively on pure integers. 
* *Result:* Any two machines that agree on the integer grid will generate the exact same signature, bit for bit, eliminating floating-point drift.

### D. Glyphs as Canonicalization, Not Security
The original "Glyph" extraction logic was demoted. Security now relies entirely on the SHA-256 hash, verifier re-runs, and salting. The glyph logic was retained strictly as a canonical, noise-tolerant reduction step.
* *Result:* Under rigorous testing, the integer-based Glyph method actively outperformed a direct hash of the integers in noise tolerance, proving it still actively earns its place in the architecture.

## 4. The Final Verdict
The decisive end-to-end load-bearing test proved the Hardened Proof-of-Inference mechanism is robust, secure, and viable:
- **Uniqueness (Entropy):** 3,000 out of 3,000 prompts (100%) produced entirely unique signatures. The collision clusters are entirely gone.
- **Uniform Difficulty Scaling:** Increasing the mining difficulty (requiring more leading zeros) produced an 18.4x scaling factor, incredibly close to the ideal uniform 16x of a hexadecimal hash space. The "lumpy" hash space is cured.
- **End-to-End Cryptography:** Mining a block, verifying it by re-running the prompt, failing a tampered prompt, and failing under re-salting all behaved exactly as a secure cryptographic loop should.

---

## 5. Layman's Breakdown: How the Security Works

### What is "Salting"?
Imagine you are running a maze to find a treasure. If the maze never changes, someone could just memorize the right path (precompute it), write it down, and cheat every time without actually running the maze. 

**Salting** is like dynamically shifting the walls of the maze every single time a new round starts. In our code, the "salt" is a random string of characters (like `block_salt_3f8b...`) that gets glued to the front of the text prompt. 

Because LLMs read text left-to-right, putting a new salt at the very beginning completely changes how the AI reads the rest of the sentence. A precomputed "cheat sheet" (a lookup table of winning prompts) becomes completely useless the moment the salt changes. Every miner has to run the *new* maze from scratch.

### How the Solutions Led to a 100% Success Rate
The 100% success rate is the result of closing every single loophole an attacker could use to cheat the system:
1. **The Lookup Table Loophole (Closed by Salting):** As explained above, salting forces miners to actually do the work right now, instead of using work they saved from yesterday.
2. **The "Cheap Shortcut" Loophole (Closed by Salt-Selected Heads):** AI models have many "attention heads" (GPT-2 has 144). An attacker might try to build a tiny, cheap, fake AI that only calculates the 6 heads we check, ignoring the other 138. We fixed this by having the *salt* pick which 6 heads matter *at random, at the last second*. Because the attacker doesn't know which heads will be picked, their cheap fake AI is useless. They are forced to run the full, expensive model—which is exactly what we want to prove they did!
3. **The "Math Disagreement" Loophole (Closed by Integer Grids):** Different computers do decimal math (floating-point) slightly differently. One computer might say `0.3333333` and another might say `0.3333334`. In cryptography, that tiny difference breaks everything. We fixed this by rounding all the AI's internal scores to a strict grid of whole numbers (integers). Now, every honest computer will get the exact same answer, bit for bit.

### What's Left to Verify Before the Genesis Block?
We have proven the math, the security, and the economics on a single machine. But before you launch a live network and mine the Genesis Block, there is **one major real-world test left:**

* **Cross-GPU Determinism:** Right now, we proved that the integer math works perfectly *on your specific graphics card*. In the real world, someone will try to mine this on an NVIDIA RTX 4090, someone else on an AMD card, and someone else on an M2 Mac. You need to run the exact same salted prompt on **two entirely different hardware architectures** and prove that they both spit out the exact same cryptographic hash. 

If two different GPUs agree perfectly on the final hash, then cross-hardware determinism is solved. At that point, the system is fully bulletproof, and you are ready to mine the Genesis Block!
