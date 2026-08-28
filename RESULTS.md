# Results

## Measured results

- Teacher: a local Krea2 Turbo BF16 checkpoint plus an authorized direct-factor identity LoKr at strength 1.0.
- Base: 430 tensors. Private source hashes are intentionally omitted from the public repository.
- Adapter: 768 tensors / 256 modules. Its filename, identity label, and source hash are intentionally omitted.
- Mapping: 256/256 targets, no unmatched or unsupported keys.
- Unit/smoke tests: 6 passed.
- Coefficient p=0.35 and p=0.65 pairs were built and reopened successfully.
- Custom nodes registered and executed through the local ComfyUI API.
- Coefficient p=0.65 recovery: 99.9353%; restored SSIM 0.99691.
- Top-singular rank-8 recovery: 97.5660%; restored SSIM 0.99612.
- Blocks 0–13 recovery: 99.8425%; restored SSIM 0.99729.
- Blocks 0–13 private outperformed the tested top-rank-8 control on carrier gap while preserving restoration.
- It was the only block range generated, so the experiment did not establish a best block range. It was neither a traversal result nor a universal optimum.

## Assumptions

- The existing Krea2 Turbo BF16 evaluation route matched the authorized local adapter used for the POC.
- The adapter has no text-encoder tensors, so CLIP strength is zero.
- Installed ComfyUI direct-factor LoKr semantics are authoritative: stored alpha is not applied when direct `lokr_w1` and `lokr_w2` are present.

## Incomplete components

- Roughly 30-prompt evaluation and per-prompt contact sheets
- LPIPS/DINO/CLIP similarity; no model was downloaded
- Latent/model-output hooks and exact peak VRAM/RAM measurement
- Tail-singular and hybrid image runs; construction paths exist but were not run
- Sharded safetensors carrier output
- Lazy per-layer `.spcore` reads; current 57–117 MB payloads load as one handle
- Architectures beyond detected Krea2 direct LoKr

## Failed or below-target results

- Coefficient p=0.35 BF16 weight diagnostic: 81.15% recovery.
- Coefficient p=0.65 BF16 weight diagnostic: 89.787%, slightly below 90% because of two BF16 rounding stages.
- Top-rank-8 carrier retained substantial teacher similarity (SSIM 0.89943), so rank 8 is too small for a strong split.

## Unsupported features

- DoRA, Tucker/decomposed or convolutional LoKr, and ambiguous mappings
- Native C++/CUDA backend or binary obfuscation
- Cryptographic secrecy claims
- Remote authorization, telemetry, anti-debugging, or credential handling
