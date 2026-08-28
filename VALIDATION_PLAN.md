# Recipient validation plan

## Goal

Run this project on the recipient's authorized local checkpoint, adapter, and ComfyUI workflow, then return one evidence-backed conclusion to the copyright holder:

- **Feasible**: the split works on the tested assets and a useful non-coefficient operating point was demonstrated.
- **Conditionally feasible**: the mechanism works, but quality, compatibility, runtime, or evidence is not yet sufficient for the intended use.
- **Not currently feasible**: mapping, reconstruction, runtime integration, or restored quality fails on the tested assets.

The goal is not to prove cryptographic security. Evaluate resistance only against immediate casual reuse, standard LoRA loading/merging, and low-effort republishing.

## Required TODO

### Stage 0 — Preserve and record

- [ ] Read `AGENTS.md`, `README.md`, this plan, and the filled configuration.
- [ ] Confirm every model, adapter, workflow, and prompt is locally owned or authorized.
- [ ] Record GPU, RAM, OS, Python, PyTorch, CUDA, ComfyUI revision, and free disk space.
- [ ] Confirm source files are read-only from the experiment's perspective and outputs are outside ComfyUI/source directories.
- [ ] Run the unit and mock smoke tests.

### Stage 1 — Compatibility gate

- [ ] Run asset inspection and create the environment, asset, and mapping reports.
- [ ] Confirm architecture from tensors/configuration rather than the filename.
- [ ] Account for every adapter tensor and module.
- [ ] Stop carrier construction if any key is unmatched, ambiguous, unsupported, or shape-incompatible.
- [ ] If stopped, complete the final report with the exact missing architecture/mapping work and do not manufacture results.

### Stage 2 — Coefficient control

- [ ] Build one coefficient carrier/core pair, starting with private fraction `0.35`.
- [ ] Reopen and verify every carrier key and shape.
- [ ] Verify core/carrier hash binding and rejection of a deliberately mismatched carrier.
- [ ] Measure per-layer and aggregate float32 reconstruction error separately from BF16 storage error.
- [ ] Run fixed-seed Teacher / Carrier / Restored inference with identical parameters.

This stage proves implementation correctness only. Do not present coefficient splitting as meaningful protection.

### Stage 3 — Small non-coefficient search

- [ ] Run delta analysis and `scripts/plan_block_search.py` on the recipient's actual adapter.
- [ ] Do not treat the original POC's `blocks 0-13` as a default.
- [ ] Select no more than four first-stage candidates: representative top/tail singular, contiguous block, and/or hybrid candidates justified by the analysis.
- [ ] Compare candidates on 6–8 fixed prompts before expanding the search.
- [ ] Promote at most two candidates to the final prompt set.
- [ ] Cache reusable analysis and do not launch a combinatorial search.

### Stage 4 — Final deterministic evaluation

- [ ] Evaluate the coefficient control and promoted non-coefficient candidates on approximately 30 prompts, or document why fewer were used.
- [ ] Keep prompts, negative prompts, seeds, resolution, sampler, scheduler, steps, guidance, VAE, text encoder, and architecture-specific settings identical.
- [ ] Measure Teacher-to-Carrier gap and Restored-to-Teacher distance per prompt.
- [ ] Report recovery distributions, not only the best prompt or average.
- [ ] Generate labeled Teacher / Carrier / Restored contact sheets for human review.
- [ ] Use only locally available perceptual metrics; do not download large metric models automatically.
- [ ] Measure load time, generation time, peak VRAM/RAM where available, and artifact sizes.

### Stage 5 — Return the decision

- [ ] Copy `VALIDATION_REPORT_TEMPLATE.md` to `reports/final_feasibility_report.md`.
- [ ] Fill every section, using `not measured` instead of omitting unavailable evidence.
- [ ] Select exactly one conclusion: Feasible, Conditionally feasible, or Not currently feasible.
- [ ] Give prioritized next steps divided into **must fix**, **worth testing**, and **defer**.
- [ ] State whether further static search is justified before any dynamic-core or native-runtime work.
- [ ] Return the report plus only the non-sensitive summary artifacts authorized by the owner.

## Suggested decision rubric

Use these as decision aids, not as permission to hide failures:

### Feasible

- Mapping is complete: zero silently ignored, unmatched, or ambiguous adapter tensors.
- Float32 public/private reconstruction is near numerical precision; storage rounding is reported separately.
- Carrier/core hash binding and mismatch rejection pass.
- A non-coefficient candidate creates a repeatable Teacher-to-Carrier gap beyond run-to-run noise and a noticeable loss of the target behavior in human review.
- Restored median recovery is at least `0.90`, and at least 80% of prompts recover `0.85` or better.
- Carrier-only remains functional, the core is materially smaller than the checkpoint, and overhead is acceptable for the intended workflow.

### Conditionally feasible

- Static mapping and reconstruction work, but the visual gap is weak, recovery is inconsistent, the prompt set is too small, runtime is too costly, or only the coefficient control has passed.
- State the smallest next experiment that could change the conclusion.

### Not currently feasible

- Reliable mapping cannot be established, unsupported tensors affect the teacher, restored output cannot approach the teacher, core/carrier binding fails, or the runtime cannot generate correctly without changing inference conditions.

## Stop conditions

Stop full carrier creation and report diagnostics when mapping is incomplete. Stop expanding candidate search when restoration fails systematically, disk/VRAM safety margins are insufficient, or three attempts at the same execution blocker have failed. Do not begin dynamic-core training, C++/CUDA work, obfuscation, or broad architecture support as part of this validation.
