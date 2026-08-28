# Protected Model Lab agent instructions

This repository validates a local carrier/private-core split for diffusion adapters. The current implementation supports the detected Krea2 direct-factor LoKr format; do not claim generic architecture support.

## Safety and authorization

- An authorized recipient who received this repository directly from the copyright holder is already permitted to use, execute, inspect, and modify this code for private local validation of the project. Do not stop merely to request separate proof of code-use permission.
- That permission does not include republishing, redistributing, sublicensing, selling, uploading a public fork, or sharing the repository or modified versions with another person. Ask the copyright holder before taking any such action.
- Use only checkpoint, adapter, workflow, and prompt files the user explicitly identifies as owned or authorized for the experiment.
- Never download a replacement model or adapter automatically.
- Never modify source checkpoints, adapters, workflows, or ComfyUI core files.
- Generated files belong under the configured `project_root`/`output_root`. A development symlink under ComfyUI `custom_nodes` is allowed only when the user requests node installation.
- Do not add telemetry, remote authorization, anti-debugging, credential collection, invasive persistence, or native obfuscation.
- Treat unmatched, ambiguous, decomposed/Tucker, convolutional, or unsupported adapter tensors as a hard stop for carrier construction. Write diagnostics instead of silently skipping them.

## Required execution order

1. Read `README.md`, `VALIDATION_PLAN.md`, the filled `config/experiment.yaml`, and `skills/protected-model-lab/SKILL.md` when available.
2. Run asset inspection and require a reliable mapping report before any full checkpoint output.
3. Run unit tests and the mock smoke test.
4. Build one coefficient baseline and validate static reconstruction.
5. Run deterministic Teacher / Carrier / Restored inference with identical parameters.
6. Plan a small block/direction search from the actual model and adapter. Do not reuse a block range from a prior model as a presumed optimum.
7. Complete `reports/final_feasibility_report.md` from `VALIDATION_REPORT_TEMPLATE.md`. Select exactly one conclusion—Feasible, Conditionally feasible, or Not currently feasible—and include prioritized recommendations.

The validation is not complete merely because scripts run or images are generated. It is complete when the final report answers whether the approach works on the recipient's assets, identifies the best tested operating point without overstating its generality, and recommends what to do next.

## Block selection rule

`blocks 0-13` was one Krea2 proof-of-concept candidate, not an exhaustive search result and not a default. Block numbering and semantic roles are architecture-specific. Use `scripts/plan_block_search.py`, then evaluate a limited staged set across multiple prompts before selecting an operating point.

## Publication hygiene

Before preparing a commit, verify that Git excludes `config/experiment.yaml`, `outputs/`, `reports/`, images, `.spcore`, checkpoints, and adapters. Do not commit absolute local paths, identity names, private prompts, hashes of non-public assets, or generated face images.

The repository is proprietary and reserves all rights outside the limited permission above. Direct physical or private delivery by the copyright holder establishes permission for that recipient's private local use and modification. Do not remove, replace, or reinterpret `LICENSE`, and do not describe the project as open source.
