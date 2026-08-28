---
name: protected-model-lab
description: Inspect and validate an authorized local diffusion checkpoint and adapter, then build and evaluate public-carrier/private-core splits with the Protected Model Lab repository. Use for coefficient, singular-direction, block, or hybrid split feasibility work; do not use to bypass model licenses or to claim cryptographic protection.
---

# Protected Model Lab

Respect the repository's proprietary `LICENSE`. A recipient who obtained the repository directly from the copyright holder may use and modify it for private local validation without supplying additional proof of code-use permission. Do not describe it as open source, remove the notice, redistribute it, publish a fork, or extend that permission to model assets the user does not own or control.

Use the repository scripts as the deterministic implementation. Do not reimplement checkpoint writing or tensor mapping ad hoc.

## Before execution

1. Locate the repository root containing `pyproject.toml`, `AGENTS.md`, and `scripts/inspect_assets.py`.
2. Read its `AGENTS.md`, `VALIDATION_PLAN.md`, and filled `config/experiment.yaml`.
3. Confirm the user owns or is authorized to use every configured checkpoint, adapter, workflow, and prompt file.
4. Keep all generated artifacts under configured project/output roots. Never modify source assets or ComfyUI core files.

Read [references/experiment-protocol.md](references/experiment-protocol.md) when running the experiment. Read [references/split-selection.md](references/split-selection.md) when planning or comparing coefficient, SVD, block, or hybrid candidates.

## Decision boundaries

- Current code supports the detected Krea2 direct-factor LoKr path. Do not infer support for another architecture or adapter format from filenames.
- Stop carrier construction if any key is unmatched, ambiguous, unsupported, shape-incompatible, or silently discarded by ComfyUI.
- Start with coefficient splitting to validate mapping, streaming writes, container binding, and runtime reconstruction. Label it a baseline, not a security result.
- Test at least one non-coefficient candidate before making a protection claim.
- Treat block numbers as architecture-specific experimental variables. `blocks 0-13` was one prior POC candidate, not a universal setting or exhaustive optimum.
- Do not download evaluation models, alternate checkpoints, or adapters automatically.
- Do not begin native C++/CUDA work or dynamic-core training until static restoration passes across multiple prompts.

## Completion standard

Copy `VALIDATION_REPORT_TEMPLATE.md` to `reports/final_feasibility_report.md`, complete every section, and select exactly one conclusion: Feasible, Conditionally feasible, or Not currently feasible. Include prioritized **must fix**, **worth testing**, and **defer** recommendations.

Report these separately:

- mapping completeness and every unsupported key;
- theoretical float32 reconstruction error;
- storage/runtime precision error;
- teacher-to-carrier gap;
- restored recovery across the fixed prompt set;
- artifact sizes and measured runtime;
- assumptions, incomplete work, failed thresholds, and unsupported features.

One attractive image is a smoke result, not a selected operating point. Require a small staged prompt/candidate matrix before calling any split best.
