# Protected Model Lab

Protected Model Lab is a local, research-oriented framework for testing whether an owned diffusion adapter can be split into:

```text
teacher = base checkpoint + configured adapter
teacher ≈ public carrier + private runtime core
```

The public carrier remains a normal inference checkpoint. The private `.spcore` is accepted only by the bundled ComfyUI custom nodes and is bound to one carrier SHA-256.

This is an engineering protection proof of concept, not cryptographic secrecy. It does not implement telemetry, remote authorization, anti-debugging, credential collection, or native binary obfuscation.

## License

Copyright (c) 2026. All rights reserved. This repository is not open source. A recipient who receives it directly from the copyright holder may use, run, inspect, and modify it for private local validation, but may not republish, redistribute, sublicense, sell, use it commercially, or share a public fork. See [LICENSE](LICENSE).

## Current support

The current implementation supports the format proven in the initial POC:

- Krea2 diffusion-only BF16 monolithic safetensors
- exactly one direct-factor LoKr containing `lokr_w1`, `lokr_w2`, and optional `alpha`
- coefficient, top/tail singular-direction, block-selected, and hybrid planning/build paths
- local ComfyUI `ModelPatcher` integration

It intentionally stops on unsupported or ambiguous tensors. SDXL, Flux, Z-Image, standard LoRA, DoRA, Tucker/decomposed LoKr, convolutional LoKr, and sharded checkpoint output must not be assumed supported without adding and testing an architecture adapter.

## What belongs on GitHub

Commit source code, tests, configuration templates, `AGENTS.md`, and the bundled agent skill. Do not commit filled configuration, reports, hashes of private assets, generated images, carriers, cores, checkpoints, or adapters. See [PUBLISHING.md](PUBLISHING.md).

## Setup

Clone the repository beside, not inside, the ComfyUI installation. Reuse ComfyUI's Python environment so PyTorch/CUDA and ComfyUI APIs match the installed runtime:

```bash
git clone <YOUR_REPOSITORY_URL> protected_model_lab
cd protected_model_lab
cp config/experiment.example.yaml config/experiment.yaml
```

Edit every path and the inference section in `config/experiment.yaml`. Then define the interpreter for the current shell:

```bash
export COMFYUI_ROOT=/ABSOLUTE/PATH/ComfyUI
export COMFYUI_PYTHON="$COMFYUI_ROOT/.venv/bin/python"
"$COMFYUI_PYTHON" -m pip install -e . --no-deps
```

Do not let pip replace the working ComfyUI PyTorch installation. Install missing lightweight packages deliberately after reviewing them.

## Execution sequence

### 1. Inspect assets and validate mapping

```bash
"$COMFYUI_PYTHON" scripts/inspect_assets.py config/experiment.yaml
```

Do not continue unless `reports/lora_mapping_report.json` says `reliable: true` and reports zero errors.

### 2. Run tests and mock smoke checks

```bash
"$COMFYUI_PYTHON" -m pytest -q
```

### 3. Build the coefficient baseline

```bash
"$COMFYUI_PYTHON" scripts/build_carrier.py --config config/experiment.yaml --private-fraction 0.35
"$COMFYUI_PYTHON" scripts/build_core.py --config config/experiment.yaml --private-fraction 0.35
"$COMFYUI_PYTHON" scripts/run_validation.py --config config/experiment.yaml --private-fraction 0.35
```

Repeat only the configured small fraction set after the first pair passes.

### 4. Analyze adapter energy and plan block candidates

```bash
"$COMFYUI_PYTHON" scripts/analyze_delta.py config/experiment.yaml
"$COMFYUI_PYTHON" scripts/plan_block_search.py --config config/experiment.yaml
```

Block ranges are architecture- and adapter-specific. The initial POC tested only `blocks 0-13`; it was not compared against another block range, so it established feasibility rather than an optimum.

### 5. Build controlled non-coefficient candidates

```bash
"$COMFYUI_PYTHON" scripts/build_svd_candidate.py \
  --config config/experiment.yaml \
  --strategy top_singular_private \
  --private-rank 8

"$COMFYUI_PYTHON" scripts/build_svd_candidate.py \
  --config config/experiment.yaml \
  --strategy block_selected_private \
  --private-rank 0 \
  --selected-blocks 0-13
```

Replace the example block range with candidates justified by the generated plan and deterministic image comparisons.

### 6. Install the custom node for development

```bash
"$COMFYUI_PYTHON" scripts/install_dev_node.py --config config/experiment.yaml
systemctl --user restart comfyui.service
```

The installer creates a symlink under `custom_nodes`; it does not patch ComfyUI core files.

### 7. Run deterministic ComfyUI validation

```bash
"$COMFYUI_PYTHON" scripts/run_comfy_validation.py \
  --config config/experiment.yaml \
  --private-fraction 0.65
```

Teacher, carrier, and restored workflows must use identical prompts, seeds, dimensions, sampler, scheduler, steps, CFG, text encoder, and VAE.

### 8. Generate local metrics and manifests

```bash
"$COMFYUI_PYTHON" scripts/generate_metrics.py --help
"$COMFYUI_PYTHON" scripts/generate_output_manifest.py
```

No metric model is downloaded automatically. Pixel error is diagnostic; use SSIM only when already installed, and add perceptual metrics only after reviewing their local dependencies.

## Agent use

Codex reads repository-level instructions from `AGENTS.md`. The reusable skill is in `skills/protected-model-lab/` and can be installed with:

```bash
python3 scripts/install_agent_skill.py
```

After installation, a typical request is:

```text
Use $protected-model-lab to inspect my authorized local checkpoint and adapter, then run the smallest safe split feasibility experiment.
```

The repository remains usable by other coding agents through `AGENTS.md` and this README even if they do not support Codex skills.

For a recipient validation, follow [VALIDATION_PLAN.md](VALIDATION_PLAN.md). The Agent must copy [VALIDATION_REPORT_TEMPLATE.md](VALIDATION_REPORT_TEMPLATE.md) to `reports/final_feasibility_report.md` and return an explicit feasibility decision plus prioritized recommendations. A successful command run by itself is not the deliverable.

## Interpretation

A useful operating point requires all of the following across multiple prompts:

- carrier-only remains functional;
- carrier-only loses a measurable part of the target behavior;
- restored recovers at least 90% of the teacher-to-carrier gap;
- static reconstruction error is near numerical precision before storage rounding;
- the core remains materially smaller than the checkpoint;
- a non-coefficient split is tested;
- results and failures are reported without selecting a convenient single image.
