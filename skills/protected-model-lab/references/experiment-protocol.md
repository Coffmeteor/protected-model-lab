# Experiment protocol

## Preflight

Use the Python interpreter from the configured ComfyUI installation. Run:

```bash
"$COMFYUI_PYTHON" scripts/inspect_assets.py config/experiment.yaml
"$COMFYUI_PYTHON" -m pytest -q
```

Inspect `reports/lora_mapping_report.json`. Continue only when mapping is reliable and error-free.

## Static baseline

Build one coefficient pair first:

```bash
"$COMFYUI_PYTHON" scripts/build_carrier.py --config config/experiment.yaml --private-fraction 0.35
"$COMFYUI_PYTHON" scripts/build_core.py --config config/experiment.yaml --private-fraction 0.35
"$COMFYUI_PYTHON" scripts/run_validation.py --config config/experiment.yaml --private-fraction 0.35
```

Theoretical public/private deltas must reconstruct the original update near float32 numerical precision. Report BF16 storage rounding separately.

## ComfyUI runtime

Install the development node only after static validation and only when the user authorizes the symlink:

```bash
"$COMFYUI_PYTHON" scripts/install_dev_node.py --config config/experiment.yaml
```

Restart the user's established ComfyUI service or process without changing its launch configuration. Verify the three protected nodes appear in `/object_info` before submitting inference.

Run Teacher, Carrier, and Restored with identical inference values from the config. Preserve expanded workflow JSON and local outputs under the project.

## Evidence

Use multiple prompts before selecting a candidate. Prefer already-installed perceptual metrics; never download large metric models automatically. Contact sheets and human review complement metrics but do not replace fixed-parameter records.

Hash generated carriers/cores and preserve manifests. The `.spcore` must reject a different carrier hash.
