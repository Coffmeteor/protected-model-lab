#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def request_json(request_or_url, timeout: int = 120):
    with urllib.request.urlopen(request_or_url, timeout=timeout) as response:
        return json.load(response)


def load_settings(config_path: Path) -> dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inference = raw.get("inference") if isinstance(raw, dict) else None
    if not isinstance(inference, dict):
        raise ValueError("config must contain an inference mapping; copy the fields from experiment.example.yaml")
    required = ("teacher_model_name", "adapter_name", "text_encoder_name", "text_encoder_type", "vae_name", "prompt")
    missing = [key for key in required if not inference.get(key)]
    if missing:
        raise ValueError("missing inference settings: " + ", ".join(missing))
    return inference


def build_workflow(variant: str, settings: dict, private_fraction: float, candidate_tag: str | None):
    seed = int(settings.get("seed", 812301))
    coefficient_tag = f"p{round(private_fraction * 100):03d}"
    nodes = {
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": settings["text_encoder_name"], "type": settings["text_encoder_type"]}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": settings["vae_name"]}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": settings["prompt"], "clip": ["clip", 0]}},
        "negative": {"class_type": "CLIPTextEncode", "inputs": {"text": settings.get("negative_prompt", ""), "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": int(settings.get("width", 768)), "height": int(settings.get("height", 1024)), "batch_size": 1}},
    }
    if variant == "teacher":
        nodes["base"] = {"class_type": "UNETLoader", "inputs": {"unet_name": settings["teacher_model_name"], "weight_dtype": "default"}}
        nodes["teacher"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["base", 0], "lora_name": settings["adapter_name"], "strength_model": float(settings.get("adapter_strength_model", 1.0))}}
        model = ["teacher", 0]
    else:
        artifact_tag = candidate_tag or f"coeff_{coefficient_tag}"
        nodes["carrier"] = {"class_type": "ProtectedCarrierLoader", "inputs": {"carrier_name": f"carrier_{artifact_tag}.safetensors"}}
        model = ["carrier", 0]
        if variant == "restored":
            nodes["core"] = {"class_type": "ProtectedCoreLoader", "inputs": {"core_name": f"core_{artifact_tag}.spcore", "protected_identity": ["carrier", 1]}}
            nodes["restored"] = {"class_type": "ApplyProtectedCore", "inputs": {"model": model, "protected_core": ["core", 0], "strength": 1.0}}
            model = ["restored", 0]
    nodes["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model,
            "seed": seed,
            "steps": int(settings.get("steps", 8)),
            "cfg": float(settings.get("cfg", 1.0)),
            "sampler_name": settings.get("sampler_name", "euler"),
            "scheduler": settings.get("scheduler", "beta"),
            "positive": ["positive", 0],
            "negative": ["negative", 0],
            "latent_image": ["latent", 0],
            "denoise": float(settings.get("denoise", 1.0)),
        },
    }
    nodes["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}}
    output_variant = variant if candidate_tag is None else f"{variant}_{candidate_tag}"
    output_variant = re.sub(r"[^a-z0-9_]", "_", output_variant.lower())
    nodes["save"] = {"class_type": "ProtectedValidationSave", "inputs": {"images": ["decode", 0], "variant": output_variant, "seed": seed}}
    return {"prompt": nodes, "client_id": str(uuid.uuid4())}, seed


def run(api: str, variant: str, settings: dict, private_fraction: float, candidate_tag: str | None):
    payload, seed = build_workflow(variant, settings, private_fraction, candidate_tag)
    suffix = "" if candidate_tag is None else "_" + candidate_tag
    expanded = ROOT / "outputs/workflows" / f"{variant}{suffix}_api.json"
    expanded.parent.mkdir(parents=True, exist_ok=True)
    expanded.write_text(json.dumps(payload["prompt"], indent=2, ensure_ascii=False), encoding="utf-8")
    request = urllib.request.Request(api + "/prompt", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    response = request_json(request)
    if "error" in response:
        raise RuntimeError(json.dumps(response, ensure_ascii=False))
    prompt_id = response["prompt_id"]
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        time.sleep(2)
        history = request_json(f"{api}/history/{prompt_id}")
        if prompt_id not in history:
            continue
        status = history[prompt_id].get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(json.dumps(status, ensure_ascii=False))
        print(variant, prompt_id, f"seed={seed}")
        return
    raise TimeoutError(prompt_id)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "config/experiment.yaml")
parser.add_argument("--api", default="http://127.0.0.1:8188")
parser.add_argument("--private-fraction", type=float, default=0.65)
parser.add_argument("--candidate-tag")
parser.add_argument("--skip-teacher", action="store_true")
args = parser.parse_args()
settings = load_settings(args.config.resolve())
variants = ("carrier", "restored") if args.skip_teacher else ("teacher", "carrier", "restored")
for selected_variant in variants:
    run(args.api.rstrip("/"), selected_variant, settings, args.private_fraction, args.candidate_tag)
