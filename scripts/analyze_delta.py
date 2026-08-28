#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from safetensors import safe_open
from common.config import load_config
from delta_split.kron_svd import kron_singular_values
from lora_mapping.lokr import inspect_direct_lokr

if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config/experiment.yaml")
    layers, errors = inspect_direct_lokr(cfg.lora_files[0].path, cfg.base_model)
    if errors:
        raise SystemExit("mapping errors; inspect reports/lora_mapping_report.json")
    analysis=[]
    with safe_open(cfg.lora_files[0].path, framework="pt", device="cpu") as adapter:
        for layer in layers:
            s = kron_singular_values(adapter.get_tensor(layer.w1_key), adapter.get_tensor(layer.w2_key))
            energy=(s.square()/s.square().sum()).tolist()
            analysis.append({"model_key":layer.model_key,"effective_rank":len(s),"singular_values":s.tolist(),"normalized_energy":energy,"delta_frobenius_norm":float(torch.linalg.vector_norm(s))})
    out=cfg.project_root/"reports/delta_analysis.json"
    out.write_text(json.dumps({"errors":errors,"layers":analysis},indent=2),encoding="utf-8")
    print(out)
