#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safetensors import safe_open
from common.config import load_config
from common.hashing import sha256_file
from core_format.spcore import CoreLayer, ProtectedCore, write_core
from lora_mapping.lokr import inspect_direct_lokr

parser=argparse.ArgumentParser()
parser.add_argument("--config",default=str(ROOT/"config/experiment.yaml"))
parser.add_argument("--private-fraction",type=float,default=0.35)
args=parser.parse_args()
cfg=load_config(args.config)
layers,errors=inspect_direct_lokr(cfg.lora_files[0].path,cfg.base_model)
if errors: raise SystemExit("refusing core build because mapping is not reliable")
tag=f"p{round(args.private_fraction*100):03d}"
carrier=cfg.output_root/"carriers"/f"carrier_coeff_{tag}.safetensors"
if not carrier.is_file(): raise SystemExit(f"carrier not found: {carrier}")
items=[]
with safe_open(cfg.lora_files[0].path,framework="pt",device="cpu") as adapter:
    for layer in layers:
        items.append(CoreLayer(layer.model_key,adapter.get_tensor(layer.w1_key),adapter.get_tensor(layer.w2_key)*args.private_fraction,cfg.lora_files[0].strength_model))
core=ProtectedCore(cfg.architecture,sha256_file(carrier),f"coefficient_split_p={args.private_fraction}",tuple(items))
out=cfg.output_root/"cores"/f"core_coeff_{tag}.spcore"
write_core(out,core)
print(out)
