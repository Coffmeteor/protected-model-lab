#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carrier_builder.coefficient import build_coefficient_carrier
from common.config import load_config
from lora_mapping.lokr import inspect_direct_lokr

parser=argparse.ArgumentParser()
parser.add_argument("--config",default=str(ROOT/"config/experiment.yaml"))
parser.add_argument("--private-fraction",type=float,default=0.35)
args=parser.parse_args()
cfg=load_config(args.config)
layers,errors=inspect_direct_lokr(cfg.lora_files[0].path,cfg.base_model)
if errors: raise SystemExit("refusing carrier build because mapping is not reliable")
tag=f"p{round(args.private_fraction*100):03d}"
out=cfg.output_root/"carriers"/f"carrier_coeff_{tag}.safetensors"
build_coefficient_carrier(cfg.base_model,cfg.lora_files[0].path,out,layers,args.private_fraction,cfg.lora_files[0].strength_model)
print(out)
