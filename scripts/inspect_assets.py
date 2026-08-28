#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asset_inspection.report import generate_reports
from common.config import load_config

if __name__ == "__main__":
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config/experiment.yaml")
    _, mapping = generate_reports(config)
    print(f"mapped={mapping['mapped_count']} errors={len(mapping['errors'])} reliable={mapping['reliable']}")
