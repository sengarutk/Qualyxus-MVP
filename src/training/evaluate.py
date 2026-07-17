from __future__ import annotations
import argparse
import json
from ultralytics import YOLO
from src.config.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--real_data", type=str, default=None)
    parser.add_argument("--synthetic_data", type=str, default=None)
    parser.add_argument("--blended_data", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.load()
    data_yaml = cfg.get("dataset", "yolo", "yaml_path")

    model = YOLO(args.weights)
    metrics = model.val(data=data_yaml)
    print("[QUALYXUS] ✅ Validation complete.")
    print(metrics)

    def _extract_map(payload) -> float:
        for attr in ("box", "seg"):
            obj = getattr(payload, attr, None)
            if obj is not None and hasattr(obj, "map"):
                return float(getattr(obj, "map", 0.0) or 0.0)
        return float(getattr(payload, "map", 0.0) or 0.0)

    domain_gap = {}
    for label, path in {
        "real": args.real_data,
        "synthetic": args.synthetic_data,
        "blended": args.blended_data,
    }.items():
        if not path:
            continue
        result = model.val(data=path)
        domain_gap[label] = {"mAP": round(_extract_map(result), 4)}

    if domain_gap:
        print("Domain Gap Report")
        print(json.dumps(domain_gap, indent=2))


if __name__ == "__main__":
    main()
