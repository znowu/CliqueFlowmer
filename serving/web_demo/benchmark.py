"""Measure the real CPU workflow; no training datasets or GPU are needed.

Run from the repository root: python -m serving.web_demo.benchmark
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import resource
import time

from serving.web_demo.service import EXAMPLES, MANIFEST, load_pinned_model, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=["encode", "reconstruct", "optimize"], default="optimize")
    parser.add_argument("--budget", choices=["quick", "standard"], default="quick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("serving/web_demo/benchmark-results.local.json"))
    args = parser.parse_args()
    started = time.monotonic()
    model = load_pinned_model()
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pymatgen", "streamlit")},
        "checkpoint": MANIFEST, "device": "cpu", "torch_threads": __import__("torch").get_num_threads(),
        "model_load_seconds": time.monotonic() - started,
        "operation": args.operation, "budget": args.budget, "seed": args.seed,
        "cases": [],
    }
    failed = False
    for name, cif in EXAMPLES.items():
        start = time.monotonic()
        def check():
            if time.monotonic() - start > 300:
                raise TimeoutError("Benchmark case exceeded five minutes.")
        try:
            result = run_pipeline(model, cif, operation=args.operation, budget=args.budget,
                                  seed=args.seed, cancel_check=check)
            case = {
                "example": name, "status": "completed", "seconds": time.monotonic() - start,
                "input": result["input_summary"], "output": result["output_summary"],
                "initial_score": result["initial_score"], "final_score": result["final_score"],
                "decode_attempts": result["metadata"]["decode_attempts"],
            }
        except Exception as exc:
            failed = True
            case = {"example": name, "status": "failed", "seconds": time.monotonic() - start, "error": str(exc)}
        # Linux reports KiB; keep the platform in the report for interpretation.
        case["process_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
