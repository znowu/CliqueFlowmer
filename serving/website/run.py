"""Launch the CPU API, optionally reading a private JSON environment file.

    python -m serving.website.run --config .cache-demo/website-runtime.json

The file contains only CLIQUEFLOWMER_* variables plus HOST and PORT. Never
commit it. For production, supply those environment variables directly instead.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.config:
        config = json.loads(args.config.read_text())
        if not isinstance(config, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            or not (k.startswith("CLIQUEFLOWMER_") or k in {"HOST", "PORT"})
            for k, v in config.items()
        ):
            parser.error("Use a JSON object of CLIQUEFLOWMER_* string values, HOST, and PORT.")
        os.environ.update(config)
    if not os.environ.get("CLIQUEFLOWMER_PROXY_KEY"):
        parser.error("Set CLIQUEFLOWMER_PROXY_KEY before exposing the model service.")
    import uvicorn
    uvicorn.run("serving.website.api:app", host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "7860")), workers=1,
                access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
