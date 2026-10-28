# CliqueFlowmer inference and demos

**[Open the live crystal workspace](https://cliqueflowmer.znowu.workers.dev)**

Upload one CIF, inspect its representation, reconstruct or optimize a candidate,
and rotate and download the resulting structure. No installation is needed for
the public website.

| Component | Purpose |
| --- | --- |
| [website/](website/README.md) | Maintained custom website, Cloudflare proxy, and CPU API |
| [web_demo/](web_demo/README.md) | Shared bounded inference service and optional Streamlit UI |
| [inference.py](inference.py) | Python encoding, decoding, reconstruction, and optimization helpers |
| [data_tools.py](data_tools.py) | Lightweight structure conversion and length-prior helpers |
| [relaxation.py](relaxation.py) | Optional research relaxation with the separate MatGL stack |

The maintained demos share a revision- and checksum-pinned public checkpoint,
three example CIFs, and a JSON length prior. They run on CPU without training
datasets, MatGL, or DGL. Start with the linked deployment instructions or install
the Python inference dependencies from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The superseded Gradio/Colab entrypoint and conflicting installation lists were
removed. Their history remains in Git. The research inference helpers, including
optional relaxation, are retained; see [research setup](../docs/research.md).
