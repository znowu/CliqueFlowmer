# CliqueFlowmer

**Optimize Crystals with Clique and Flow Models** · Accepted at **NeurIPS 2026**

Jakub Grudzien Kuba, Benjamin Kurt Miller, Sergey Levine, Pieter Abbeel

**[Live demo](https://cliqueflowmer.znowu.workers.dev)** ·
**[Paper](https://arxiv.org/abs/2603.06082)** ·
**[Model weights](https://huggingface.co/iamkuba/CliqueFlowmer)** ·
**[Camera-ready sources](paper/)**

<p align="center">
  <img src="bair.png" width="280" alt="Berkeley Artificial Intelligence Research" />
</p>

CliqueFlowmer learns a continuous representation of crystals for offline
materials optimization. It encodes a structure, optimizes its latent
representation using evolution strategies, and decodes a candidate crystal.

<p align="center">
  <img src="figures/CliqueFlowmer-figma-glow-bigger_page-0001.jpg" width="88%" alt="CliqueFlowmer encoder, latent optimization, and flow decoder" />
</p>

## Try the model

Open the **[crystal workspace](https://cliqueflowmer.znowu.workers.dev)**. Start
with silicon, strontium titanate, or Cs₂AgBiBr₆, or upload your own CIF with up to
20 ordered sites. You can:

- inspect a 121-dimensional representation;
- reconstruct or optimize a crystal;
- rotate, zoom, and compare original and generated structures;
- download CIF structures and representations in NumPy, CSV, or JSON format.

The demo runs on CPU. Its target is **formation energy per atom (eV/atom)**;
displayed values are raw model scores because the released checkpoint does not
include target-scaling metadata. Generated structures are unrelaxed, may change
composition, and need independent physical evaluation.

The public frontend runs on Cloudflare, with inference on Grace. Availability
depends on the CPU service and its tunnel. See the
[website setup and limits](serving/website/README.md).

## Local installation

The maintained inference environment uses **Python 3.11 on Linux** and CPU
PyTorch. It requires no training dataset, CUDA, MatGL, or DGL.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m serving.web_demo.benchmark --operation encode --output /tmp/cliqueflowmer-encode.json
```

The first operation downloads the public checkpoint. Its revision and SHA-256
are pinned in [checkpoint.json](serving/web_demo/checkpoint.json). For offline
use, set `CLIQUEFLOWMER_CHECKPOINT` to a previously downloaded copy; the same
checksum is verified.

For the complete local interface, follow the
[custom website instructions](serving/website/README.md). The
[Streamlit application](serving/web_demo/README.md) remains an optional, simpler
frontend to the same inference service.

## Research implementation

The repository includes the models, MP20 formation-energy and band-gap
configurations, training, latent optimization, and S.U.N. evaluation utilities.
The original research workflow uses a separate oracle environment, prepared
datasets, Google Cloud Storage, and Weights & Biases. It is **not installed by
the CPU inference requirements**.

See [research setup and reproducibility scope](docs/research.md) before running
`train.py` or `optimize.py`. The older environment snapshot is retained for
provenance; it is not a working installation lock. This release validates the
CPU demo and paper build, rather than claiming to rerun the full paper study.

## Repository layout

```text
architectures/          Shared neural-network building blocks
models/                 CliqueFlowmer, transformer, and flow models
optimization/           Latent optimization and S.U.N. evaluation
data/                   Atomic metadata and structure utilities
configs/                Research experiment configurations
train.py                Distributed training entrypoint
optimize.py             Research optimization and oracle evaluation
serving/inference.py    Python inference helpers
serving/web_demo/       Shared CPU service, examples, optional Streamlit UI
serving/website/        Custom React website, Cloudflare Worker, and CPU API
paper/                  Camera-ready LaTeX source and required figures
tests/                  Model, inference, API, and representation-export checks
```

## Development checks

```bash
python -m pip install -r requirements-dev.txt
npm ci --prefix serving/website/site
npm run typecheck --prefix serving/website/site
npm run build --prefix serving/website/site
python -m pytest tests -q
```

Use Node.js 22.13 or later. Installing the frontend dependencies enables the
NumPy/CSV/JSON export round-trip test. CI checks the Python suite, TypeScript,
frontend build, and Worker bundle. A manually triggered workflow additionally
runs inference with the actual published checkpoint.

Build the camera-ready paper with `bash paper/build.sh`; see
[paper/README.md](paper/README.md) for TeX dependencies.

## Citation

The paper is accepted at NeurIPS 2026. Until proceedings metadata is available,
please cite the public preprint:

```bibtex
@article{kuba2026offline,
  title={Offline Materials Optimization with CliqueFlowmer},
  author={Kuba, Jakub Grudzien and Miller, Benjamin Kurt and Levine, Sergey and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2603.06082},
  year={2026}
}
```

## License

The code is released under the [MIT license](LICENSE). Third-party libraries,
logos, the NeurIPS style, and separately distributed datasets/model weights
retain their own licenses and notices. The code license does not relicense the
paper text or its figures.
