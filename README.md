# CliqueFlowmer

**Optimize Crystals with Clique and Flow Models** · Accepted at **NeurIPS 2026**

Jakub Grudzien Kuba, Benjamin Kurt Miller, Sergey Levine, Pieter Abbeel

**[Paper](https://arxiv.org/abs/2603.06082)** ·
**[Model weights](https://huggingface.co/iamkuba/CliqueFlowmer)** ·
**[Live demo](https://cliqueflowmer.znowu.workers.dev)**

<p align="center">
  <img src="bair.png" width="280" alt="Berkeley Artificial Intelligence Research" />
</p>

CliqueFlowmer learns a continuous representation of crystals for offline
materials optimization. It encodes a structure, optimizes its latent
representation using evolution strategies, and decodes a candidate crystal.
This repository contains the research implementation, experiment
configurations, evaluation utilities, and Python inference helpers.

<p align="center">
  <img src="figures/CliqueFlowmer-figma-glow-bigger_page-0001.jpg" width="88%" alt="CliqueFlowmer encoder, latent optimization, and flow decoder" />
</p>

## Installation and inference

The maintained inference environment uses **Python 3.11 on Linux** and CPU
PyTorch. It requires no training dataset, CUDA, MatGL, or DGL.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Load the released checkpoint and encode an example crystal:

```python
from pymatgen.core import Structure
from serving.pretrained import load_pretrained_model
from serving.inference import encode_structure

model = load_pretrained_model(device="cpu")
structure = Structure.from_file("examples/silicon.cif")
z = encode_structure(model, structure)
print(z.shape)  # torch.Size([121])
```

The checkpoint revision and SHA-256 are pinned in
[serving/checkpoint.json](serving/checkpoint.json). For offline use, pass
`checkpoint_path` to `load_pretrained_model`, or set `CLIQUEFLOWMER_CHECKPOINT`;
the same checksum is verified. The length prior is bundled as JSON.

This convenience checkpoint uses Alexandria pretraining and MP20 fine-tuning;
it is separate from the MP20-only models used for the paper's reported
experiments. Its optimization target is **formation energy per atom**, but the
artifact omits target-scaling metadata: raw predictor outputs are model scores,
not calibrated eV/atom or independently evaluated candidate energies.

See [Python inference examples](serving/README.md) for reconstruction, latent
optimization, CIF export, and loading other checkpoints. Generated structures
may change composition and require independent physical evaluation.

For an interactive introduction without installation, use the
[hosted crystal workspace](https://cliqueflowmer.znowu.workers.dev).

## Research experiments

The repository includes MP20 formation-energy and band-gap configurations,
distributed training, latent optimization, and S.U.N. evaluation utilities.
The original research workflow requires a separate oracle environment,
prepared datasets, Google Cloud Storage, and Weights & Biases.

See [research setup and reproducibility scope](docs/research.md) before running
`train.py` or `optimize.py`. The older environment snapshot is retained for
provenance; it is not a working installation lock. Current automated checks
cover core model utilities and inference. Full training, oracle evaluation,
and DFT calculations have not been rerun as part of this release cleanup.

## Repository layout

```text
architectures/          Shared neural-network building blocks
models/                 CliqueFlowmer, transformer, and flow models
optimization/           Latent optimization and S.U.N. evaluation
data/                   Atomic metadata and structure utilities
configs/                Research experiment configurations
train.py                Distributed training entrypoint
optimize.py             Research optimization and oracle evaluation
serving/                Python inference, checkpoint loading, optional relaxation
examples/               Example crystal structures in CIF format
requirements/           Inference constraints and historical research dependencies
tests/                  Model and inference regression checks
```

## Development checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

CI runs the Python suite and dependency checks. A manually triggered run also
checks inference with the actual published checkpoint.

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
logos, and separately distributed datasets/model weights retain their own
licenses and notices. The code license does not relicense the paper text or
its figures.
