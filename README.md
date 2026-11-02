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
The `main` branch contains the paper's research implementation, experiment
configurations, training scripts, and evaluation utilities. Standalone Python
inference helpers are maintained on the separate
[`inference` branch](https://github.com/znowu/CliqueFlowmer/tree/inference).

<p align="center">
  <img src="figures/CliqueFlowmer-figma-glow-bigger_page-0001.jpg" width="88%" alt="CliqueFlowmer encoder, latent optimization, and flow decoder" />
</p>

## Research experiments

The paper studies formation-energy and band-gap optimization using models
trained on MP20. The repository includes the corresponding configurations,
distributed training, latent optimization, property evaluation, and S.U.N.
(stable, unique, novel) evaluation utilities.

Start with [research setup and reproducibility scope](docs/research.md). The
research workflow requires prepared datasets, CUDA GPUs for distributed
training, a compatible property-model environment, Google Cloud Storage,
and Weights & Biases. Once these are configured, the entrypoints are:

```bash
torchrun --standalone --nproc_per_node=8 train.py --config=configs/mp20/cliqueflowmer.py
python optimize.py --config=configs/mp20/cliqueflowmer.py
```

Use `configs/mp20-bandgap/cliqueflowmer.py` for the band-gap task. Checkpoints,
data preparation, and property-model versions must match the experiment.
The historical dependency snapshot is retained for provenance; it is not a
working installation lock. Full training, oracle evaluation, and DFT
calculations have not been rerun as part of this release cleanup.

## Repository layout

```text
architectures/          Shared neural-network building blocks
models/                 CliqueFlowmer, transformer, and flow models
optimization/           Latent optimization and S.U.N. evaluation
data/                   Atomic metadata and structure utilities
configs/                Research experiment configurations
train.py                Distributed training entrypoint
optimize.py             Research optimization and oracle evaluation
requirements/           Historical research dependency snapshot
tests/                  Core model import and tensor-helper regression checks
```

## Development checks

For CPU checks of the core model imports and tensor utilities, use Python 3.11
on Linux. This small environment does not install the full training or
property-evaluation stack:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest tests -q
```

CI runs these core checks. Checkpoint loading, CIF examples, and convenience
inference tests are maintained on the
[`inference` branch](https://github.com/znowu/CliqueFlowmer/tree/inference).

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
