# Research setup and reproducibility scope

This page describes the original research scripts, whose dependency and data
requirements are different from the lightweight Python inference environment
documented in [the root README](../README.md).

## What is included

The reference CliqueFlowmer and transformer architectures, ES and gradient
learners, MP20 configurations, distributed training, structure conversion,
relaxation helpers, property evaluation, and S.U.N. utilities are included.
The public checkpoint and its length prior are identified in
[the checkpoint manifest](../serving/checkpoint.json). This convenience
checkpoint uses Alexandria pretraining and MP20 fine-tuning; the paper's
reported experiments use MP20-only models. Checkpoints and
training datasets are distributed separately from Git.

The [paper](https://arxiv.org/abs/2603.06082) documents experimental settings and DFT processing.
The release does not include a self-contained rerun of all DFT calculations,
baselines, or full training. The unpublished Alexandria preprocessing/training
experiments are not presented as validated reproduction scripts.

## Environments

`requirements.txt` now installs the tested **CPU inference** stack. Install it
in its own Python 3.11 environment. It does not install the research oracles.

[requirements/legacy-research.txt](../requirements/legacy-research.txt) preserves
the previous research dependency list for provenance. **Do not use it as a
working lock file.** It combines incompatible TensorFlow 2.15/NumPy 2 pins,
contains a CUDA-specific PyTorch build without its package index, and omits
`google-cloud-storage`. The older MatGL/DGL and pymatgen versions also require
a compatible Python/NumPy/PyTorch environment.

The research scripts additionally need `absl-py`, `ml-collections`, Weights &
Biases, Google Cloud Storage, ASE, plotting tools, and the MatGL property/force
models used by the paper. `data/tools.py` still imports the research stack and
loads the MEGNet band-gap model at module import. Keep this environment separate
from lightweight inference. The cleanup removes unused TensorFlow, legacy M3GNet, and Matbench
imports, but does not certify a newly resolved full research environment.

## Data and storage

Edit `storage.bucket` in your selected configuration to your own Google Cloud
Storage bucket. Authenticate Google Cloud and Weights & Biases using their
normal local credential mechanisms; do not put credentials in the repository.
The scripts expect objects under these bucket paths:

```text
CliqueFlowmer/data/preprocessed/<task>/train.pickle
CliqueFlowmer/data/preprocessed/<task>/val.pickle
CliqueFlowmer/data/preprocessed/<task>/test.pickle
CliqueFlowmer/models/states/<model-class>/<task>/<model-spec>.pth
```

Each data split is a pickled mapping whose first two values are the ordered
structure and target lists. Use trusted research artifacts only. The exact
model specification is constructed from the configuration's model fields, so
training and optimization must use matching configurations and checkpoints.
Data preparation, target normalization, and oracle versions must match the
experiment being reproduced.

## Entrypoints

Once the research environment, data, and storage are prepared, the original
CliqueFlowmer entrypoints are:

```bash
torchrun --standalone --nproc_per_node=8 train.py --config=configs/mp20/cliqueflowmer.py
python optimize.py --config=configs/mp20/cliqueflowmer.py
```

Training uses NCCL and requires the requested number of CUDA GPUs. Choose the
band-gap configuration at `configs/mp20-bandgap/cliqueflowmer.py` for that task.
Optimization now stops with a clear error if its configured checkpoint is
missing, rather than continuing with random weights.

Transformer baseline configurations are retained as historical experiment
artifacts. They need reconciliation with the current entrypoints before reuse
(the learner name and optional model fields are not aligned). They are not a
validated command path in this release.

## Verification boundaries

The release checks cover tensor-helper parity, lightweight import boundaries,
inference and decoder behavior, checkpoint verification, and latent optimization.
A manually triggered CI run also exercises encoding and latent optimization
with the actual published checkpoint. Full distributed training, external
oracle evaluation, and DFT calculations were not rerun during repository cleanup.
