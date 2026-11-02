# Research setup and reproducibility scope

This page describes the paper's research scripts on `main`, including their
dependency, data, and evaluation requirements. Standalone Python inference
has its own setup on the
[`inference` branch](https://github.com/znowu/CliqueFlowmer/tree/inference).

## What is included

The reference CliqueFlowmer and transformer architectures, ES and gradient
learners, MP20 configurations, distributed training, structure conversion,
relaxation helpers, property evaluation, and S.U.N. utilities are included.
The paper's reported experiments use MP20-only models. The
[released model weights](https://huggingface.co/iamkuba/CliqueFlowmer) also
include a separate Alexandria-pretrained, MP20-fine-tuned convenience model;
that model is not a replacement for the paper's experimental checkpoints.
Checkpoints and training datasets are distributed separately from Git.

The [paper](https://arxiv.org/abs/2603.06082) documents experimental settings and DFT processing.
The release does not include a self-contained rerun of all DFT calculations,
baselines, or full training. The unpublished Alexandria preprocessing/training
experiments are not presented as validated reproduction scripts.

## Environments

`requirements.txt` installs only the **CPU core-model dependencies** (PyTorch
and NumPy); `requirements-dev.txt` adds the test runner. These support the
core checks described below, using Python 3.11 on Linux. They do not install
the distributed-training tools, structure libraries, or property models
needed by `train.py` and `optimize.py`.

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
from the core-check environment. Unused TensorFlow, legacy M3GNet, and Matbench
imports have been removed; a newly resolved full research environment has not
been certified.

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

Checks on `main` cover tensor-helper parity with the research implementation,
periodic distances, padding and masks, and core model import boundaries. They
do not load checkpoints or evaluate material properties. Standalone inference
and checkpoint tests remain on the `inference` branch. Full distributed
training, external oracle evaluation, and DFT calculations were not rerun
during repository cleanup.
