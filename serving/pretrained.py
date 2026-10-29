"""Load the pinned public formation-energy checkpoint for standalone inference.

This convenience checkpoint uses Alexandria pretraining and MP20 fine-tuning;
it is distinct from the paper's MP20-only experiment checkpoints. Its latent
predictor reports raw model scores, not calibrated formation energies in eV/atom.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from serving.data_tools import normal_lengths_from_mle
from serving.inference import load_model_from_checkpoint


HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "checkpoint.json").read_text())


def load_pretrained_model(checkpoint_path=None, device="cpu"):
    """Return the published checkpoint in evaluation mode on ``device``.

    An explicit path takes precedence over ``CLIQUEFLOWMER_CHECKPOINT``. If
    neither is set, download the manifest's exact Hugging Face revision. Every
    path, including local overrides, must match the published SHA-256 checksum.
    The matching length prior is bundled as JSON, so no prior download or
    unpickling is required. PyTorch's thread settings are left to the caller.
    """
    path = checkpoint_path
    if path is None:
        path = os.environ.get("CLIQUEFLOWMER_CHECKPOINT") or None
    if path is None:
        path = hf_hub_download(
            repo_id=MANIFEST["repo_id"],
            filename=MANIFEST["filename"],
            revision=MANIFEST["revision"],
        )
    path = Path(path)

    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MANIFEST["sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match the published checkpoint.")

    prior = json.loads((HERE / "length_prior.json").read_text())
    return load_model_from_checkpoint(
        path,
        model_kwargs={"initial_length_dist": normal_lengths_from_mle(prior, device)},
        device=device,
    )
