from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pickle
import numpy as np
import torch

from huggingface_hub import hf_hub_download
from pymatgen.core import Structure

from models import CliqueFlowmer
from models.graphops import separate_latents
from models.tools import report_progress
from optimization.design import Design
from optimization.learner import ES
import serving.data_tools as data_tools


DEFAULT_MODEL_KWARGS = {
    "n_cliques": 8,
    "clique_dim": 16,
    "knot_dim": 1,
    "transformer_dim": 256,
    "n_registers": 2,
    "mlp_dim": 128,
    "n_mlp": 2,
    "n_blocks": 4,
    "n_heads": 4,
    "dropout_rate": 0.1,
    "alpha_vae": 1e-4,
    "alpha_mse": 1.0,
    "beta_mse": 1e-4,
    "warmup": 1e5,
    "temp_atom": 1.0,
    "temp_flow": 16.0,
    "temp_distance": 0.0,
    "drop_type": 0.0,
    "drop_latent": 0.1,
    "lr": 1.4e-4,
}


def _device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _resolve_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    return obj


def build_model(
    model_kwargs: Optional[dict] = None,
    device: Optional[str] = None,
) -> CliqueFlowmer:
    kwargs = dict(DEFAULT_MODEL_KWARGS)
    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    model = CliqueFlowmer(**kwargs)
    model.eval()
    model.to(_device(device))
    return model


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_kwargs: Optional[dict] = None,
    device: Optional[str] = None,
    strict: bool = True,
) -> CliqueFlowmer:
    dev = _device(device)
    model = build_model(model_kwargs=model_kwargs, device=str(dev))

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _strip_module_prefix(_resolve_state_dict(raw))
    model.load_state_dict(state_dict, strict=strict)

    model.eval()
    model.to(dev)
    return model


def load_model_from_hf(
    repo_id: str,
    filename: str,
    model_kwargs: Optional[dict] = None,
    device: Optional[str] = None,
    strict: bool = True,
    length_mle="Length_MLE_Offset_True_MP20.pickle",
    revision: Optional[str] = None,
) -> CliqueFlowmer:

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    length_mle_path = hf_hub_download(repo_id=repo_id, filename=length_mle, revision=revision)

    with open(length_mle_path, 'rb') as f:
        length_mle = pickle.load(f)
    
    kwargs = dict(model_kwargs or {})
    kwargs["initial_length_dist"] = data_tools.normal_lengths_from_mle(length_mle, _device(device))


    return load_model_from_checkpoint(
        checkpoint_path=ckpt_path,
        model_kwargs=kwargs,
        device=device,
        strict=strict,
    )


def structure_from_cif_text(cif_text: str) -> Structure:
    return Structure.from_str(cif_text, fmt="cif")


def structure_to_cif_text(structure: Structure) -> str:
    return structure.to(fmt="cif")


def _structures_to_batch(structures, device=None):
    abc, angles, atomic, pos, mask = data_tools.unpack_structures(
        list(structures),
        shuffle=False,
    )
    return data_tools.move_to_device((abc, angles, atomic, pos, mask), _device(device))


@torch.no_grad()
def encode_structure(
    model: CliqueFlowmer,
    structure: Structure,
    separate: bool = False,
) -> torch.Tensor:
    dev = next(model.parameters()).device
    abc, angles, atomic, pos, mask = _structures_to_batch([structure], device=str(dev))
    atomic_emb = model.atomic_emb(atomic)
    z = model.encode(abc, angles, atomic_emb, pos, mask, separate=separate)
    return z[0]


@torch.no_grad()
def decode_latent(
    model: CliqueFlowmer,
    z: Union[np.ndarray, torch.Tensor],
    integration: str = "cfg",
    min_dist: float = 0.5,
    relax: bool = True,
    *,
    n_steps: Optional[int] = None,
    beam_width: int = 10,
    min_atoms: int = 3,
    max_atoms: int = 100,
    progress_callback=None,
    cancel_check=None,
) -> Structure:
    """Decode one latent; callbacks receive (completed_work, total_work).

    ``cancel_check`` takes no arguments and raises to abort. Relaxation remains
    optional and is separate from the bounded neural decoding progress.
    """
    dev = next(model.parameters()).device

    if isinstance(z, np.ndarray):
        z = torch.tensor(z, dtype=torch.float32, device=dev)
    else:
        z = z.to(device=dev, dtype=torch.float32)

    if z.ndim == 1:
        z = z.unsqueeze(0)
    if z.ndim != 2 or z.shape != (1, model.latent_dim) or not torch.isfinite(z).all():
        raise ValueError(f"Expected one finite {model.latent_dim}-dimensional latent vector.")
    if not np.isfinite(min_dist) or min_dist < 0:
        raise ValueError("min_dist must be finite and nonnegative.")

    abc, angles, atomic, pos, mask = model.decode(
        z, integration=integration, n_steps=n_steps, beam_width=beam_width,
        min_atoms=min_atoms, max_atoms=max_atoms,
        progress_callback=progress_callback, cancel_check=cancel_check,
    )
    if cancel_check is not None:
        cancel_check()
    structures = data_tools.tensors_to_structure(
        abc, angles, atomic, pos, mask, min_dist=min_dist
    )
    if not structures:
        raise ValueError("No valid structure produced by decoder.")

    structure = structures[0]

    if relax:
        structure = data_tools.refine_to_primitive_fast_strong(structure)
    if cancel_check is not None:
        cancel_check()

    return structure


def reconstruct_structure(
    model: CliqueFlowmer,
    structure: Structure,
    integration: str = "cfg",
    min_dist: float = 0.5,
    relax=True,
    **decode_kwargs,
):
    z = encode_structure(model, structure, separate=False)
    recon = decode_latent(
        model=model,
        z=z,
        integration=integration,
        min_dist=min_dist,
        relax=relax,
        **decode_kwargs,
    )
    return z, recon


def latent_to_text(z: Union[np.ndarray, torch.Tensor], precision: int = 6) -> str:
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    arr = np.asarray(z).reshape(-1)
    return ",".join(f"{x:.{precision}g}" for x in arr.tolist())


def text_to_latent(latent_text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in latent_text.split(",") if x.strip()]
    if not vals:
        raise ValueError("No latent values found.")
    return np.asarray(vals, dtype=np.float32)


@torch.no_grad()
def optimize_es(model, z, steps=1000, weight_decay=0.4, *,
                progress_callback=None, cancel_check=None, return_history=False):
    """Minimize the checkpoint's latent objective without modifying ``z``.

    Returns ``(latent, score)`` by default. With ``return_history=True``, returns
    ``(latent, score, history)``; history contains ``step`` and ``objective`` at
    step zero, every ten steps and the final step. Scores are predictor outputs,
    whose physical meaning and any normalization depend on the checkpoint.
    Callbacks receive ``(step, steps)``; ``cancel_check()`` raises to cancel.
    """
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    dev = next(model.parameters()).device
    z = z.detach().to(device=dev, dtype=torch.float32).clone()
    if z.ndim not in (1, 2) or z.numel() == 0 or z.shape[-1] != model.latent_dim or not torch.isfinite(z).all():
        raise ValueError(f"Expected finite {model.latent_dim}-dimensional latents.")
    index_matrix = model.index_matrix.to(dev)

    struct_fn = lambda x: separate_latents(x, index_matrix)

    kwargs = {
        "structure_fn": struct_fn,
        "design_steps": steps,
        "decay": weight_decay,
        "lr": 3e-4,
        "n_pert": 20,
        "scale_pert": 0.05,
        "antithetic": True,
        "rank": True
    }

    pred_fn = model.target_regressor
    design = Design(z.view(-1, z.shape[-1]))

    learner = ES(design, pred_fn, **kwargs)

    history = []

    def record(step):
        value = learner.value().detach()
        if not torch.isfinite(value).all():
            raise ValueError("The latent predictor produced a nonfinite objective.")
        if return_history:
            history.append({"step": step, "objective": float(value.cpu())})
        return value

    report_progress(0, steps, progress_callback, cancel_check)
    score = record(0)
    for step in range(1, steps + 1):
        learner.train_step()
        if step % 10 == 0 or step == steps:
            score = record(step)
        report_progress(step, steps, progress_callback, cancel_check)

    z_new = learner.design_fn().detach().clone()
    if return_history:
        return z_new, score, history
    return z_new, score
