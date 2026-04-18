from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pickle
import numpy as np
import torch


from huggingface_hub import hf_hub_download
from pymatgen.core import Structure

from models import CliqueFlowmer
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

    raw = torch.load(checkpoint_path, map_location="cpu")
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
    length_mle = "Length_MLE_Offset_True_MP20.pickle"
) -> CliqueFlowmer:

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    length_mle_path = hf_hub_download(repo_id=repo_id, filename=length_mle)

    with open(length_mle_path, 'rb') as f:
        length_mle = pickle.load(f)
    
    model["initial_length_dist"] = data_tools.normal_lengths_from_mle(length_mle, device)


    return load_model_from_checkpoint(
        checkpoint_path=ckpt_path,
        model_kwargs=model_kwargs,
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


def decode_latent(
    model: CliqueFlowmer,
    z: Union[np.ndarray, torch.Tensor],
    integration: str = "cfg",
    min_dist: float = 0.5,
    relax: bool = True
) -> Structure:
    dev = next(model.parameters()).device

    if isinstance(z, np.ndarray):
        z = torch.tensor(z, dtype=torch.float32, device=dev)
    else:
        z = z.to(device=dev, dtype=torch.float32)

    if z.ndim == 1:
        z = z.unsqueeze(0)

    abc, angles, atomic, pos, mask = model.decode(z, integration=integration)
    structures = data_tools.tensors_to_structure(
        abc, angles, atomic, pos, mask, min_dist=min_dist
    )
    if not structures:
        raise ValueError("No valid structure produced by decoder.")

    structure = structures[0]

    if relax:
        structure = data_tools.refine_to_primitive_fast_strong(structure)

    return structures[0]


def reconstruct_structure(
    model: CliqueFlowmer,
    structure: Structure,
    integration: str = "cfg",
    min_dist: float = 0.5,
    relax=True
):
    z = encode_structure(model, structure, separate=False)
    recon = decode_latent(
        model=model,
        z=z,
        integration=integration,
        min_dist=min_dist,
        relax=relax
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