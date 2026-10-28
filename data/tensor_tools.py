"""Tensor preprocessing shared by training models and lightweight inference.

These helpers preserve the implementations in :mod:`data.tools`, without
importing its visualization, storage, or relaxation dependencies.
"""

from copy import deepcopy

import numpy as np
import torch

from data.constants import atomic_numbers


def move_to_device(data, device):
    return tuple(d.to(device) for d in data)


def pad_sequences_fast(sequences, pad_value=atomic_numbers['end']):

    #
    # Convert sequences to tensors first
    #
    if isinstance(sequences[0][0], (int, float)):
        #
        # For scalar values (species)
        #
        tensors = [torch.tensor(seq, dtype=torch.float) for seq in sequences]

    elif isinstance(sequences[0], np.ndarray):
        #
        # For vectors (positions) - Stack arrays first, then convert to tensor
        #
        tensors = [torch.from_numpy(np.array(seq, dtype=np.float32)) for seq in sequences]

    elif isinstance(sequences[0][0], np.ndarray):
        #
        # For when we have a list of lists of positions
        #
        tensors = [torch.stack([torch.from_numpy(s) for s in seq], dim=0) for seq in sequences]

    else:
        #
        # For a list of tensors
        #
        tensors = deepcopy(sequences)


    #
    # Get lengths for mask creation
    #
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = lengths.max()

    #
    # Use PyTorch's built-in padding
    #
    padded = torch.nn.utils.rnn.pad_sequence(
        tensors,
        batch_first=True,
        padding_value=pad_value
    )

    #
    # Create mask using arange - more efficient than loops
    #
    batch_size = len(sequences)
    mask = lengths.unsqueeze(1) > torch.arange(max_len).unsqueeze(0)

    return padded, mask.float()


def _metric_from_lattice(abc, angles):
    alpha, beta, gamma = angles.unbind(-1)
    a, b, c = abc.unbind(-1)

    ca, cb, cg = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    G11 = a*a;   G22 = b*b;   G33 = c*c
    G12 = a*b*cg; G13 = a*c*cb; G23 = b*c*ca

    G = torch.stack([
        torch.stack([G11, G12, G13], dim=-1),
        torch.stack([G12, G22, G23], dim=-1),
        torch.stack([G13, G23, G33], dim=-1),
    ], dim=-2)
    return G


def _min_image_exact(diff, G):

    o = diff.new_tensor([-1., 0., 1.])
    offsets = torch.stack(torch.meshgrid(o, o, o, indexing="ij"), dim=-1).reshape(-1, 3)

    cand = diff.unsqueeze(-2) + offsets.view(1, 1, 1, 27, 3)

    #
    # u^T G via einsum keeps batch as a single B:
    #
    Gu = torch.einsum('bmntk,bkl->bmntl', cand, G)
    sq = (cand * Gu).sum(dim=-1)

    idx = sq.argmin(dim=-1, keepdim=True)
    best = cand.gather(dim=-2, index=idx.unsqueeze(-1).expand(-1, -1, -1, 1, 3)).squeeze(-2)
    return best


def compute_pairwise_distances(abc, angles, positions, periodic=True, exact_min_image=True):

    B, N, _ = positions.shape

    #
    # (optional) normalize shapes defensively
    #
    abc = abc.view(B, 3); angles = angles.view(B, 3)

    G = _metric_from_lattice(abc, angles)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)

    if periodic:
        if exact_min_image:
            diff = _min_image_exact(diff, G)
        else:
            diff = diff - torch.floor(diff + 0.5)

    #
    # Distance^2 = u^T G u  (no extra B thanks to einsum)
    #
    Gu = torch.einsum('bmnk,bkl->bmnl', diff, G)
    sq = (diff * Gu).sum(dim=-1).clamp_min_(0.0)
    return torch.sqrt(sq)


def causal_mask(seq):

    shape = seq.shape
    seq_len = shape[-1]

    r = torch.arange(seq_len)
    mask = 1. * (r.unsqueeze(1) >= r.unsqueeze(0))

    return mask.view((len(shape) - 1) * (1,) + mask.shape)
