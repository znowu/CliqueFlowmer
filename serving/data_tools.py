"""Structure conversion utilities without optional relaxation dependencies."""

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
import random

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from data.constants import atomic_numbers


def move_to_device(data, device):
    return tuple(d.to(device) for d in data)


def pad_sequences_fast(sequences, pad_value=atomic_numbers["end"]):
    """
    Minimal extraction of data.tools.pad_sequences_fast.
    Preserves:
    - scalar species handling
    - vector position handling
    - float mask with 1 for valid tokens, 0 for padded positions
    """
    if isinstance(sequences[0][0], (int, float)):
        tensors = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    elif isinstance(sequences[0], np.ndarray):
        tensors = [torch.from_numpy(np.array(seq, dtype=np.float32)) for seq in sequences]
    elif isinstance(sequences[0][0], np.ndarray):
        tensors = [torch.stack([torch.from_numpy(s) for s in seq], dim=0) for seq in sequences]
    else:
        tensors = deepcopy(sequences)

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max())

    padded = torch.nn.utils.rnn.pad_sequence(
        tensors,
        batch_first=True,
        padding_value=pad_value,
    )

    mask = lengths.unsqueeze(1) > torch.arange(max_len).unsqueeze(0)
    return padded, mask.float()


def block_shuffle_by_species(struct):
    """
    Exact serving-side extraction of the grouping logic in data.tools.
    Only needed if you want shuffle=True.
    """
    species_to_indices = defaultdict(list)
    for idx, site in enumerate(struct.sites):
        species_to_indices[site.species.elements[0].Z].append(idx)

    species_list = list(species_to_indices.keys())
    random.shuffle(species_list)

    permuted_indices = []
    for species in species_list:
        indices = species_to_indices[species]
        random.shuffle(indices)
        permuted_indices.extend(indices)

    return permuted_indices


def unpack_structures(structures, shuffle=False):
    """
    Minimal serving extraction of data.tools.unpack_structures.

    Important:
    - returns lattice angles in radians, matching your real implementation
    - uses start/end tokens exactly the same way
    - returns float mask with 1 for valid positions
    """
    batch_size = len(structures)

    abc = torch.zeros((batch_size, 3))
    angles = torch.zeros((batch_size, 3))

    species_list = []
    positions_list = []

    for i, struct in enumerate(structures):
        abc[i] = torch.tensor(struct.lattice.abc, dtype=torch.float32)
        angles[i] = torch.tensor(struct.lattice.angles, dtype=torch.float32)

        n_sites = len(struct.sites)
        perm = block_shuffle_by_species(struct) if shuffle else range(n_sites)

        species_list.append(
            [atomic_numbers["start"]]
            + [struct.sites[j].species.elements[0].Z for j in perm]
            + [atomic_numbers["end"]]
        )

        positions_list.append(
            [np.zeros((3,), dtype=np.float32)]
            + [np.asarray(struct.sites[j].frac_coords, dtype=np.float32) for j in perm]
            + [np.zeros((3,), dtype=np.float32)]
        )

    species, mask = pad_sequences_fast(species_list)
    positions, _ = pad_sequences_fast(positions_list)

    return (
        abc.float(),
        torch.deg2rad(angles).float(),
        species.long(),
        positions.float(),
        mask.float(),
    )


def tensors_to_structure(abc, angles, atomic, pos, mask, min_dist=0.5):
    """
    Lightweight decoder postprocessing for serving.

    Assumes:
    - abc: [B, 3]
    - angles: [B, 3] in radians
    - atomic includes start/end tokens
    - pos are fractional coordinates
    - mask uses 1 for valid entries, 0 for padding
    """
    abc = abc.detach().cpu().numpy()
    angles = angles.detach().cpu().numpy()
    atomic = atomic.detach().cpu().numpy()
    pos = pos.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()

    start_tok = atomic_numbers["start"]
    end_tok = atomic_numbers["end"]

    structures = []

    for i in range(len(abc)):
        valid = mask[i] > 0.5
        atomic_i = atomic[i][valid]
        pos_i = pos[i][valid]

        species = []
        coords = []

        for z, frac in zip(atomic_i, pos_i):
            z = int(z)
            if z == start_tok or z == end_tok:
                continue
            species.append(z)
            coords.append(frac.tolist())

        if len(species) == 0:
            continue

        try:
            lattice = Lattice.from_parameters(
                a=float(abc[i][0]),
                b=float(abc[i][1]),
                c=float(abc[i][2]),
                alpha=float(np.rad2deg(angles[i][0])),
                beta=float(np.rad2deg(angles[i][1])),
                gamma=float(np.rad2deg(angles[i][2])),
            )
            structure = Structure(
                lattice=lattice,
                species=species,
                coords=coords,
                coords_are_cartesian=False,
            )

            if len(structure) >= 2:
                dmat = structure.distance_matrix
                bad = False
                for r in range(len(structure)):
                    for c in range(r + 1, len(structure)):
                        if dmat[r, c] < min_dist:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue

            structures.append(structure)
        except Exception:
            continue

    return structures


def refine_to_primitive_fast_strong(
    struct: Structure,
    *,
    target_fmax: float = 0.02,
    max_steps_pos: int = 1100,
    max_steps_cell: int = 350,
    do_cell_relax: bool = True,
    cell_trigger_force: float = 0.12,
    cell_trigger_strain: float = 0.08,
    min_dist: float = 0.5,
    calculator=None,
) -> Structure:
    """Run optional M3GNet relaxation without importing its stack for inference."""
    from serving.relaxation import refine_to_primitive_fast_strong as refine

    return refine(
        struct,
        target_fmax=target_fmax,
        max_steps_pos=max_steps_pos,
        max_steps_cell=max_steps_cell,
        do_cell_relax=do_cell_relax,
        cell_trigger_force=cell_trigger_force,
        cell_trigger_strain=cell_trigger_strain,
        min_dist=min_dist,
        calculator=calculator,
    )


def normal_lengths_from_mle(length_mle_vals, device, independent=True):

    mu_a, sigma_a = length_mle_vals.get("a")
    mu_b, sigma_b = length_mle_vals.get("b")
    mu_c, sigma_c = length_mle_vals.get("c")

    cov = length_mle_vals.get("cov", None)
    mu = torch.tensor([mu_a, mu_b, mu_c], device=device).float()

    if independent:
        Sigma = torch.diag(
            torch.tensor([sigma_a, sigma_b, sigma_c], device=device).float()
        )**2 
    else:
        Sigma = torch.from_numpy(cov).float().to(device)
        Sigma = (Sigma + Sigma.T) / 2

    mvn = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
    return mvn