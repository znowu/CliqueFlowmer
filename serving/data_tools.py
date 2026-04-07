# serving/tools_light.py

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
import random

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

import matgl
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.ext.ase import M3GNetCalculator
from ase.optimize import FIRE, LBFGS, LBFGSLineSearch
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


def _load_m3gnet_calculator(*, compute_stress: bool = False) -> M3GNetCalculator:
    #
    # Load a PES potential that provides energies, forces (and stress if requested)
    #
    potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    return M3GNetCalculator(potential=potential, compute_stress=compute_stress)


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

    # 
    # Internal calculator cache
    # 
    if not hasattr(refine_to_primitive_fast_strong, "_calc_pos"):
        refine_to_primitive_fast_strong._calc_pos = None
    if not hasattr(refine_to_primitive_fast_strong, "_calc_cell"):
        refine_to_primitive_fast_strong._calc_cell = None

    def _get_calc_pos():
        if calculator is not None:
            return calculator
        if refine_to_primitive_fast_strong._calc_pos is None:
            refine_to_primitive_fast_strong._calc_pos = _load_m3gnet_calculator(compute_stress=False)
        return refine_to_primitive_fast_strong._calc_pos

    def _get_calc_cell():
        if refine_to_primitive_fast_strong._calc_cell is None:
            refine_to_primitive_fast_strong._calc_cell = _load_m3gnet_calculator(compute_stress=True)
        return refine_to_primitive_fast_strong._calc_cell

    # 
    # Optional fast validity helpers (safe import)
    # 
    _sv = _cv = None
    try:
        from optimization.sun import structurally_valid as _sv, compositionally_valid as _cv
    except Exception:
        try:
            # if you have them elsewhere in your repo
            from data.validation import structurally_valid as _sv, compositionally_valid as _cv
        except Exception:
            _sv = _cv = None

    # 
    # Helpers
    # 
    def _max_force(atoms) -> float:
        f = atoms.get_forces()
        return float(np.max(np.linalg.norm(f, axis=1)))

    def _cell_is_suspicious(pm_struct) -> bool:
        try:
            s_prim = _safe_to_primitive(pm_struct)
            a = float(pm_struct.lattice.volume)
            b = float(s_prim.lattice.volume)
            if a <= 0.0 or b <= 0.0:
                return True
            return (abs(a - b) / b) > cell_trigger_strain
        except Exception:
            return True

    # 
    # 0) Canonicalize early
    # 
    s0 = struct.get_primitive_structure()

    # 
    # 1) ASE atoms + positions-only calculator
    # 
    atoms = AseAtomsAdaptor.get_atoms(s0)
    atoms.calc = _get_calc_pos()

    # 
    # 2) Pre-sanitize
    # 
    _puff_if_clashing(atoms, relax_cell=False)
    if _has_nan_energy_or_forces(atoms):
        pos = atoms.get_positions()
        cell = atoms.cell
        frac = np.linalg.solve(cell.T, pos.T).T
        frac = _deterministic_nudge_frac(frac, eps=2e-4)
        atoms.set_positions(np.dot(frac, cell))

    # 
    # 2.5) Fast bail-out (only if helpers exist)
    # 
    if _sv is not None and _cv is not None:
        pm_now = AseAtomsAdaptor.get_structure(atoms)
        if (not _sv(pm_now, min_dist=min_dist)) or (not _cv(pm_now)):
            return _safe_to_primitive(pm_now)

    # 
    # 3) Positions-only relax: FIRE settle -> LBFGS finish
    # 
    n_fire0 = min(220, max_steps_pos)
    n_fire1 = min(220, max(0, max_steps_pos - n_fire0))
    n_lbfgs = max(0, max_steps_pos - n_fire0 - n_fire1)

    if n_fire0:
        FIRE(atoms, maxmove=0.06, logfile=None).run(fmax=0.06, steps=n_fire0)
    if n_fire1:
        FIRE(atoms, maxmove=0.035, logfile=None).run(fmax=0.03, steps=n_fire1)

    try:
        mf = _max_force(atoms)
    except Exception:
        mf = float("inf")

    if mf > target_fmax and n_lbfgs:
        LBFGS(atoms, maxstep=0.04, logfile=None).run(fmax=target_fmax, steps=n_lbfgs)

    # 
    # 4) Rare, bounded cell relax (gated)
    # 
    if do_cell_relax:
        try:
            mf2 = _max_force(atoms)
        except Exception:
            mf2 = float("inf")

        pm_mid = AseAtomsAdaptor.get_structure(atoms)
        suspicious = _cell_is_suspicious(pm_mid)

        if (mf2 > cell_trigger_force) or suspicious:
            atoms.calc = _get_calc_cell()
            obj = UnitCellFilter(atoms)

            n_fire_c = min(160, max_steps_cell)
            n_lbfgs_c = max(0, max_steps_cell - n_fire_c)

            if n_fire_c:
                FIRE(obj, maxmove=0.04, logfile=None).run(fmax=max(3.0 * target_fmax, 0.03), steps=n_fire_c)
            if n_lbfgs_c:
                LBFGS(obj, maxstep=0.04, logfile=None).run(fmax=target_fmax, steps=n_lbfgs_c)

    # 
    # 5) Return primitive
    # 
    relaxed = AseAtomsAdaptor.get_structure(atoms)
    return _safe_to_primitive(relaxed)
