import os
import io 
from io import BytesIO
from copy import deepcopy

import math 
import random 
import numpy as np 
import pandas as pd 
from numpy.random import permutation
from collections import defaultdict

import torch 
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ase.io import write

from PIL import Image
import pickle
from google.cloud import storage

from data.constants import atomic_numbers, atomic_colors

import py3Dmol
from pymatgen.core import Structure, Lattice, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

from ase.visualize.plot import plot_atoms
from pymatgen.io.xyz import XYZ

from glob import glob

from concurrent.futures import ProcessPoolExecutor

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import spglib
from typing import Tuple, Optional, Any, Callable, Union, Iterable


#
# Refine a primitive Structure with M3GNet (ASE) and return a primitive Structure
#
import matgl
from matgl.ext.ase import M3GNetCalculator
from ase.optimize import FIRE, LBFGS, LBFGSLineSearch
from ase.optimize.precon import PreconLBFGS, Exp
from ase.constraints import UnitCellFilter
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms

from ase.data import atomic_numbers as ase_atomic_numbers
from ase.data.colors import jmol_colors
from ase.data import covalent_radii

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import to_rgba

from matgl import load_model
import matplotlib.patheffects as pe




def move_to_device(data, device):
    return tuple(d.to(device) for d in data)


def pad_sequences(sequences, pad_value=atomic_numbers['end']):
    #
    # Find maximum length
    #
    max_len = max(len(lst) for lst in sequences)
    batch_size = len(sequences)
    
    #
    # Get the shape of individual elements to determine padding dimensions
    #
    sample_shape = np.array(sequences[0][0]).shape
    
    #
    # Initialize tensors with appropriate shapes
    #
    if len(sample_shape) == 0:  # scalar values (for species)
        padded = torch.zeros(batch_size, max_len)
    else:  # vectors (for positions)
        padded = torch.zeros(batch_size, max_len, *sample_shape)
    
    mask = torch.ones((batch_size, max_len), dtype=torch.float)  # 1 means padded
    
    #
    # Fill in actual values and mask
    #
    for i, seq in enumerate(sequences):
        length = len(seq)
        padded[i, :length] = torch.tensor(seq)
        mask[i, :length] = 0.
    
    return padded, mask


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


def block_shuffle_by_species(struct):
    #
    # Map from atomic number to list of site indices
    #
    species_to_indices = defaultdict(list)
    for idx, site in enumerate(struct.sites):
        species_to_indices[site.species.elements[0].Z].append(idx)

    #
    # Randomly shuffle the species block order
    #
    species_list = list(species_to_indices.keys())
    random.shuffle(species_list)

    #
    # Concatenate site indices from each species block
    #
    permuted_indices = []
    for species in species_list:
        indices = species_to_indices[species]
        random.shuffle(indices)  # optional: shuffle within block
        permuted_indices.extend(indices)

    return permuted_indices



def unpack_structures(structures, shuffle=True):

    batch_size = len(structures)
    
    #
    # Get lattice parameters and angles
    #
    abc = torch.zeros((batch_size, 3))  # a, b, c lattice parameters
    angles = torch.zeros((batch_size, 3))  # alpha, beta, gamma angles
    volume = torch.zeros(batch_size)
    density = torch.zeros(batch_size)
    
    #
    # Prepare lists for padding
    #
    species_list = []
    positions_list = []
    
    for i, struct in enumerate(structures):
        #
        # Get lattice parameters and angles
        #
        abc[i] = torch.tensor(struct.lattice.abc)
        angles[i] = torch.tensor(struct.lattice.angles)
        
        #
        # Generate random permutation for sites
        #
        n_sites = len(struct.sites)
        perm = block_shuffle_by_species(struct) if shuffle else range(n_sites)

        #
        # Collect atomic numbers and positions
        #
        species_list.append(
            [atomic_numbers['start']] +\
            [struct.sites[j].species.elements[0].Z for j in perm] +\
            [atomic_numbers['end']]
            )
        positions_list.append(
            [np.zeros((3,))] +\
            [struct.sites[j].frac_coords for j in perm] +\
            [np.zeros((3,))]
            )
    
    #
    # Pad species and positions
    #
    species, mask = pad_sequences_fast(species_list)  # Will be (batch_size, max_atoms)
    positions, _ = pad_sequences_fast(positions_list)  # Will be (batch_size, max_atoms, 3)
    
    return abc.float(), torch.deg2rad(angles).float(), species.long(), positions.float(), mask.float() 


#
# Computes the matrix of this lattice's distance metric
#
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


#
# Computes distance in a periodic lattice
#
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



def _pmg_to_spglib_cell(struct: Structure) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    lattice = np.ascontiguousarray(struct.lattice.matrix, dtype="double")     
    positions = np.ascontiguousarray(struct.frac_coords, dtype="double")      
    
    #
    # Use integer atomic numbers with dtype 'intc' (what spglib prefers).
    #
    numbers = np.ascontiguousarray([site.species.elements[0].Z for site in struct.sites], dtype="intc")
    return lattice, positions, numbers


def _wrap01(x):
    x = np.mod(x, 1.0)
    # snap 1.0-ε → 0.0 to avoid 0/1 duplicates from fp noise
    x[x >= 1.0 - 1e-12] -= 1.0
    return x


class StructureAugmenter:
    def __init__(
        self,
        symprec: float = 1e-3,
        angle_tolerance: float = -1.0,
        # Augment options (conservative defaults)
        use_symmetry_ops: bool = True,
        use_fractional_jitter: bool = True,
        jitter_sigma: float = 0.003,
        jitter_clip: float = 0.01,
        min_distance_factor: float = 0.80,
    ):
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.use_symmetry_ops = use_symmetry_ops
        self.use_fractional_jitter = use_fractional_jitter
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.min_distance_factor = min_distance_factor

    def apply_random_spacegroup_op(
        self,
        struct: Structure,
        *,
        symmetrize_if_needed: bool = True,
        filter_identity: bool = True,
        retries: int = 1,
        sym_ops: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        return_op: bool = False,
        ensure_primitive_output: bool = False,   # NEW
    ) -> Union[Structure, Tuple[Structure, Tuple[np.ndarray, np.ndarray]]]:

        s = struct

        for attempt in range(retries + 1):
            if sym_ops is not None:
                R_all, t_all = sym_ops
                R_all = np.asarray(R_all, dtype=int)
                t_all = np.asarray(t_all, dtype=np.float64)
            else:
                cell = _pmg_to_spglib_cell(s)
                sym = spglib.get_symmetry(cell, symprec=self.symprec, angle_tolerance=self.angle_tolerance)
                R_all = np.asarray(sym.get("rotations", []), dtype=int)
                t_all = np.asarray(sym.get("translations", []), dtype=np.float64)

            if R_all.size > 0:
                if filter_identity:
                    I = np.eye(3, dtype=int)
                    keep_idx = []
                    for i, (R, t) in enumerate(zip(R_all, t_all)):
                        is_I = np.array_equal(R, I)
                        is_zero_t = np.allclose(_wrap01(t.copy()), 0.0, atol=1e-12)
                        if not (is_I and is_zero_t):
                            keep_idx.append(i)
                    if keep_idx:
                        R_all = R_all[keep_idx]
                        t_all = t_all[keep_idx]

                j = random.randrange(len(R_all))
                R, t = R_all[j], t_all[j]

                f_new = s.frac_coords @ R.T + t
                f_new = _wrap01(f_new)
                out = Structure(s.lattice, s.species, f_new)

                #
                # Make the returned structure primitive if requested
                #
                if ensure_primitive_output:
                    try:
                        out = SpacegroupAnalyzer(
                            out,
                            symprec=self.symprec,
                            angle_tolerance=self.angle_tolerance if self.angle_tolerance >= 0 else 5.0,
                        ).get_primitive_standard_structure()
                    except Exception:
                        out = out.get_primitive_structure()

                if return_op:
                    #
                    # Note: (R, t) refers to the pre-primitive lattice if conversion happened
                    #
                    return out, (R, t)
                return out

            if not symmetrize_if_needed or attempt == retries:
                out = Structure(s.lattice, s.species, s.frac_coords.copy())
                if ensure_primitive_output:
                    try:
                        out = SpacegroupAnalyzer(
                            out,
                            symprec=self.symprec,
                            angle_tolerance=self.angle_tolerance if self.angle_tolerance >= 0 else 5.0,
                        ).get_primitive_standard_structure()
                    except Exception:
                        out = out.get_primitive_structure()
                return out

            sga = SpacegroupAnalyzer(
                s,
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance if self.angle_tolerance >= 0 else 5.0,
            )
            try:
                s_ref = sga.get_refined_structure()
                s = SpacegroupAnalyzer(
                    s_ref,
                    symprec=self.symprec,
                    angle_tolerance=self.angle_tolerance if self.angle_tolerance >= 0 else 5.0,
                ).get_primitive_standard_structure()
            except Exception:
                pass

        out = Structure(struct.lattice, struct.species, struct.frac_coords.copy())
        if ensure_primitive_output:
            try:
                out = SpacegroupAnalyzer(
                    out,
                    symprec=self.symprec,
                    angle_tolerance=self.angle_tolerance if self.angle_tolerance >= 0 else 5.0,
                ).get_primitive_standard_structure()
            except Exception:
                out = out.get_primitive_structure()
        return out



    def fractional_jitter(self, struct: Structure):
        #
        # Add small noise to fractional coordinates with rejection if too close
        #
        noise = np.clip(np.random.normal(scale=self.jitter_sigma,
                                         size=struct.frac_coords.shape),
                        -self.jitter_clip, self.jitter_clip)
        coords = (struct.frac_coords + noise) % 1.0
        new_struct = Structure(lattice=struct.lattice,
                               species=struct.species,
                               coords=coords)

        #
        # Reject if atoms too close
        #
        D_old = struct.distance_matrix
        np.fill_diagonal(D_old, 1000000)

        D_new = new_struct.distance_matrix
        np.fill_diagonal(D_new, 1000000)

        dmin_old = D_old.reshape(-1).min()
        dmin_new = D_new.reshape(-1).min()
        if dmin_new < self.min_distance_factor * dmin_old:
            return struct  # fallback to original
        return new_struct

    def augment(self, struct: Structure, probs=None):
        """
        Apply a random sequence of augmentations.
        probs: dict with keys ['spacegroup', 'jitter']
        """
        if probs is None:
            probs = {'spacegroup': math.sqrt(0.9), 'jitter': math.sqrt(0.9)}

        s = Structure.from_dict(struct.as_dict())

        if 0 < probs['spacegroup'] and np.random.rand() < probs['spacegroup']:
            s = self.apply_random_spacegroup_op(s)
        if 0 < probs['jitter'] and np.random.rand() < probs['jitter']:
            s = self.fractional_jitter(s)

        return s

    def to_torch(self, struct: Structure):
        #
        # Convert to torch tensors for use in PyTorch models
        #
        lattice = torch.tensor(struct.lattice.matrix, dtype=torch.float32)
        coords = torch.tensor(struct.frac_coords, dtype=torch.float32)
        atomic_numbers = torch.tensor([sp.Z for sp in struct.species], dtype=torch.long)
        return lattice, coords, atomic_numbers


class MatbenchDataset(Dataset):
    def __init__(self, inputs, targets, augment=True, **kwargs):
        """
        Args:
            inputs: Raw inputs (e.g., list of pymatgen Structure objects)
            targets: Corresponding target values
        """

        if not isinstance(inputs, pd.Series) and not isinstance(inputs, pd.DataFrame):
            inputs = pd.Series(inputs)

        self.inputs = inputs
        self.targets = targets
        self.augment = augment

        if augment:
            self.augmenter = StructureAugmenter(**kwargs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs, targets = self.inputs.iloc[idx], self.targets[idx]

        if not self.augment or not hasattr(self, "augmenter"):
            return inputs, targets 
        
        return self.augmenter.augment(inputs, probs={'spacegroup': 0, 'jitter': 0.9}), targets 

    def shuffle(self):

        indices = np.random.permutation(len(self))
        
        self.inputs = self.inputs.iloc[indices].reset_index(drop=True)
        
        if isinstance(self.targets, np.ndarray) or isinstance(self.targets, torch.Tensor):
            self.targets = self.targets[indices]
        
        elif isinstance(self.targets, list):
            self.targets[:] = [self.targets[i] for i in indices]
        
        elif hasattr(self.targets, 'iloc'): 
            self.targets = self.targets.iloc[indices].reset_index(drop=True)
        else:
            raise TypeError("Unsupported type for targets. Must be numpy array, list, or pandas DataFrame.")


def collate_structure(batch):
    inputs, targets = zip(*batch)
    
    abc, angles, species, positions, mask = unpack_structures(inputs)
    targets_batch = torch.tensor(targets)
    
    return abc, angles, species, positions, mask, targets_batch


class InfiniteDataLoader:

    def __init__(self, dataset, batch_size):

        self.batch_size = batch_size

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_structure,
            shuffle=True,
            pin_memory=True,
            drop_last=True, 
        )
        self.data_iter = iter(self.data_loader)

    def __iter__(self):
        return self
    
    def __next__(self):
        
        try:
            batch = next(self.data_iter)
        
        except StopIteration:
            #
            # Reset iterator when dataset is exhausted
            #
            self.data_loader = torch.utils.data.DataLoader(
                self.data_loader.dataset,
                batch_size=self.batch_size,
                collate_fn=collate_structure,
                shuffle=True,
                pin_memory=True,
                drop_last=True
            )
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)

        return batch


def get_distributed_loader(dataset, batch_size=64, num_workers=8, 
                          shuffle=True, pin_memory=True, drop_last=True,
                          seed=None, prefetch_factor=2):
    #
    # Get distributed context
    #
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    #
    # Create distributed sampler with optional seed
    #
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed or 0  # Default to 0 if None
    )

    #
    # Worker initialization function for reproducibility
    #
    def seed_worker(worker_id):
        worker_seed = seed + worker_id if seed is not None else None
        if worker_seed is not None:
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

    #
    # Calculate per-device batch size and workers
    #
    per_device_batch = max(1, batch_size // world_size)
    per_device_workers = max(0, num_workers // world_size)  # Allow 0 workers

    #
    # Create data loader with distributed sampler
    #
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        sampler=sampler,
        num_workers=per_device_workers,
        collate_fn=collate_structure,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=per_device_workers > 0,
        prefetch_factor=prefetch_factor if per_device_workers > 0 else None,
        worker_init_fn=seed_worker if seed is not None else None,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None
    )

    return loader, sampler



def ensure_destination_path(bucket, destination_blob_name):
    #
    # Check if there are any blobs that might conflict with the desired path
    #
    blobs = bucket.list_blobs(prefix=destination_blob_name)
    for blob in blobs:
        if blob.name == destination_blob_name:
            return True  # Path exists and is valid
    return False


def save_pickled_object_to_gcs(bucket_name, destination_blob_name, obj):

    if not destination_blob_name.endswith(".pickle"):
        destination_blob_name += ".pickle"

    #
    # Serialize the object to a byte stream
    #
    pickled_data = pickle.dumps(obj)
    
    #
    # Initialize a GCS client
    #
    storage_client = storage.Client()

    #
    # Get the bucket
    #
    bucket = storage_client.bucket(bucket_name)

    #
    # Ensure the destination path exists
    #
    if not ensure_destination_path(bucket, destination_blob_name):
        print(f"Warning: {destination_blob_name} does not currently exist. Proceeding to create it.")

    #
    # Create a blob object with the desired destination name
    #
    blob = bucket.blob(destination_blob_name)

    #
    # Upload the pickled data
    #
    blob.upload_from_string(pickled_data, content_type="application/octet-stream")

    print(f"Pickled object uploaded to {destination_blob_name}.")


def load_pickled_object_from_gcs(bucket_name, blob_name):

    if not blob_name.endswith(".pickle"):
        blob_name += ".pickle"

    #
    # Initialize a GCS client
    #
    storage_client = storage.Client()

    #
    # Get the bucket and blob
    #
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    #
    # Download the blob's content as bytes
    #
    pickled_data = blob.download_as_bytes()

    #
    # Deserialize the object
    #
    obj = pickle.loads(pickled_data)
    return obj  


def count_blobs_in_path(bucket_name: str, path: str) -> int:

    #
    # Initialize the storage client
    #
    client = storage.Client()

    #
    # Get the bucket
    #
    bucket = client.bucket(bucket_name)

    #
    # List blobs in the given path
    #
    blobs = client.list_blobs(bucket_name, prefix=path)

    #
    # Count the blobs
    #
    blob_count = sum(1 for _ in blobs)

    return blob_count


def tensors_to_structure(abc, angles, atomic, pos, mask, min_dist=0.5):  # Using 0.5 Angstrom as minimum distance

    #
    # Convert tensors to numpy arrays
    #
    abc_np = abc.detach().cpu().numpy()
    angles_np = torch.rad2deg(angles).detach().cpu().numpy()
    atomic_np = atomic.detach().cpu().numpy()
    pos_np = pos.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    
    structures = []
    
    for i in range(len(abc_np)):
        try:
            #
            # Remove start/end tokens and get valid atoms
            #
            valid_indices = mask_np[i].astype(bool)
            valid_atomic = atomic_np[i][valid_indices][1:-1]
            valid_pos = pos_np[i][valid_indices][1:-1]
            
            if len(valid_atomic) == 0:
                continue
            
            #
            # Convert atomic numbers to elements
            #
            elements = []
            for z in valid_atomic:
                try:
                    elements.append(Element.from_Z(int(z)))
                except ValueError:
                    print(f"Warning: Invalid atomic number {z}")
                    continue
            
            if not elements:
                continue
            #    
            # Create lattice
            #
            lattice = Lattice.from_parameters(
                abs(abc_np[i][0]), abs(abc_np[i][1]), abs(abc_np[i][2]),
                abs(angles_np[i][0]), abs(angles_np[i][1]), abs(angles_np[i][2])
            )
            
            #
            # Filter positions based on distances
            #
            final_elements = [elements[0]]
            final_positions = [valid_pos[0]]
            
            for j in range(1, len(elements)):
                valid_position = True
                test_pos = valid_pos[j]
                
                #
                # Create temporary structure to check distances
                #
                temp_struct = Structure(
                    lattice=lattice,
                    species=final_elements + [elements[j]],
                    coords=np.vstack([final_positions, test_pos]),
                    coords_are_cartesian=False,
                    validate_proximity=False
                )
                
                #
                # Check distance to all existing atoms
                #
                for k in range(len(final_elements)):
                    dist = temp_struct.get_distance(k, len(final_elements))
                    if dist < min_dist:
                        valid_position = False
                        break
                
                if valid_position:
                    final_elements.append(elements[j])
                    final_positions.append(test_pos)
            
            if len(final_elements) > 0:
                #
                # Create final structure
                #
                structure = Structure(
                    lattice=lattice,
                    species=final_elements,
                    coords=final_positions,
                    coords_are_cartesian=False,
                    validate_proximity=False  # We've already validated
                )
                structures.append(structure)
            
        except Exception as e:
            print(f"Error processing structure {i}: {str(e)}")
            continue
    
    return structures



def get_composition_string(structure):
    composition = structure.composition
    composition_string = "".join([str(el) * int(composition[el]) 
                                for el in composition.elements])
    return composition_string


def save_structures(structures, dir='structures'):

    #
    # Make sure the directory exists
    #
    if not os.path.isdir(dir):
        os.makedirs(dir)
    
    #
    # Make a viz for each struct
    #
    for struct in structures:
        
        #
        # Get the name for the structure file
        #
        composition = get_composition_string(struct)
        name = os.path.join(dir, composition)

        #
        # Save with a given name
        #
        struct.to(filename=name + ".cif")


def visualize_cif_structures(cif_dir='structures', viz_dir='visualizations', form='png'):
    #
    # Check if input directory exists
    #
    if not os.path.isdir(cif_dir):
        raise ValueError(f"Input directory {cif_dir} does not exist")
    
    #
    # Create visualization directory if it doesn't exist
    #
    if not os.path.isdir(viz_dir):
        os.makedirs(viz_dir)
    
    #
    # Get all CIF files in the directory
    #
    cif_files = glob(os.path.join(cif_dir, "*.cif"))
    
    for cif_file in cif_files:
        #
        # Get base name without extension
        #
        base_name = os.path.splitext(os.path.basename(cif_file))[0]
        form_path = os.path.join(viz_dir, base_name + "." + form)
        
        #
        # Skip if visualization already exists
        #
        if os.path.exists(form_path):
            continue
        
        #
        # Load structure and create visualization
        #
        structure = Structure.from_file(cif_file)
        atoms = AseAtomsAdaptor.get_atoms(structure)
        fig = plot_atoms(atoms, rotation=('45x,45y,45z'))
        plt.savefig(form_path)
        plt.close()


def _log_structures_to_wandb(structures, step=None, desc=None, run=None):
    if run is None:
        run = wandb.run
        if run is None:
            raise ValueError("No active wandb run found. Initialize wandb first.")
    
    images = {}
    for idx, struct in enumerate(structures):
        try:
            atoms = AseAtomsAdaptor.get_atoms(struct)
            
            #
            # Create figure first
            #
            fig = plt.figure(figsize=(8, 8))
            
            #
            # Plot atoms and store the returned axes
            #
            ax = plot_atoms(atoms,
                rotation=('10x,70y,20z'),
                show_unit_cell=True,
                radii=0.3,
                scale=0.8)
            
            #
            # Improve figure aesthetics
            #
            plt.title(f"{struct.composition.reduced_formula}")
            plt.xlabel("X (Å)")
            plt.ylabel("Y (Å)")
            
            #
            # Save with higher quality
            #
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            pil_image = Image.open(buf)
            wandb_desc = f"materials/{struct.composition.reduced_formula}" if desc is None else f"materials-{desc}/{idx}/{struct.composition.reduced_formula}"
            images[wandb_desc] = wandb.Image(pil_image)
            
            plt.close()
            buf.close()
            
        except Exception as e:
            print(f"Error visualizing structure {idx}: {str(e)}")
            continue
    
    if images:
        if step is not None:
            run.log(images, step=step)
        else:
            run.log(images)


def get_clear_paper_palette():
    return {
        "As": "#2D7CFF", "In": "#7B5CFF", "Tb": "#4B5EFF",
        "Rh": "#7F94B8", "Hg": "#6F778A", "Si": "#5E7F9A",
        "Mg": "#2FB7D6", "Cl": "#22C7A8",
        "Br": "#FF3D8D", "Cu": "#E07A3A", "S": "#FF4FA8",
        "Fe": "#D04A42", "O": "#C93B45", "Se": "#D23A3A",
        "C": "#1B2233", "N": "#2B5FA6", "P": "#6F66F0",
        "I": "#7B5CFF", "H": "#C2CAD6",
        "V":  "#7B5CFF",  # vanadium -> paper violet (was falling back to gray)
    }


def element_to_hex(symbol: str, palette: dict, fallback="#6B7280"):
    return palette.get(symbol, fallback)


# 
# Unit cell edges (hi-tech: subtle glow + thinner stroke)
# 
def draw_unit_cell_edges(ax, cell_vecs_3x3, R, xshift, yshift,
                         color="#1B2A4A", lw=0.85, alpha=0.55, zorder=2):
    a, b, c = cell_vecs_3x3[0], cell_vecs_3x3[1], cell_vecs_3x3[2]
    aR, bR, cR = (R @ a), (R @ b), (R @ c)

    O = np.zeros(3)
    corners = [
        O,
        aR, bR, cR,
        aR + bR,
        aR + cR,
        bR + cR,
        aR + bR + cR,
    ]
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7)
    ]

    C = np.array(corners)
    for i, j in edges:
        p, q = C[i], C[j]
        ln = ax.plot([p[0] - xshift, q[0] - xshift],
                     [p[1] - yshift, q[1] - yshift],
                     color=color, lw=lw, alpha=alpha,
                     solid_capstyle="round", zorder=zorder)[0]

        #
        # Subtle sci-fi glow so edges feel “rendered”, not sketched
        #
        ln.set_path_effects([
            pe.Stroke(linewidth=lw + 2.2, foreground=color, alpha=0.10),
            pe.Normal()
        ])


def _rotmat(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis.lower() == "x":
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]], dtype=float)
    if axis.lower() == "y":
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=float)
    if axis.lower() == "z":
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]], dtype=float)
    raise ValueError(f"Bad axis: {axis}")


def rotation_matrix_from_string(rot: str) -> np.ndarray:
    """
    rot example: "10x,70y,20z" (applied in listed order)
    """
    rot = rot.replace(" ", "")
    R = np.eye(3, dtype=float)
    if not rot:
        return R
    for p in rot.split(","):
        if not p:
            continue
        axis = p[-1]
        deg = float(p[:-1])
        R = _rotmat(axis, deg) @ R
    return R


# 
# Main W&B logger (paper-compatible, hi-tech, translucent)
# 
def log_structures_to_wandb(
    structures,
    step=None,
    desc=None,
    run=None,
    rotation="10x,70y,20z",
    figsize=5.0,
    dpi=300,
    radii_scale=0.30,     # increase for more “ball” look; decrease for more “crystal” look
    show_unit_cell=True,
    zoom_out=1.25,        # NEW: >1.0 zooms out so the lattice is not cut
):
    if run is None:
        run = wandb.run
        if run is None:
            raise ValueError("No active wandb run found. Initialize wandb first.")

    palette = get_clear_paper_palette()

    #
    # Paper-compatible background: white
    #
    BG = "#FFFFFF"

    #
    # Unit cell + outlines: match paper inks
    #
    CELL = "#1B2A4A"     # spine tone
    cell_alpha = 0.40
    cell_line_width = 0.70

    OUTLINE = "#0B1020"  # deep ink
    outline_lw = 0.55

    #
    # Specular highlight (glass sphere feel)
    #
    RIM = "#FFFFFF"
    rim_alpha = 0.55
    rim_lw = 0.75

    R = rotation_matrix_from_string(rotation)
    images = {}

    for idx, struct in enumerate(structures):
        try:
            atoms = AseAtomsAdaptor.get_atoms(struct)

            pos = atoms.get_positions()
            posR = (R @ pos.T).T

            xy = posR[:, :2]
            xy_center = xy.mean(axis=0)
            xy = xy - xy_center

            #
            # Per-element radii from covalent radii
            #
            Zs = [atomic_numbers[s] for s in atoms.get_chemical_symbols()]
            r = np.array([covalent_radii[z] for z in Zs], dtype=float) * radii_scale

            # 
            # NEW FRAMING: atoms ∪ cell corners, with zoom_out
            # 
            pad = zoom_out * 2.2 * float(r.max() if len(r) else 1.0)

            xmin_a, ymin_a = xy.min(axis=0) - pad
            xmax_a, ymax_a = xy.max(axis=0) + pad

            if show_unit_cell:
                cell = np.array(atoms.cell)               
                cellR = (R @ cell.T).T                    

                O = np.zeros(3)
                aR, bR, cR = cellR[0], cellR[1], cellR[2]
                corners = np.array([
                    O,
                    aR, bR, cR,
                    aR + bR,
                    aR + cR,
                    bR + cR,
                    aR + bR + cR,
                ])                                        

                corners_xy = corners[:, :2] - xy_center   
                xmin_c, ymin_c = corners_xy.min(axis=0) - 0.6 * pad
                xmax_c, ymax_c = corners_xy.max(axis=0) + 0.6 * pad

                xmin = min(xmin_a, xmin_c); xmax = max(xmax_a, xmax_c)
                ymin = min(ymin_a, ymin_c); ymax = max(ymax_a, ymax_c)
            else:
                xmin, xmax, ymin, ymax = xmin_a, xmax_a, ymin_a, ymax_a

            # 
            # Figure
            # 
            fig = plt.figure(figsize=(figsize, figsize), dpi=dpi, facecolor=BG)
            ax = fig.add_subplot(111, facecolor=BG)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            #
            # Draw unit cell (glowy hi-tech edges)
            #
            if show_unit_cell:
                cell = np.array(atoms.cell)
                draw_unit_cell_edges(
                    ax, cell_vecs_3x3=cell, R=R,
                    xshift=xy_center[0], yshift=xy_center[1],
                    color=CELL, lw=cell_line_width, alpha=cell_alpha, zorder=1
                )

            #
            # Draw atoms back-to-front
            #
            zdepth = posR[:, 2]
            order = np.argsort(zdepth)

            syms = atoms.get_chemical_symbols()
            colors = [element_to_hex(sym, palette) for sym in syms]

            for i in order:
                s_pts2 = (r[i] * 100.0) ** 2
                z = 10 + (zdepth[i] - zdepth.min()) / (zdepth.ptp() + 1e-9)

                # body: translucent
                ax.scatter(
                    [xy[i, 0]], [xy[i, 1]],
                    s=s_pts2,
                    c=[colors[i]],
                    alpha=0.78,
                    edgecolors="none",
                    zorder=z,
                )

                # outline: ink
                ax.scatter(
                    [xy[i, 0]], [xy[i, 1]],
                    s=s_pts2,
                    c="none",
                    edgecolors=OUTLINE,
                    linewidths=outline_lw,
                    alpha=0.90,
                    zorder=z + 0.2,
                )

                # specular highlight (offset)
                ax.scatter(
                    [xy[i, 0] - 0.06 * r[i]], [xy[i, 1] + 0.06 * r[i]],
                    s=0.18 * s_pts2,
                    c=[RIM],
                    alpha=rim_alpha,
                    edgecolors="none",
                    zorder=z + 0.35,
                )

                # faint bright ring
                ax.scatter(
                    [xy[i, 0]], [xy[i, 1]],
                    s=0.92 * s_pts2,
                    c="none",
                    edgecolors=RIM,
                    linewidths=rim_lw,
                    alpha=0.20,
                    zorder=z + 0.30,
                )

            buf = BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor=BG,
            )
            buf.seek(0)
            pil_image = Image.open(buf).convert("RGB")

            formula = struct.composition.reduced_formula
            key = f"materials/{formula}" if desc is None else f"materials-{desc}/{idx}/{formula}"
            images[key] = wandb.Image(pil_image)

            plt.close(fig)
            buf.close()

        except Exception as e:
            print(f"Error visualizing structure {idx}: {str(e)}")
            continue

    if images:
        run.log(images, step=step) if step is not None else run.log(images)



def log_structures3Dcif_to_wandb(structures, step=None, desc=None, run=None):
    if run is None:
        run = wandb.run
        if run is None:
            raise ValueError("No active wandb run found. Initialize wandb first.")

    images = {}
    for idx, struct in enumerate(structures):
        try:
            #
            # Convert structure to CIF format to preserve periodicity
            #
            cif_str = struct.to(fmt="cif")
            
            #
            # Create a py3Dmol view
            #
            view = py3Dmol.view(width=600, height=400)
            view.addModel(cif_str, "cif")  
            
            #
            # Adjust the bond thickness and atom size
            #
            view.setStyle({
                'stick': {'radius': 0.1},  
                'sphere': {'radius': 0.5}  
            })
            
            view.zoomTo()
            view.setBackgroundColor('0xeeeeee')
            
            #
            # Render to HTML and wandb
            #
            html = view._make_html()
            wandb_desc = f"materials3d/Structure {idx}: {struct.composition.reduced_formula}" if desc is None else f"materials3d-{desc}/Structure {idx}: {struct.composition.reduced_formula}"
            images[wandb_desc] = wandb.Html(html)
        
        except Exception as e:
            print(f"Error visualizing structure {idx}: {str(e)}")
            continue

    #
    # Log images to wandb
    #
    if images:
        if step is not None:
            run.log(images, step=step)
        else:
            run.log(images)



def lognormal_mle(data):
    log_data = np.log(data)
    mu = np.mean(log_data)
    sigma = np.std(log_data, ddof=0)  # MLE uses population std (ddof=0)
    return mu, sigma


def length_mle(structures, offset_atoms=True):
    
    lengths = []
    lengths_a = []
    lengths_b = []
    lengths_c = []

    n = len(structures)

    for i in range(n):
        structure = structures[i] # Assumes structure is in pymatgen format
        
        scale = structure.num_sites ** (-1/3) if offset_atoms else 1
        
        lattice = structure.lattice
        lengths.append(scale * lattice.a)
        lengths.append(scale * lattice.b)
        lengths.append(scale * lattice.c)
        lengths_a.append(scale * lattice.a)
        lengths_b.append(scale * lattice.b)
        lengths_c.append(scale * lattice.c)

    mu, sigma = lognormal_mle(lengths)
    mu_a, sigma_a = lognormal_mle(lengths_a)
    mu_b, sigma_b = lognormal_mle(lengths_b)
    mu_c, sigma_c = lognormal_mle(lengths_c)

    lengths_abc = np.array([lengths_a, lengths_b, lengths_c])
    cov = np.cov(np.log(lengths_abc))

    return {
        "general": (mu, sigma),
        "a": (mu_a, sigma_a),
        "b": (mu_b, sigma_b),
        "c": (mu_c, sigma_c),
        "cov": cov

    }


def lattice_mle(structures, offset_by_atoms=True):
    #
    # Length lists
    #
    lengths_a = []
    lengths_b = []
    lengths_c = []

    #
    # Angle lists
    #
    angles_alpha = []
    angles_beta  = []
    angles_gamma = []

    n = len(structures)

    for i in range(n):
        structure = structures[i]  # Assumes structure is in pymatgen format
        lattice = structure.lattice

        #
        # Adjust the scale of recorded lengths
        #
        scale = structure.num_sites ** (-1/3) if offset_by_atoms else 1

        #
        # Lengths
        #
        lengths_a.append(scale * lattice.a)
        lengths_b.append(scale * lattice.b)
        lengths_c.append(scale * lattice.c)

        #
        # Angles
        #
        angles_alpha.append(lattice.alpha)
        angles_beta.append(lattice.beta)
        angles_gamma.append(lattice.gamma)

    #
    # Lognormal MLE for lengths
    #
    lengths = np.concatenate([lengths_a, lengths_b, lengths_c])
    mu, sigma = lognormal_mle(lengths)
    mu_a, sigma_a = lognormal_mle(lengths_a)
    mu_b, sigma_b = lognormal_mle(lengths_b)
    mu_c, sigma_c = lognormal_mle(lengths_c)

    lengths_abc = np.array([lengths_a, lengths_b, lengths_c])
    cov = np.cov(np.log(lengths_abc))

    #
    # Plain mean/std for angles (in degrees)
    #
    angles_alpha = np.array(angles_alpha)
    angles_beta  = np.array(angles_beta)
    angles_gamma = np.array(angles_gamma)

    #
    # MLE-style std (ddof=0) for consistency with lognormal MLE
    #
    alpha_mean = float(np.mean(angles_alpha))
    alpha_std  = float(np.std(angles_alpha, ddof=0))

    beta_mean  = float(np.mean(angles_beta))
    beta_std   = float(np.std(angles_beta, ddof=0))

    gamma_mean = float(np.mean(angles_gamma))
    gamma_std  = float(np.std(angles_gamma, ddof=0))

    #
    # "general" over all three angles pooled
    #
    angles = np.concatenate([angles_alpha, angles_beta, angles_gamma])
    angles_mean = float(np.mean(angles))
    angles_std  = float(np.std(angles, ddof=0))

    mles = {
        "general": (mu, sigma),
        "a": (mu_a, sigma_a),
        "b": (mu_b, sigma_b),
        "c": (mu_c, sigma_c),
        "cov": cov,
        "angles_general": (angles_mean, angles_std),
        "alpha": (alpha_mean, alpha_std),
        "beta": (beta_mean, beta_std),
        "gamma": (gamma_mean, gamma_std),
    }

    lists = {
        "lengths_a": lengths_a,
        "lengths_b": lengths_b,
        "lengths_c": lengths_c,
        "angles_alpha": angles_alpha,
        "angles_beta": angles_beta,
        "angles_gamma": angles_gamma
    }

    return {
        "mle": mles,
        "lists": lists
    }


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


def find_percentiles(targets, alpha=5):
    lower_q = np.percentile(targets, alpha)
    upper_q = np.percentile(targets, 100-alpha)
    return lower_q, upper_q


def _load_m3gnet_calculator(*, compute_stress: bool = False) -> M3GNetCalculator:
    #
    # Load a PES potential that provides energies, forces (and stress if requested)
    #
    potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    return M3GNetCalculator(potential=potential, compute_stress=compute_stress)


def _deterministic_nudge_frac(f: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    #
    # Tiny, deterministic, non-random nudge to break exact degeneracies without RNG
    #
    n = f.shape[0]
    if n == 0:
        return f
    bump = (np.arange(n, dtype=np.float64)[:, None] * np.array([1.0, 2.0, 3.0])) * eps
    return _wrap01(f + bump)


def _safe_to_primitive(
    s: Structure,
    *,
    symprec_candidates: Iterable[float] = (5e-3, 2e-3, 1e-3, 5e-4),
    angle_tolerance_candidates: Iterable[float] = (5.0, 2.0, -1.0),
) -> Structure:
    #
    # Try spglib primitive with multiple tolerances; fall back safely if it fails
    #
    # Wrap and nudge before spglib to avoid coincident sites
    f = _wrap01(s.frac_coords)
    f = _deterministic_nudge_frac(f, eps=1e-10)
    s = Structure(s.lattice, s.species, f, coords_are_cartesian=False)

    for sp in symprec_candidates:
        for ang in angle_tolerance_candidates:
            try:
                out = SpacegroupAnalyzer(s, symprec=sp, angle_tolerance=ang).get_primitive_standard_structure()
                if out is not None and np.isfinite(out.lattice.volume) and out.lattice.volume > 1e-6:
                    return out
            except Exception:
                pass

    #
    # Fallbacks that avoid heavy spglib calls
    #
    try:
        out = s.get_primitive_structure()
        if out is not None and out.lattice.volume > 1e-6:
            return out
    except Exception:
        pass

    try:
        out = s.get_reduced_structure(reduction_algo="niggli")
        if out is not None and out.lattice.volume > 1e-6:
            return out
    except Exception:
        pass

    #
    # Ultimate fallback: return wrapped+nudged input
    #
    return s


def _has_nan_energy_or_forces(atoms) -> bool:
    #
    # Detect NaNs from the calculator and avoid line-search blowups
    #
    try:
        e = atoms.get_potential_energy()
        if not np.isfinite(e):
            return True
    except Exception:
        return True
    try:
        F = atoms.get_forces(apply_constraint=False)
        return not np.isfinite(F).all()
    except Exception:
        return True


def _puff_if_clashing(atoms, *, min_sep_A: float = 0.7, scale: float = 1.01, relax_cell: bool = False) -> None:
    pos = atoms.get_positions()
    if len(pos) < 2:
        return

    # Fast: find any neighbors within min_sep_A (PBC-aware, C-backed)
    # 'ijd' returns i, j, distances; if any exist, we have a clash
    i, j, d = neighbor_list('ijd', atoms, cutoff=min_sep_A - 1e-9)
    has_clash = (len(d) > 0)

    if not has_clash:
        return

    if relax_cell:
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    else:
        # Fixed cell: tiny deterministic nudge in fractional coords
        cell = atoms.cell
        frac = np.linalg.solve(cell.T, pos.T).T
        frac = _deterministic_nudge_frac(frac, eps=1e-3)
        atoms.set_positions(np.dot(frac, cell))


def refine_to_primitive(
    struct: Structure,
    *,
    relax_cell: bool = True,      # set True if lattice is likely strained
    target_fmax: float = 0.01,    # a bit looser than 0.01 to avoid ML noise-floor stalls
    steps: int = 6000,             # keep bounded; this routine is speed-oriented
    calculator: Optional[M3GNetCalculator] = None,
) -> Structure:
    #
    # Fast, stable relax with staged FIRE only. Always returns primitive cell on success.
    #
    calc = calculator or _load_m3gnet_calculator(compute_stress=relax_cell)

    #
    # ASE Atoms + calculator
    #
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = calc

    #
    # Optional cell relaxation wrapper
    #
    obj = UnitCellFilter(atoms) if relax_cell else atoms

    #
    # Pre-sanitize geometry to prevent NaNs and super-slow line searches
    #
    _puff_if_clashing(atoms, relax_cell=relax_cell)
    if _has_nan_energy_or_forces(atoms):
        # Deterministic micro-nudge to escape pathological geometries
        pos = atoms.get_positions()
        from ase.geometry import cell_to_cellpar  # import locally to avoid overhead if unused
        cell = atoms.cell
        frac = np.linalg.solve(cell.T, pos.T).T
        frac = _deterministic_nudge_frac(frac, eps=2e-4)
        atoms.set_positions(np.dot(frac, cell))

    #
    # Stage budgets (pure FIRE, monotone tightening of maxmove/fmax)
    #
    n0 = max(200, steps // 3)
    n1 = max(200, steps // 3)
    n2 = max(200, steps - n0 - n1)

    #
    # Stage 0: coarse settle
    #
    FIRE(obj, maxmove=0.06, logfile=None).run(fmax=0.06, steps=n0)

    #
    # Stage 1: mid pass
    #
    FIRE(obj, maxmove=0.035, logfile=None).run(fmax=0.03, steps=n1)

    #
    # Stage 2: fine pass
    #
    FIRE(obj, maxmove=0.02, logfile=None).run(fmax=target_fmax, steps=n2)

    #
    # Back to pymatgen and enforce primitive cell robustly
    #
    relaxed = AseAtomsAdaptor.get_structure(atoms)
    return _safe_to_primitive(relaxed)


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
    s0 = _safe_to_primitive(struct)

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


#
# Load once
#
_megnet_bg = load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

# Fidelity index mapping from MatGL tutorial:
# 0: PBE, 1: GLLB-SC, 2: HSE, 3: SCAN
_FIDELITY = {"PBE": 0, "GLLB-SC": 1, "HSE": 2, "SCAN": 3}


@torch.no_grad()
def predict_structure_megnet(structure, *, method: str = "PBE", device: str = "cpu") -> float:

    i = _FIDELITY[method]
    state_attr = torch.tensor([i], device=device)

    y = _megnet_bg.predict_structure(structure=structure, state_attr=state_attr)

    if torch.is_tensor(y):
        return float(y.detach().cpu().view(-1)[0].item())
    return float(y)


def structure_for_megnet(s: Structure,
    *,
    use_conventional_if_tiny: bool = True,
    min_len: float = 2.0,
    min_dist: float = 0.6,
    max_sites: int = 200) -> Structure:

    if s is None:
        raise ValueError("Structure is None")

    #
    # Copy to avoid mutating upstream objects
    #
    s = s.copy()

    #
    # Basic sanity: finite lattice
    #
    abc = np.array(s.lattice.abc, dtype=float)
    angles = np.array(s.lattice.angles, dtype=float)
    if not np.isfinite(abc).all() or not np.isfinite(angles).all():
        raise ValueError("Non-finite lattice parameters")

    #
    # Avoid near-degenerate cells (common after primitive reduction)
    #
    if (abc <= 1e-6).any():
        raise ValueError("Degenerate lattice (zero/negative lengths)")

    #
    # If primitive is too tiny, expand or switch representation
    # 
    if use_conventional_if_tiny and (abc < min_len).any():
        # Niggli-reduce tends to be safer than raw primitive
        s = s.get_reduced_structure(reduction_algo="niggli")

    #
    # Wrap into unit cell (keeps frac coords in [0,1))
    #
    s = s.copy(site_properties=s.site_properties)
    s.translate_sites(list(range(len(s))), [0, 0, 0], frac_coords=True, to_unit_cell=True)

    return s


@torch.no_grad()
def bandgap_from_primitive(s_primitive, *, method="PBE", device="cpu") -> float:

    s = structure_for_megnet(s_primitive)
    bg = predict_structure_megnet(s, method=method, device=str(device))
    bg = float(bg.detach().cpu().view(-1)[0].item()) if torch.is_tensor(bg) else float(bg)

    if not math.isfinite(bg):
        return float(15)

    return bg
