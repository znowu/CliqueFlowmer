import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os 
import wandb
import pickle 
import time 

from absl import app, flags
from ml_collections import config_flags

import pandas as pd 
import saving
import data.tools as tools

import matgl
from pymatgen.io.ase import AseAtomsAdaptor
from matgl.ext.ase import M3GNetCalculator
from ase.optimize import FIRE, LBFGS

from matgl import load_model
from models import CliqueFlowmer, Transformer
import models.graphops as graphops

from data.constants import atomic_symbols
from optimization.design import Design
from optimization.learner import GradientDescent, ES

from pymatgen.core import Structure, Lattice, Element

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from optimization.sun import (
    build_reference_metadata,
    classify_sun_for_optimized,
)



FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', int(1), 'Random seed.') 
flags.DEFINE_integer('design_batch_size', int(1e3), 'Design batch size.') 
flags.DEFINE_integer('oracle_batch_size', int(10), 'Oracle batch size.') 
flags.DEFINE_integer('N_eval', int(200), 'Frequency of evaluation.') 
flags.DEFINE_integer('top_k', int(1e2), 'The best designs for evaluation.') 
flags.DEFINE_bool('refine', bool(True), 'Whether to refine the generated structure.')
flags.DEFINE_bool('visualize', bool(False), 'Whether to visualize the structures.')
flags.DEFINE_bool('mid_evals', bool(False), 'Whether to make evals mid-optimization.')
flags.DEFINE_bool('save_structures', bool(False), 'Whether to save the generated structures.')
flags.DEFINE_bool('sun', bool(True), 'Whether to estimate the S.U.N. rate.')
flags.DEFINE_bool('use_targets_for_hull', bool(True), 'Whether to use targets to estimate the hull.')
flags.DEFINE_bool('eform_reg', bool(False), 'Whether the primary target is blended with Eform.')


config_flags.DEFINE_config_file(
    'config',
    'configs/mp20/cliqueflowmer.py',
    'File with hyperparameter configurations.',
    lock_config=False
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().item() if x.numel() == 1 else x.detach().float().cpu().numpy()
    if isinstance(x, np.generic):
        return float(x)
    return x



def main(_):

    #
    # Parse configs of the run
    #
    kwargs = dict(**FLAGS.config)

    data_kwargs = dict(kwargs['data'])
    model_kwargs = dict(kwargs['model'])
    learner_kwargs = dict(kwargs['learner'])
    storage_kwargs = dict(kwargs['storage'])

    #
    # Build model and oracle specs for loading
    #
    model_spec = ', '.join([f'{key}: {value}' for key, value in model_kwargs.items()])

    #
    # Extract specific kwargs
    #
    bucket = storage_kwargs.pop('bucket')
    task_cls = data_kwargs.pop('task')
    model_cls = model_kwargs.pop('cls')
    learner_cls = learner_kwargs.pop('cls')
    learner_steps = learner_kwargs.pop('design_steps')

    #
    # Create data, oracle, and model path for loading
    #
    data_dir = os.path.join('CliqueFlowmer', 'data', 'preprocessed', task_cls)
    model_dir = os.path.join('CliqueFlowmer', 'models', 'states', model_cls, task_cls)

    #
    # Initialize a Wandb run
    #
    wandb.init(project=f'{task_cls}', name=f'model_{model_cls}')
    wandb.config.update(FLAGS)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    #
    # Load the data
    #
    data = []

    for split in ['train', 'val', 'test']:
        data_path = os.path.join(data_dir, split)
        subset = tools.load_pickled_object_from_gcs(bucket, data_path)
        data.append(subset)

    #
    # Extract the train val split
    #
    train_data, val_data, test_data = data
    train_inputs, train_targets = list(train_data.values())[:2]
    val_inputs, val_targets = list(val_data.values())[:2]
    test_inputs, test_targets = list(test_data.values())[:2]

    train_targets = torch.tensor(train_targets).to(device)
    val_targets = torch.tensor(val_targets).to(device)
    test_targets = torch.tensor(test_targets).to(device)

    #
    # Extract Eform regularizer if necessary
    #   
    if FLAGS.eform_reg:
        train_eform = train_targets[..., -1]
        val_eform = val_targets[..., -1]
        test_eform = test_targets[..., -1]

        train_targets = train_targets[..., 0]
        val_targets = val_targets[..., 0]
        test_targets = test_targets[..., 0]
    
    else:
        train_eform = train_targets.clone()
        val_eform = val_targets.clone()
        test_eform = test_targets.clone()

    #
    # Create the reference set
    #
    inputs = train_inputs + val_inputs + test_inputs
    eform = torch.cat([train_eform, val_eform, test_eform])
    targets = torch.cat([train_targets, val_targets, test_targets])

    #
    # Find the min and max of train targets for later normalization
    #
    min_val = targets.min()
    max_val = targets.max()
    print(f'Min: {min_val.item()} Max: {max_val.item()} Mean: {targets.mean().item()}')

    #
    # Initialize the oracle
    #
    oracle = load_model("M3GNet-MP-2018.6.1-Eform")
    oracle = oracle.model
    oracle.eval()
    print("Loaded the oracle!")

    #
    # Set the ground-truth evaluation
    #
    bandgap_task = ("bandgap" in task_cls)
    eval_gt = lambda struct: tools.bandgap_from_primitive(struct) if bandgap_task else oracle.predict_structure(struct)

    #
    # Build the reference set
    #
    ref_device = torch.device("cpu")
    hull_splits = ("train", "val", "test")

    print("Building reference metadata for SUN...")
    ref_entries, by_elemset = build_reference_metadata(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        test_inputs=test_inputs,
        train_targets=train_eform,
        val_targets=val_eform,
        test_targets=test_eform,
        batch_size=128,
        device=ref_device,
        hull_splits=hull_splits,
    )
    print(f"Built {len(ref_entries)} reference entries.")
    
    #
    # Estimate the initial length distribution
    #
    length_mle_vals = tools.length_mle(train_inputs)

    print("MLE For Log-Lengths")
    for k, v in length_mle_vals.items():
        if k != "cov":
            print(f"{k}: mu {v[0]} sigma {v[1]}")

    if model_kwargs.pop('mle_prior'):
        model_kwargs["initial_length_dist"] = tools.normal_lengths_from_mle(length_mle_vals, device)

    #
    # Initialize the model
    #
    model = globals()[model_cls](**model_kwargs).to(device)

    #
    # Load the model
    #
    model_path = os.path.join(model_dir, model_spec)
    response = saving.load_model_state_dict_from_gcs(bucket, model_path, model)

    if response is None:
        raise FileNotFoundError(
            f"Model checkpoint not found: gs://{bucket}/{model_path}.pth. "
            "Train the model first or check the storage bucket and model configuration."
        )
    model = response

    model = nn.DataParallel(model).to(device)
    model.eval()

    #
    # Sample the data to initialize the designs at
    #
    indices = np.random.choice(len(inputs), FLAGS.design_batch_size, replace=False)
    init_design = tools.unpack_structures([inputs[i] for i in indices])
    abc, angles, atomic, pos, mask = tools.move_to_device(init_design, device)
    structures = tools.tensors_to_structure(abc, angles, atomic, pos, mask)

    #
    # Visualize the drawn structures
    #
    if FLAGS.visualize:
        tools.log_structures3Dcif_to_wandb(structures, desc="start", step=0)

    #
    # Make predictions
    #
    with torch.no_grad():
        true_val = torch.tensor([eval_gt(s) for s in structures]).view(-1).to(device)

    #
    # Remember the initial value
    #
    start_val = 1. * true_val
    
    #
    # Get the average of the whole population
    #
    true_val_pop = true_val.mean()
    norm_val_pop = (true_val_pop - min_val) / (max_val - min_val)

    #
    # Get the top k values
    #
    true_val, indices = torch.topk(true_val, FLAGS.top_k, largest=False)
    norm_val = (true_val - min_val) / (max_val - min_val)

    #
    # Log the stats of the original batch
    #
    info = {
        "true_val_pop": true_val_pop,
        "norm_val_pop": norm_val_pop,
        "true val_top": true_val.mean(),
        "true_val_min": true_val.min(),
        "norm_val_top": norm_val.mean(),
        "norm_val_min": norm_val.min(),
    }
    wandb.log({f'GroundTruth/{k}': v for k, v in info.items()}, step=0)

    #
    # Compute the design params and initialize the design
    #
    atomic_emb = model.module.atomic_emb(atomic)
    design = model.module.encode(abc, angles, atomic_emb, pos, mask, separate=False)
    design = Design(design)

    #
    # Initialize the learner
    #
    index_matrix = model.module.index_matrix.to(device)
    learner_kwargs['structure_fn'] = lambda x: graphops.separate_latents(x, index_matrix)
    learner_model = nn.DataParallel(model.module.target_regressor.to(device))
    learner = globals()[learner_cls](design, learner_model, **learner_kwargs)

    #
    # Start measuring time
    #
    start_timer = time.perf_counter()

    #
    # Train the learner
    #
    for step in range(learner_steps):
        
        #
        # Make a train step
        #
        if step > 0:
            train_info = learner.train_step()
        
        if FLAGS.mid_evals and (step % FLAGS.N_eval == 0 or step + 1 == learner_steps):
            #
            # Obtain the (standardized) grount-truth value of the design
            #
            with torch.no_grad():
                design = learner.design_fn()                    
                abc, angles, atomic, pos, mask = model.module.decode(design)
                structures = tools.tensors_to_structure(abc, angles, atomic, pos, mask)
            
            #
            # Measure material lengths
            #
            lengths = mask.sum(-1).cpu() - 2 # two stems from the start and end token
            
            #
            # Make predictions
            #
            with torch.no_grad():
                true_val = torch.tensor([eval_gt(s) for s in structures]).view(-1).to(device)

            #
            # Get the average of the whole population
            #
            true_val_pop = true_val.mean()
            norm_val_pop = (true_val_pop - min_val) / (max_val - min_val)

            #
            # Get the top k values
            #
            true_val, indices = torch.topk(true_val, FLAGS.top_k, largest=False)
            norm_val = (true_val - min_val) / (max_val - min_val)

            #
            # Estimate the model's perceived value
            # 
            with torch.no_grad():
                val = learner.value() 

            info = {
                "model val": val.item(),
                "true_val_pop": true_val_pop,
                "norm_val_pop": norm_val_pop,
                "true val": true_val.mean(),
                "true_val_min": true_val.min(),
                "norm_val": norm_val.mean(),
                "norm_val_min": norm_val.min(),
                "lengths_mean": lengths.mean(),
                "lengths_std": lengths.std()
            }
            wandb.log({f'optim/{k}': v for k, v in info.items()}, step=step)

    #
    # Stop measuring time
    #
    end_timer = time.perf_counter()
    es_time = end_timer - start_timer

    print(f"ES takes {es_time} seconds!")

    #
    # Start measuring time
    #
    start_timer = time.perf_counter()

    #
    # Turn tensor representations of structures into structures
    #
    with torch.no_grad():
        design = learner.design_fn()                    
        abc, angles, atomic, pos, mask = model.module.decode(design)
        structures = tools.tensors_to_structure(abc, angles, atomic, pos, mask)

        #
        # Find the best (expected) structures
        #
        best_values, best_indices = learner.best(FLAGS.top_k)

    #
    # Stop measuring time
    #
    end_timer = time.perf_counter()
    decode_time = end_timer - start_timer

    print(f"Decoding takes {decode_time} seconds!")

    #
    # Refine structures if necessary
    #
    if FLAGS.refine:
        #
        # Reuse one calculator across the batch to avoid reloading the model
        #
        for i in range(FLAGS.design_batch_size):
            print(f"Refining structure {i+1}.")
            
            #
            # Grab a structure
            #
            struct = structures[i]

            #
            # Refine the structure (fixed cell)
            #
            refined_primitive = tools.refine_to_primitive_fast_strong(struct)

            #
            # Update the structure
            #
            structures[i] = refined_primitive

    #
    # Make predictions
    #
    with torch.no_grad():
        true_val = torch.tensor([eval_gt(s) for s in structures]).view(-1).to(device)

    #
    # Find how many examples we improved
    #
    improvement_rate = (1. * (true_val < start_val)).mean()
    wandb.log({'Improvement Rate': improvement_rate})

    #
    # Find how many of the best examples came from improvement
    #
    best_true_val = true_val.index_select(0, best_indices)
    best_start_val = start_val.index_select(0, best_indices)
    best_improvement_rate = (1. * (best_true_val < best_start_val)).mean()
    wandb.log({'Best Improvement Rate': best_improvement_rate})

    #
    # Get the average of the whole population
    #
    true_val_pop = true_val.mean()
    norm_val_pop = (true_val_pop - min_val) / (max_val - min_val)
    
    #
    # Get the average of expected best values
    #
    best_true_val = best_true_val.mean()
    best_norm_val = (best_true_val - min_val) / (max_val - min_val)

    #
    # Get the top k values
    #
    true_val_top, indices = torch.topk(true_val, FLAGS.top_k, largest=False)
    norm_val_top = (true_val_top - min_val) / (max_val - min_val)

    #
    # Log the stats of the final batch
    #
    info = {
        "true_val_pop": true_val_pop,
        "norm_val_pop": norm_val_pop,
        "true val_top": true_val_top.mean(),
        "true_val_min": true_val.min(),
        "norm_val_top": norm_val_top.mean(),
        "norm_val_min": norm_val.min(),
        "best_true_val": best_true_val,
        "best_norm_val": best_norm_val
    }
    wandb.log({f'GroundTruth/{k}':  _to_float(v) for k, v in info.items()}, step=learner_steps+1)    

    #
    # Log the final structures
    #
    if FLAGS.visualize:
        tools.log_structures3Dcif_to_wandb(structures, desc="optimized")

    #
    # Save the optimized structures
    #
    if FLAGS.save_structures:
        with open(f'{task_cls}/optimized_structures.pkl', 'wb') as f:
            pickle.dump(structures, f)

    #
    # Save the expected best optimized structures 
    #
    best_indices = best_indices.cpu().tolist()
    best_structures = [structures[index] for index in best_indices]

    if FLAGS.save_structures:
        with open(f'{task_cls}/best_optimized_structures.pkl', 'wb') as f:
            pickle.dump(best_structures, f)

    if FLAGS.sun:
        # 
        # Compute SUN metrics on optimized structures
        # 
        print("Evaluating SUN metrics on optimized structures...")

        indexes, sun_results = classify_sun_for_optimized(
            optimized_structs=structures,
            train_inputs=train_inputs,
            val_inputs=val_inputs,
            test_inputs=test_inputs,
            ref_entries=ref_entries,
            by_elemset=by_elemset,
            oracle=oracle,
            stable_threshold=0.0,
            metastable_threshold=0.08,
            min_dist=0.5,
        )


        #
        # Calculate in-SUN metrics
        #
        sun_indexes = indexes[:sun_results['N_SUN_stable']]
        
        sun_true_val = [true_val[i] for i in sun_indexes]
        sun_band_gap = [band_gap[i] for i in sun_indexes]

        sun_results['Target-per-SUN'] = float(np.mean(sun_true_val))
        sun_results['Target-per-SUN-min'] = float(np.min(sun_true_val))
        sun_results['BandGap-per-SUN'] = float(np.mean(sun_band_gap))
        sun_results['BandGap-per-SUN-min'] = float(np.min(sun_band_gap))

        print("SUN evaluation (strictly stable, Ehull < 0.0 eV/atom):")
        print(f"Target property          = {np.mean(true_val)}")
        print(f"Band Gap                 = {np.mean(band_gap)}")
        print(f"N_gen                    = {sun_results['N_gen']}")
        print(f"N_stable                 = {sun_results['N_stable']}")
        print(f"N_SUN_stable             = {sun_results['N_SUN_stable']}")
        print(f"StabilityRate_stable     = {sun_results['StabilityRate_stable']:.4f}")
        print(f"SUNRate_stable           = {sun_results['SUNRate_stable']:.4f}")

        print("\nSUN evaluation (metastable-or-better, Ehull < 0.08 eV/atom):")
        print(f"N_metastable             = {sun_results['N_metastable']}")
        print(f"N_SUN_metastable         = {sun_results['N_SUN_metastable']}")
        print(f"StabilityRate_metastable = {sun_results['StabilityRate_metastable']:.4f}")
        print(f"SUNRate_metastable       = {sun_results['SUNRate_metastable']:.4f}")

        #
        # Clean the results up for logging
        #
        info = {}

        for k, v in sun_results.items():
            if isinstance(v, float) or isinstance(v, int):
                info[k] = v 

        info["TargetProperty"] = np.mean(true_val)

        #
        # Log the final structures
        #
        wandb.log({f'Stability/{k}': v for k, v in info.items()})


    #
    # End the run
    #
    wandb.log({}, commit=True)
    wandb.finish()

    #
    # Remind times
    #
    print(f"ES Time {es_time}. Decode Time {decode_time}.")


if __name__ == '__main__':
    app.run(main)
