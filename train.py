import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
import pickle 
import os 

from absl import app, flags
from ml_collections import config_flags

from models import CliqueFlowmer, Transformer
import models.graphops as graphops
import data.tools as tools
import saving

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', int(37), 'Random seed.') 
flags.DEFINE_integer('batch_size', int(1024), 'Batch size.') 
flags.DEFINE_integer('N_eval', int(1e2), 'Evaluation frequency.')
flags.DEFINE_integer('N_save', int(1e4), 'Saving frequency.')
flags.DEFINE_integer('N_epochs', int(2.5 * 1e4), 'Number of epochs.')
flags.DEFINE_bool('From_scratch', bool(True), 'Whether to start training from scratch.')
flags.DEFINE_bool('offset_atoms', bool(True), 'Whether to offset the lattice lengths by atomns.')
flags.DEFINE_bool('standardize', bool(False), 'Whether to standardize the targets.')
flags.DEFINE_bool('augment', bool(False), 'Whether to augment the structures.')
flags.DEFINE_bool('eform_reg', bool(False), 'Whether the primary target is blended with Eform.')
flags.DEFINE_float('eform_tau', -0.2, 'Threshold to start penalizing Eform.' )
flags.DEFINE_integer('eform_lambda', 0, 'Strength of the Eform penalty.')

config_flags.DEFINE_config_file(
    'config',
    'configs/mp20/cliqueflowmer.py',
    'File with hyperparameter configurations.',
    lock_config=False
)


def setup_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def main(_):
    rank, world_size = setup_distributed()
    torch.set_float32_matmul_precision("highest") 
    device = torch.device(f"cuda:{rank}")
    torch.cuda.empty_cache()
    process_seed = 100 * rank + FLAGS.seed 

    #
    # Parse configs of the run
    #
    kwargs = dict(**FLAGS.config)

    data_kwargs = dict(kwargs['data'])
    model_kwargs = dict(kwargs['model'])
    storage_kwargs = dict(kwargs['storage'])

    #
    # Build model spec string for saving
    #
    model_spec = ', '.join([f'{key}: {value}' for key, value in model_kwargs.items()])

    #
    # Extract specific kwargs
    #
    bucket = storage_kwargs.pop('bucket')
    task_cls = data_kwargs.pop('task')
    model_cls = model_kwargs.pop('cls')

    #
    # Create data path for loading. TODO for you: have your data under such a path in your Google bucket. 
    #
    data_dir = os.path.join('CliqueFlowmer', 'data', 'preprocessed', task_cls) 

    #
    # Create model path for saving 
    #
    model_dir = os.path.join('CliqueFlowmer', 'models', 'states', model_cls, task_cls)

    #
    # Initialize a Wandb run. TODO for you: have your data under such a path in your Google bucket.
    #
    if rank == 0:
        wandb.init(project=f'{task_cls}', name=f'{model_cls}')
        wandb.config.update(FLAGS)

    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    #
    # Load the data
    #
    data = []

    for split in ['train', 'val']:
        data_path = os.path.join(data_dir, split)
        subset = tools.load_pickled_object_from_gcs(bucket, data_path)
        data.append(subset)

    #
    # Extract the train val split
    #
    train_data, val_data = data
    train_inputs, train_targets = list(train_data.values())[:2]
    val_inputs, val_targets = list(val_data.values())[:2]

    #
    # Maybe regularize with formation energy
    #
    def _form_target(x):
        value, eform = x 
        penalty = np.maximum(0, eform - FLAGS.eform_tau)
        target = value + FLAGS.eform_lambda * penalty
        return target 
    
    form_target = lambda x: _form_target(x) if FLAGS.eform_reg else x 

    train_targets = [form_target(x) for x in train_targets]
    val_targets = [form_target(x) for x in val_targets]

    #
    # Maybe standardize
    #
    mean_targets = np.mean(train_targets)
    std_targets = np.std(train_targets)

    preprocess_target = lambda x: (x - mean_targets) / std_targets if FLAGS.standardize else x 
    postprocess_target = lambda x: x * std_targets + mean_targets if FLAGS.standardize else x

    train_targets = [preprocess_target(x) for x in train_targets]
    val_targets = [preprocess_target(x) for x in val_targets]
    
    #
    # Create the train and val data sets
    #
    train_dataset = tools.MatbenchDataset(train_inputs, train_targets, augment=FLAGS.augment)
    val_dataset = tools.MatbenchDataset(val_inputs, val_targets, augment=FLAGS.augment)
    train_loader, train_sampler = tools.get_distributed_loader(train_dataset, FLAGS.batch_size, seed=process_seed)
    val_loader, val_sampler = tools.get_distributed_loader(val_dataset, FLAGS.batch_size)

    #
    # Estimate the initial length distribution
    #
    length_mle_vals = tools.length_mle(train_inputs, FLAGS.offset_atoms)

    print("MLE For Log-Lengths")
    for k, v in length_mle_vals.items():
        if k != "cov":
            print(f"{k}: mu {v[0]} sigma {v[1]}")

    if model_kwargs.pop('mle_prior'):
        model_kwargs["initial_length_dist"] = tools.normal_lengths_from_mle(length_mle_vals, device)

    #
    # Estimate the target percentiles
    #
    lower_p, upper_p = tools.find_percentiles(train_targets)
    model_kwargs["lower_percentile"] = lower_p
    model_kwargs["upper_percentile"] = upper_p 
    print("Percentile Estimation")
    print(f"Lower {lower_p:.3f} Upper {upper_p:.3f}")

    #
    # Initialize the model
    #
    model = globals()[model_cls](**model_kwargs).to(device) 

    if not FLAGS.From_scratch:
        model_path = os.path.join(model_dir, model_spec)
        response = saving.load_model_state_dict_from_gcs(bucket, model_path, model)

        if response is not None:
            model = response 
            model.beta_vae = 1.
            model.beta_mse = 1.
            
    #
    # Make rank 0 broadcast its parameters to all other ranks
    #
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=0)

    model = DDP(model, device_ids=[rank], output_device=rank)
    torch.cuda.synchronize()

    #
    # Train the model 
    #
    total_steps = 0
    
    for epoch in range(FLAGS.N_epochs):
    
        model.train()

        #
        # Re-initialize the loaders
        #
        val_loader = iter(val_loader)
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch) 
        
        for batch in train_loader:
            #
            # Move to device
            #
            abc, angles, spec, pos, mask, targets = tools.move_to_device(batch, device)
            
            #
            # Compute the loss and take a gradient step
            #
            train_info = model.module.training_step(abc, angles, spec, pos, mask, targets)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            #
            # Evaluate the model on the test set
            #
            if total_steps == 0 or (total_steps + 1) % FLAGS.N_eval == 0:

                model.eval()

                try:
                    batch = next(val_loader)
                except StopIteration:
                    val_loader, val_sampler = tools.get_distributed_loader(val_dataset, FLAGS.batch_size, seed=process_seed)
                    val_sampler.set_epoch(epoch)
                    val_loader = iter(val_loader)
                    batch = next(val_loader)

                abc, angles, spec, pos, mask, targets = tools.move_to_device(batch, device)
                eval_info = model.module.eval_step(abc, angles, spec, pos, mask, targets)

                #
                # Gather dictionaries across ranks
                #
                gathered_train_info = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_train_info, train_info)

                gathered_eval_info = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_eval_info, eval_info)

                if rank == 0:
                    #
                    # Aggregate the metrics (mean over ranks)
                    #
                    train_info_agg = {
                        k: sum(d[k].cpu() for d in gathered_train_info) / world_size
                        for k in gathered_train_info[0]
                    }

                    eval_info_agg = {
                        k: sum(d[k].cpu() for d in gathered_eval_info) / world_size
                        for k in gathered_eval_info[0]
                    }

                    wandb.log({f'train/{k}': v for k, v in train_info_agg.items()}, step=total_steps)
                    wandb.log({f'eval/{k}': v for k, v in eval_info_agg.items()}, step=total_steps)

                model.train()
                torch.cuda.empty_cache()

            
            dist.barrier()
            if rank == 0 and (total_steps + 1) % FLAGS.N_save == 0:
                #
                # Save the model
                #
                model_path = os.path.join(model_dir, model_spec)
                saving.save_model_state_dict_to_gcs(bucket, model_path, model.module)
            
            dist.barrier()

            #
            # Update the step count
            #
            total_steps += 1
    
    dist.barrier()

    if rank == 0:
        #
        # Save the model
        #
        model_path = os.path.join(model_dir, model_spec)
        saving.save_model_state_dict_to_gcs(bucket, model_path, model.module)

        #
        # Finish the wandb sesh
        #
        wandb.finish()

    dist.barrier()

if __name__ == '__main__':
    app.run(main)
