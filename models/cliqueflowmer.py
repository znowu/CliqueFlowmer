import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import math 

import architectures.ops as ops 
import architectures.blocks as blocks 
import architectures.backbones as backbones
import distributions.gaussian as gauss

import data.tensor_tools as data_tools
import data.constants as constants
import models.tools as tools 
import models.graphops as graphops

from models.flow import Flow, FlatFlow
from copy import deepcopy

EPS = 1e-6



class CliqueFlowmerEncoder(nn.Module):

    def __init__(self, transformer_dim, n_cliques, clique_dim, knot_dim, n_blocks=2, n_heads=2, n_registers=0, dropout_rate=0.1, submodule=ops.SwiGLU, act=nn.GELU()):
    
        super().__init__()

        #
        # Embedding of the lattice tokens
        #
        self.lattice_emb = nn.Parameter(torch.randn(1, 2, transformer_dim) * 0.02)

        #
        # The projection steps from the lattice to the transformer spaces
        #
        self.abc_emb = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.angle_emb = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.pos_emb = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.index_emb = ops.IndexEmbedding(transformer_dim)


        #
        # Transformer and structure that produce the latent space
        #
        self.transformer = backbones.Transformer(transformer_dim, n_blocks, n_heads, dropout_rate, submodule, act, n_registers)
        self.index_matrix = graphops.chain_of_cliques(n_cliques, clique_dim, knot_dim)
        self.latent_dim = (clique_dim  - knot_dim) * n_cliques + knot_dim

        #
        # The projection steps from the transformer to the latent space
        #
        self.pool = ops.AttentionPool(transformer_dim)
        self.latent_norm = nn.LayerNorm(transformer_dim)
        self.latent_emb = nn.Linear(transformer_dim, 2 * self.latent_dim)
        self.act = act 

    def forward(self, abc, angles, atomic, pos, mask, separate=True):

        #
        # Compute distances for the attention shift
        #
        distances = data_tools.compute_pairwise_distances(abc, angles, pos, periodic=False)

        #
        # Embed molecular information in HD spaces
        #
        abc = torch.log(abc + EPS)
        abc = self.abc_emb(abc)
        angles = self.angle_emb(angles)
        pos = self.pos_emb(pos)

        #
        # Merge the embeddings of geometry information
        #
        x = tools.into_structure_tensor(abc, angles, pos)
        distances = F.pad(distances, (2, 0, 2, 0), value=0)
        attention_mask = F.pad(mask, (2, 0), value=1)
        emb = torch.cat([self.lattice_emb.repeat(abc.shape[0], 1, 1), atomic], dim=1)

        #
        # Build index information
        #
        index_emb = self.index_emb(x.shape[-2] - 2).to(x.device)
        index_emb = F.pad(index_emb, (0, 0, 2, 0), value=0)
        index_emb = index_emb.repeat(emb.size(0), 1, 1)

        #
        # Process with a transformer
        #
        x = self.transformer(x, emb, index_emb, shift=distances, mask=attention_mask)

        #
        # Separate the processed geometry info
        #
        x = self.pool(x, attention_mask)
        x = self.act(x)
        x = self.latent_norm(x)

        #
        # Merge and compute the output
        #
        z = self.latent_emb(x)

        #
        # Output distribution parameters
        #
        mu , log_sigma = z.chunk(2, -1)
        log_sigma = torch.clamp(log_sigma, -10, 10)
        sigma = torch.exp(log_sigma)

        if separate:
            return graphops.separate_latents(mu, self.index_matrix), graphops.separate_latents(sigma, self.index_matrix)

        return mu, sigma


class CliqueFlowmerDecoder(nn.Module):
   
    def __init__(self, n_cliques, clique_dim, knot_dim, transformer_dim, n_blocks=2, n_heads=2, n_registers=0, dropout_rate=0.1, submodule=ops.SwiGLU, act=nn.GELU()):

        super().__init__()

        self.n_cliques = n_cliques
        self.clique_dim = clique_dim 
        self.transformer_dim = transformer_dim 
        self.n_blocks = n_blocks  
        self.act = act 

        #
        # Extracting expressive information from the latent variable 
        #
        self.index_matrix = graphops.chain_of_cliques(n_cliques, clique_dim, knot_dim)
        self.latent_dim = knot_dim + n_cliques * (clique_dim - knot_dim)
        self.index_emb = ops.IndexEmbedding(transformer_dim)
        self.latent_mlp = nn.Linear(self.latent_dim, transformer_dim) 
        self.latent_norm = nn.LayerNorm(transformer_dim)

        #
        # The atomic symbol and molecule geometry transformers
        #
        self.atom_transformer = backbones.Transformer(transformer_dim, n_blocks, n_heads, dropout_rate, submodule, act, n_registers)

        #
        # Mapping transformer features to atomic symbol probabilities
        #
        self.atom_mlp = ops.MLP(transformer_dim, 120, (transformer_dim,))

    def modulate_latent(self, z):
        z = self.latent_mlp(z)
        z = self.act(z)
        z = self.latent_norm(z)
        return z.unsqueeze(-2)

    def forward(self, z, atomic, mask):
        #
        # Adjust the mask for causality
        #
        B, L = atomic.shape[:2]
        mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)) * data_tools.causal_mask(mask).to(mask.device)

        #
        # Embed the sequence index information
        #
        index_emb = self.index_emb(atomic.shape[-2]).to(z.device)
        index_emb = index_emb.repeat(atomic.shape[0], 1, 1)

        #
        # Process the atomic symbols and the latent with the transformer
        #
        z = z.repeat(1, index_emb.shape[-2], 1)
        x = self.atom_transformer(atomic, index_emb, z, mask=mask)

        #
        # Map the features to log-probabilities over symbols
        # 
        x = self.atom_mlp(x)

        return F.log_softmax(x, dim=-1)



class CliqueFlowmer(nn.Module):

    def __init__(self, n_cliques, clique_dim, knot_dim, transformer_dim=128, n_registers=0,
                n_blocks=2, n_heads=2, n_mlp=2, mlp_dim=2 * (256,), dropout_rate=0.1, 
                submodule=ops.SwiGLU, act = nn.GELU(), alpha_vae=1, alpha_mse=1, beta_mse=0, temp_atom=1, 
                temp_flow=1, warmup=1e4, lr=3e-4, polyak_tau=5e-3, temp_distance=0., drop_type=0.1, drop_latent=0.1,
                lower_percentile=None, upper_percentile=None, initial_length_dist=None, use_flat_flow=False):

        super().__init__()

        self.n_cliques = n_cliques
        self.clique_dim = clique_dim 
        self.knot_dim = knot_dim
        self.transformer_dim = transformer_dim 
        self.latent_dim = (clique_dim - knot_dim) * n_cliques + knot_dim
        self.modulator = backbones.PMLP(n_cliques, clique_dim, transformer_dim, mlp_dim, n_mlp, dropout_rate, act) 
        self.modulator_norm = nn.LayerNorm(transformer_dim)

        self.atomic_emb = ops.AtomicEmbedding(transformer_dim)
        self.encoder = CliqueFlowmerEncoder(transformer_dim, n_cliques, clique_dim, knot_dim, n_blocks, n_heads, n_registers, dropout_rate, submodule, act)
        self.decoder = CliqueFlowmerDecoder(n_cliques, clique_dim, knot_dim, transformer_dim, n_blocks, n_heads, n_registers, dropout_rate, submodule, act)
        
        self.use_flat_flow = use_flat_flow
        flow_class = FlatFlow if use_flat_flow else Flow
        self.geo_flow = flow_class(transformer_dim, n_blocks, n_heads, n_cliques, clique_dim, knot_dim, n_registers, dropout_rate, submodule, act, initial_length_dist=initial_length_dist)
        self.regressor = backbones.DMLP(n_cliques, clique_dim, mlp_dim, n_mlp, dropout_rate, act)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = ops.linear_warmup_decay(
            self.optimizer,
            max_lr=lr,
            warmup_steps=1000,
            decay_steps=int(1e6),
        )
        self.target_regressor = deepcopy(self.regressor)

        self.index_matrix = self.encoder.index_matrix
        self.alpha_vae = alpha_vae 
        self.alpha_mse = alpha_mse 
        self.beta_vae = 0
        self.beta_mse = beta_mse 
        self.temp_atom = temp_atom
        self.temp_flow = temp_flow
        self.warmup = warmup
        self.polyak_tau = polyak_tau
        self.temp_distance = temp_distance
        self.drop_type = drop_type
        self.drop_latent = drop_latent
        self.act = act 

        self.lower_percentile = lower_percentile if lower_percentile is not None else torch.tensor([-float('inf')])
        self.upper_percentile = upper_percentile if upper_percentile is not None else torch.tensor([float('inf')])

    def modulate_latent(self, z):
        if self.use_flat_flow:
            z = self.decoder.modulate_latent(z)
        else:
            z = graphops.separate_latents(z, self.index_matrix.to(z.device))
            z = self.modulator(z)
            z = self.act(z)
            z = self.modulator_norm(z)
        return z

    def posterior(self, abc, angles, atomic, pos, mask):
        #
        # Sample the latents
        #
        z_mu, z_sigma = self.encoder(abc, angles, atomic, pos, mask, separate=False)
        noise = torch.randn(abc.shape[0], self.latent_dim).to(abc.device)
        z = z_mu + z_sigma * noise.to(abc.device)

        return z, z_mu, z_sigma
    
    @torch.no_grad()
    def encode(self, abc, angles, atomic, pos, mask, separate=True, batch_limit=1000):
        
        z = []
        batch_size = abc.shape[0]
        n_batches = math.ceil(batch_size / batch_limit)

        for i in range(n_batches):
            start = batch_limit * i 
            end = min(start + batch_limit, batch_size)

            z_i, _ = self.encoder(abc[start : end], angles[start : end], atomic[start : end], pos[start : end], mask[start : end], separate)
            z.append(z_i)

        z = torch.cat(z, dim=0)

        return z 
    
    @torch.no_grad()
    def decode(self, z, integration="cfg", batch_limit=1000, *, n_steps=None,
               beam_width=10, min_atoms=3, max_atoms=100,
               progress_callback=None, cancel_check=None):

        if integration not in {"cfg", "euler", "dopri5", "rk4"}:
            raise ValueError(f"Unknown geometry integration method: {integration}")
        if n_steps is None:
            n_steps = 10000 if integration == "rk4" else 1000
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")
        if not isinstance(batch_limit, int) or batch_limit < 1:
            raise ValueError("batch_limit must be a positive integer.")
        if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] != self.latent_dim:
            raise ValueError(f"Expected a nonempty batch of {self.latent_dim}-dimensional latents.")

        batch_size = z.shape[0]
        n_batches = math.ceil(batch_size / batch_limit)
        device = z.device
        beam_steps = max_atoms + 1
        total_steps = beam_steps + n_batches * n_steps
        tools.report_progress(0, total_steps, progress_callback, cancel_check)

        def beam_progress(step, _total):
            if progress_callback is not None:
                progress_callback(step, total_steps)

        #
        # Modulate the latents 
        #
        z_dec = []
        z_flow = []

        for i in range(n_batches):

            start = batch_limit * i 
            end = min(start + batch_limit, batch_size)

            z_i = z[start : end]

            z_dec_i = self.decoder.modulate_latent(z_i)
            z_flow_i = z_dec_i.clone() if self.use_flat_flow else self.modulate_latent(z_i)

            z_dec.append(z_dec_i)
            z_flow.append(z_flow_i)

        z_dec = torch.cat(z_dec, dim=0)
        z_flow = torch.cat(z_flow, dim=0)

        #
        # Find the atomic symbols with beam search
        #
        atomic = tools.batched_beam_search(
            self, z_dec, beam_width=beam_width, min_atoms=min_atoms,
            max_atoms=max_atoms, progress_callback=beam_progress,
            cancel_check=cancel_check,
        )
        atomic_pad = data_tools.pad_sequences_fast(atomic)
        atomic, mask = data_tools.move_to_device(atomic_pad, device)
        atomic = atomic.long()
        atomic_emb = self.atomic_emb(atomic)

        #
        # Find the geometry with the flow
        #
        abc = []
        angles = []
        pos = []

        for i in range(n_batches):

            start = batch_limit * i 
            end = min(start + batch_limit, batch_size)

            z_dec_i = z_dec[start : end]
            z_flow_i = z_flow[start : end]
            mask_i = mask[start : end]
            atomic_emb_i = atomic_emb[start : end]

            def flow_progress(step, _total):
                if progress_callback is not None:
                    progress_callback(beam_steps + i * n_steps + step, total_steps)

            flow_hooks = {"progress_callback": flow_progress, "cancel_check": cancel_check}
        
            #
            # Generate the geometry with the flow
            #
            if integration == "euler":
                abc_i, angles_i, pos_i = self.geo_flow.sample(z_flow_i, atomic_emb_i, mask_i, n_steps=n_steps, **flow_hooks)
            elif integration == "cfg":
                abc_i, angles_i, pos_i = self.geo_flow.sample_cfg(z_flow_i, self.modulate_latent, atomic_emb_i, mask_i, n_steps=n_steps, omega=2, **flow_hooks)
            elif integration == "dopri5":
                abc_i, angles_i, pos_i = self.geo_flow.sample_dopri5(z_flow_i, atomic_emb_i, mask_i, max_nfe=n_steps, **flow_hooks)
            elif integration == "rk4":
                abc_i, angles_i, pos_i = self.geo_flow.sample_rk4(z_flow_i, atomic_emb_i, mask_i, n_steps=n_steps, **flow_hooks)

            abc.append(abc_i)
            angles.append(angles_i)
            pos.append(pos_i)
        
        abc = torch.cat(abc, dim=0)
        angles = torch.cat(angles, dim=0)
        pos = torch.cat(pos, dim=0)
            

        return abc, angles, atomic, pos, mask
    
    @torch.no_grad()
    def generate(self, n_samples=1000, integration="cfg"):
        
        device = tools.get_device(self)
        z = torch.rand(n_samples, self.latent_dim).to(device)
        abc, angles, atomic, pos, mask = self.decode(z, integration)

        return z, abc, angles, atomic, pos, mask

    def predict(self, z):
        z = graphops.separate_latents(z, self.index_matrix.to(z.device))
        return self.regressor(z) 

    def vae(self, abc, angles, atomic, pos, mask):
        #
        # Embed the atomic symbols
        #
        atom_emb = self.atomic_emb(atomic)

        #
        # Compute the posterior over the latents
        #
        z, z_mu, z_sigma = self.posterior(abc, angles, atom_emb, pos, mask)

        #
        # Structure the distribution params
        #
        index_matrix = self.index_matrix.to(z.device)
        z_mu = graphops.separate_latents(z_mu, index_matrix)
        z_sigma = graphops.separate_latents(z_sigma, index_matrix)

        #
        # Compute the VIB of the posterior
        #
        kl = gauss.standard_kl(z_mu, z_sigma)
        indexes = torch.randint(0, kl.shape[1], (kl.shape[0], 1)).to(abc.device)
        kl_rand = torch.gather(kl, 1, indexes)

        #
        # Map the latent to a HD space
        #
        z_dec = self.decoder.modulate_latent(z)

        #
        # Mask some latents for flow matching
        #
        if self.training and self.drop_latent > 0:
            z = ops.mask_in_batch(z, self.drop_latent)

        z_flow = z_dec.clone() if self.use_flat_flow else self.modulate_latent(z)

        #
        # Compute atomic log probs
        #
        log_probs = self.decoder(z_dec, atom_emb[:, :-1], mask[:, :-1])
        log_probs = torch.gather(log_probs, -1, atomic[:, 1:].unsqueeze(-1)).squeeze(-1)

        #
        # Decide whether to regularize for inter-atomic distance
        #
        get_distance = (self.temp_distance > 0) 

        #
        # Mask some atoms for flow matching
        #
        if self.training and self.drop_type > 0:
            atom_emb = ops.mask_in_sequence(atom_emb, constants.atomic_null_emb, self.drop_type)

        #
        # Get the flow matching error
        #
        error_abc, error_angles, error_pos = self.geo_flow.flow_matching(z_flow, abc, angles, atom_emb, pos, mask, get_distance)
        flow_mask =  tools.true_atom_mask(mask)

        #
        # Summarize the losses in info
        #
        info = {
            'prob': tools.masked_mean(torch.exp(log_probs), mask[:, :-1]).mean(),
            'prob_first': torch.exp(log_probs[..., 0]).mean(),
            'prob_min': torch.min(torch.exp(log_probs) * mask[:, :-1] + 1e3 * (1 - mask[:, :-1]), dim=-1)[0].mean(),
            'prob_min_min': torch.min(torch.exp(log_probs) * mask[:, :-1] + 1e3 * (1 - mask[:, :-1]), dim=-1)[0].min(),
            'atom_log_prob_pd': tools.masked_mean(log_probs, mask[:, :-1]).mean(),
            'atom_log_prob': tools.masked_sum(log_probs, mask[:, :-1]).view(-1),
            'error_abc': error_abc.view(-1),
            'error_angles': error_angles.view(-1),
            'error_pos': tools.masked_sum(error_pos, flow_mask).view(-1),
            'error_pos_pd': tools.masked_mean(error_pos, flow_mask).mean(),
            'kl': kl_rand.view(-1)
        }

        return z, info

    def training_step(self, abc, angles, atomic, pos, mask, target, world_size=1):
        #
        # Compute importance weights of the examples 
        #
        outlier = torch.relu(target - self.upper_percentile) + torch.relu(self.lower_percentile - target) 
        weight = torch.ones(abc.shape[0]).to(abc.device)  
        n_devices = torch.ones(1).to(abc.device)

        #
        # Compute masks for variable-length losses
        #
        flow_mask = tools.true_atom_mask(mask)
        atom_mask = mask[:, :-1]

        #
        # Sum up the masks to get per-example signal counts
        #
        flow_weight = flow_mask.sum(-1) * weight
        atom_weight = atom_mask.sum(-1) * weight

        #
        # Compute global normalization constants
        #
        weight_sums = torch.cat([weight.clone(), flow_weight, atom_weight])
        torch.distributed.all_reduce(weight_sums, op=torch.distributed.ReduceOp.SUM)
        weight_sum, flow_weight_sum, atom_weight_sum = torch.split(weight_sums, abc.shape[0])

        #
        # Compute specific weights
        #
        flow_weight = weight / flow_weight_sum.sum()
        atom_weight = weight / atom_weight_sum.sum()
        weight = weight / weight_sum.sum()

        #
        # Run variational inference
        #
        z, info = self.vae(abc, angles, atomic, pos, mask)

        #
        # Make a prediction about the target
        #
        pred = self.predict(z)
        error = (pred.view(-1) - target.view(-1))**2

        #
        # Compute individual losses
        #
        kl_loss = flow_weight * info['kl']
        error_loss = flow_weight * error  
        abc_loss = weight * info['error_abc'] 
        angles_loss = weight * info['error_angles']  
        pos_loss = flow_weight * info['error_pos']
        atom_loss = -atom_weight * info['atom_log_prob'] 

        #
        # Get the world size 
        #
        torch.distributed.all_reduce(n_devices, op=torch.distributed.ReduceOp.SUM)
        n_devices = float(n_devices.item())

        #
        # Take a gradient step
        #
        temp_kl = self.alpha_vae * self.beta_vae
        temp_mse = self.alpha_mse * self.beta_mse * self.beta_vae
        temp_lattice = 1
        temp_flow = self.temp_flow
        temp_atom = self.temp_atom

        loss = world_size * (temp_kl * kl_loss + temp_mse * error_loss + temp_lattice * (abc_loss + angles_loss) + temp_flow * pos_loss + temp_atom * atom_loss).sum()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.geo_flow.parameters(), 1)        
        torch.nn.utils.clip_grad_norm_(self.modulator.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.regressor.parameters(), 1)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        #
        # Update the regressor's weights with Polyak averaging
        #
        tools.fast_polyak(self.target_regressor, self.regressor, self.polyak_tau)

        #
        # Make KL warmup step
        # 
        self.beta_vae = min(1, self.beta_vae + 1 / self.warmup)

        if self.beta_vae >= 1:
            self.beta_mse = min(1, self.beta_mse + 1 / self.warmup)

        #
        # Take some stats for logging
        #
        r2 = tools.r2(pred.detach().view(-1), target.view(-1))
        mae = torch.abs(pred.detach().view(-1) - target.view(-1)).mean()

        #
        # Fill in the info 
        #
        info['loss'] = loss
        info['mae'] = mae 
        info['mse'] = error.mean()
        info['r2'] = r2 
        info['kl'] = info['kl'].mean()
        info['error_abc'] = info['error_abc'].mean()
        info['error_angles'] = info['error_angles'].mean()
        info['error_pos'] = info['error_pos'].mean()
        info['atom_log_prob'] = info['atom_log_prob'].mean()

        return info 

    def eval_step(self, abc, angles, atomic, pos, mask, target, world_size=1):
        #
        # Compute importance weights of the examples 
        #
        weight = torch.ones(abc.shape[0]).to(abc.device) 

        #
        # Compute weights normalized globally
        #
        global_weight_sum = weight.clone()
        torch.distributed.all_reduce(global_weight_sum, op=torch.distributed.ReduceOp.SUM)
        weight = weight / global_weight_sum.sum()

        self.eval()

        with torch.no_grad():
            #
            # Run variational inference
            #
            z, info = self.vae(abc, angles, atomic, pos, mask)

            #
            # Make a prediction about the target
            #
            pred = self.predict(z)
            error = (pred.view(-1) - target.view(-1))**2

            #
            # Calculate the loss 
            #
            temp_kl = self.alpha_vae * self.beta_vae
            temp_mse = self.alpha_mse * self.beta_mse * self.beta_vae
            temp_lattice = 1
            temp_flow = self.temp_flow
            temp_atom = self.temp_atom
            
            loss = temp_kl * info['kl'] + temp_mse * error + temp_lattice * (info['error_abc'] + info['error_angles']) + temp_flow * info['error_pos']  - temp_atom * info['atom_log_prob']
            loss = (loss * weight).sum()

        #
        # Take some stats for logging
        #
        r2 = tools.r2(pred.detach().view(-1), target.view(-1))
        mae = torch.abs(pred.detach().view(-1) - target.view(-1)).mean()

        #
        # Fill in the info 
        #
        info['loss'] = loss 
        info['mae'] = mae 
        info['mse'] = error.mean()
        info['r2'] = r2 
        info['kl'] = info['kl'].mean()
        info['error_abc'] = info['error_abc'].mean()
        info['error_angles'] = info['error_angles'].mean()
        info['error_pos'] = info['error_pos'].mean()
        info['atom_log_prob'] = info['atom_log_prob'].mean()

        self.train()

        return info 

