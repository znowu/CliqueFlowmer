import torch 
import torch.nn as nn 
import torch.nn.functional as F
import math 

import architectures.ops as ops 
import architectures.blocks as blocks 
import architectures.backbones as backbones

import models.graphops as graphops

import data.tensor_tools as data_tools
import models.tools as tools 
import os 

rank = int(os.environ.get("RANK", 0))

EPS = 1e-6

def lifted_logitnorm(shape, device, eps=0.1):

  is_uni = 1. * (torch.rand(shape, device=device) < eps) 
  
  t_uni = torch.rand(shape, device=device)
  t_ln = torch.sigmoid(torch.randn(shape, device=device))
  
  t = is_uni * t_uni + (1 - is_uni) * t_ln 

  return t


def _mk_time_like(t_scalar: float, like_tensor: torch.Tensor) -> torch.Tensor:
    return torch.full(like_tensor.shape, t_scalar, device=like_tensor.device, dtype=like_tensor.dtype)


class Flow(nn.Module):

    def __init__(self, transformer_dim, n_blocks, n_heads, n_cliques, clique_dim, knot_dim, n_registers=0, dropout_rate=0.1, submodule=ops.SwiGLU, act=nn.GELU(), initial_length_dist=None, offset_by_atoms=True):

        super().__init__()

        self.lattice_emb = nn.Parameter(torch.randn(1, 2, transformer_dim) * 0.02)
        self.time_emb = ops.TimeEmbedding(transformer_dim)

        #
        # The projection steps from the lattice to the transformer spaces
        #
        self.abc_emb = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.angle_emb = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)
        self.pos_emb = ops.MLP(3, transformer_dim, (2 * transformer_dim,), act_final=True)

        #
        # Transformer and structure that produce the latent space
        #
        self.modulator = ops.MLP(2 * transformer_dim, transformer_dim, (4 * transformer_dim,), act_final=True)
        self.modulator_norm = nn.LayerNorm(transformer_dim)
        self.transformer = backbones.TransformerDecoder(transformer_dim, n_blocks, n_heads, dropout_rate, submodule, act)

        #
        # The projection from the transformer features
        #
        self.abc_from_model = ops.MLP(transformer_dim, 3, (transformer_dim,))
        self.angles_from_model = ops.MLP(transformer_dim, 3, (transformer_dim,))
        self.pos_from_model = ops.MLP(transformer_dim, 3, (transformer_dim,))

        #
        # Set up the initial noise dist
        #
        self.initial_length_dist = initial_length_dist
        self.act = act

        #
        # Set up the latent structure
        #
        self.index_matrix = graphops.chain_of_cliques(n_cliques, clique_dim, knot_dim)
        self.latent_dim = (clique_dim  - knot_dim) * n_cliques + knot_dim
        self.offset_by_atoms = offset_by_atoms

    def forward(self, z, t, abc, angles, atomic, pos, mask):

        #
        # Make the time into HD space
        #
        t_emb = self.time_emb(t)
        t = t.view(-1, 1, 1)

        #
        # Create the mask of valid atoms and filter out the invalid ones
        #
        keep = tools.true_atom_mask(mask).unsqueeze(-1)
        pos = pos * keep 

        #
        # Embed molecular information in HD spaces
        #
        abc = self.abc_emb(abc)
        angles = self.angle_emb(angles)
        pos = self.pos_emb(pos) 

        #
        # Merge the embeddings of geometry information 
        #
        x = tools.into_structure_tensor(abc, angles, pos) 
        mask =  F.pad(mask, (2, 0), value=1)
        
        #
        # Merge type and time embeddings
        #
        emb = torch.cat([self.lattice_emb.repeat(abc.shape[0], 1, 1), atomic], dim=1)
        t_emb = t_emb.repeat(1, emb.size(1), 1)

        #
        # Process with a transformer
        #
        x = self.transformer(x, z, emb, t_emb, mask = mask)

        #
        # Separate the geometry information and process independently
        #
        abc, angles, pos = tools.from_structure_tensor(x)

        abc = self.abc_from_model(abc)
        angles = self.angles_from_model(angles)
        pos = self.pos_from_model(pos)

        return abc, angles, pos

    def initial_noise(self, mask):
        #
        # Note the important details
        #
        B, L = mask.shape
        device = mask.device
        keep = tools.true_atom_mask(mask)
        n_atoms = keep.sum(-1, keepdim=True)

        #
        # Draw the random noise for each attribute
        #
        if self.initial_length_dist is None:
            noise_abc = torch.randn(B, 3).to(device) * 0.4 + 2 

        else:
            noise_abc = self.initial_length_dist.rsample((B,))  # reparameterized

        noise_abc = noise_abc + (1/3) * torch.log(n_atoms)if self.offset_by_atoms else noise_abc # Offset by the number of atoms
        noise_angles = torch.rand(B, 3).to(device) * math.pi / 3 + math.pi / 3 
        noise_pos = torch.rand(B, L, 3).to(device) 
        
        #
        # Zero out non-true atoms (start/end/invalid) — differentiable
        #
        noise_pos = noise_pos * keep.unsqueeze(-1)         

        return noise_abc, noise_angles, noise_pos

    def flow_matching(self, z, abc, angles, atomic, pos, mask, get_distance=False):
        #
        # Get the shape
        #
        B, N = mask.shape 
        eps = 1e-3
        keep = tools.true_atom_mask(mask)

        #
        # Transform lengths abc into log-space
        #
        abc = torch.log(abc + EPS)
        pos = pos * keep.unsqueeze(-1) 

        #
        # Draw the initial noise
        #
        noise_abc, noise_angles, noise_pos = self.initial_noise(mask)

        #
        # Obtain the regression target (velocity)
        #
        v_abc = abc - noise_abc 
        v_angles = angles - noise_angles
        v_pos = pos - noise_pos

        #
        # Obtain the noisy input
        #
        t = lifted_logitnorm((abc.shape[0], 1), z.device)
        t = ((1 + eps) * t - eps).clamp_min(0)

        noisy_abc = noise_abc + t * v_abc 
        noisy_angles = noise_angles + t * v_angles 
        noisy_pos = noise_pos + t.unsqueeze(-1) * v_pos 
        
        #
        # Make a prediction
        #
        pred_abc, pred_angles, pred_pos = self(z, t.view(-1), noisy_abc, noisy_angles, atomic, noisy_pos, mask)

        #
        # Calculate the flow errors
        #
        error_abc = v_abc - pred_abc 
        error_angles = v_angles - pred_angles
        error_pos = v_pos - pred_pos 


        return (error_abc**2).mean(-1), (error_angles**2).mean(-1), keep.squeeze(-1) * (error_pos**2).mean(-1)

    def get_latent_noise(self, batch_size, device):
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        return noise 

    @torch.no_grad()
    def sample(self, z, atomic, mask, n_steps=1000, *, progress_callback=None, cancel_check=None):
        #
        # Initialize a molecule
        #
        abc, angles, pos = self.initial_noise(mask)

        #
        # Initialize the running timestep vector
        #
        t = torch.zeros(atomic.shape[0]).to(atomic.device)

        for step in range(n_steps):
            tools.report_progress(step, n_steps, progress_callback, cancel_check)
            #
            # Predict the velocity
            #
            v_abc, v_angles, v_pos = self(z, t, abc, angles, atomic, pos, mask)

            #
            # Make an Euler step
            #
            abc = abc + v_abc / n_steps 
            angles = (angles + v_angles / n_steps).clamp(math.pi / 3 + EPS, 2 * math.pi / 3 - EPS)
            pos = (pos + v_pos / n_steps).clamp(0, 1)

            #
            # Update the time vector
            #
            t = t + 1 / n_steps
        
        tools.report_progress(n_steps, n_steps, progress_callback, cancel_check)
        if z.is_cuda:
            torch.cuda.empty_cache()

        return (torch.exp(abc) - EPS).clamp_min(0), angles.clamp(0, math.pi - EPS), pos.clamp(0, 1)
    
    @torch.no_grad()
    def sample_cfg(self, z, modulate_fn, atomic, mask, n_steps=1000, omega=2, *, progress_callback=None, cancel_check=None):
        #
        # Initialize a molecule
        #
        abc, angles, pos = self.initial_noise(mask)

        #
        # Initialize the running timestep vector
        #
        t = torch.zeros(atomic.shape[0]).to(atomic.device)

        for step in range(n_steps):
            tools.report_progress(step, n_steps, progress_callback, cancel_check)
            #
            # Predict the velocity
            #
            v_abc, v_angles, v_pos = self(z, t, abc, angles, atomic, pos, mask)

            #
            # Predict the unconditional velocity
            #
            noise = self.get_latent_noise(z.shape[0], z.device)
            noise = modulate_fn(noise)
            v_abc0, v_angles0, v_pos0 = self(noise, t, abc, angles, atomic, pos, mask)

            #
            # Combine velocities with CFG
            #
            v_abc = (1 + omega) * v_abc - omega * v_abc0
            v_angles = (1 + omega) * v_angles - omega * v_angles0
            v_pos = (1 + omega) * v_pos - omega * v_pos0

            #
            # Make an Euler step
            #
            abc = abc + v_abc / n_steps 
            angles = (angles + v_angles / n_steps).clamp(math.pi / 3 + EPS, 2 * math.pi / 3- EPS)
            pos = (pos + v_pos / n_steps).clamp(0, 1)

            #
            # Update the time vector
            #
            t = t + 1 / n_steps
        
        tools.report_progress(n_steps, n_steps, progress_callback, cancel_check)
        if z.is_cuda:
            torch.cuda.empty_cache()

        return (torch.exp(abc) - EPS).clamp_min(0), angles.clamp(0, math.pi - EPS), pos.clamp(0, 1)

    @torch.no_grad()
    def sample_dopri5(self, z, atomic, mask, max_nfe = 1000, t0 = 0.0, t1 = 1.0, rtol = 1e-4, atol = 1e-4, h_init = None, *, progress_callback=None, cancel_check=None):
        """
        Adaptive Dormand–Prince 5(4) (RK45) sampler for your flow matching model.

        Integrates the ODE:
            d/dt (abc, angles, pos) = v_theta(z, t, abc, angles, atomic, pos, mask)
        from t0 to t1, starting from your initial_noise prior.

        Non-periodic convention:
            - angles are physically in [0, π]
            - positions are physically in [0, 1]
        but the integrator state is unconstrained; clamping is applied only
        for network inputs and after accepted steps.
        """

        device = atomic.device
        dtype = atomic.dtype
        batch_size = atomic.shape[0]

        # --- Initial state (in your parameterization) ---
        abc, angles, pos = self.initial_noise(mask)  # same shapes as in Euler sampler

        # --- Time bookkeeping ---
        t = float(t0)
        if h_init is None:
            h = (t1 - t0) / 100.0  # start with ~100 steps, adapt from there
        else:
            h = float(h_init)

        # minimum step to avoid stalling
        min_h = (t1 - t0) * 1e-6

        # safety factors for adaptive step update
        safety = 0.9
        min_factor = 0.2
        max_factor = 5.0

        # Dormand–Prince 5(4) coefficients (classical tableau)
        c2 = 1.0 / 5.0
        c3 = 3.0 / 10.0
        c4 = 4.0 / 5.0
        c5 = 8.0 / 9.0
        c6 = 1.0
        c7 = 1.0

        a21 = 1.0 / 5.0

        a31 = 3.0 / 40.0
        a32 = 9.0 / 40.0

        a41 = 44.0 / 45.0
        a42 = -56.0 / 15.0
        a43 = 32.0 / 9.0

        a51 = 19372.0 / 6561.0
        a52 = -25360.0 / 2187.0
        a53 = 64448.0 / 6561.0
        a54 = -212.0 / 729.0

        a61 = 9017.0 / 3168.0
        a62 = -355.0 / 33.0
        a63 = 46732.0 / 5247.0
        a64 = 49.0 / 176.0
        a65 = -5103.0 / 18656.0

        a71 = 35.0 / 384.0
        a72 = 0.0
        a73 = 500.0 / 1113.0
        a74 = 125.0 / 192.0
        a75 = -2187.0 / 6784.0
        a76 = 11.0 / 84.0

        # 5th-order weights
        b1 = 35.0 / 384.0
        b2 = 0.0
        b3 = 500.0 / 1113.0
        b4 = 125.0 / 192.0
        b5 = -2187.0 / 6784.0
        b6 = 11.0 / 84.0
        b7 = 0.0

        # 4th-order embedded weights (for error estimate)
        b1_hat = 5179.0 / 57600.0
        b2_hat = 0.0
        b3_hat = 7571.0 / 16695.0
        b4_hat = 393.0 / 640.0
        b5_hat = -92097.0 / 339200.0
        b6_hat = 187.0 / 2100.0
        b7_hat = 1.0 / 40.0

        def eval_v(abc, angles, pos, t_scalar: float):
            """Evaluate velocity field with clamped inputs."""
            # Clamp for the model input (non-periodic convention)
            angles_in = angles.clamp(0.0, math.pi - EPS)
            pos_in = pos.clamp(0.0, 1.0)

            t_vec = _mk_time_like(t_scalar, atomic)

            v_abc, v_angles, v_pos = self(
                z, t_vec, abc, angles_in, atomic, pos_in, mask
            )
            return v_abc, v_angles, v_pos

        def clamp_state(abc, angles, pos):
            # Physical projection (applied after an accepted step)
            angles = angles.clamp(math.pi / 3 + EPS, 2 * math.pi / 3 - EPS)
            pos = pos.clamp(0.0, 1.0)
            return abc, angles, pos

        nfe = 0

        while t < t1 and nfe < max_nfe:
            tools.report_progress(nfe, max_nfe, progress_callback, cancel_check, every=1)
            # Do not step past t1
            if t + h > t1:
                h = t1 - t
            if h <= 0.0:
                break

            # --- Stage 1 ---
            k1_abc, k1_ang, k1_pos = eval_v(abc, angles, pos, t)
            nfe += 1

            # Stage 2
            t2 = t + c2 * h
            y2_abc = abc + h * (a21 * k1_abc)
            y2_ang = angles + h * (a21 * k1_ang)
            y2_pos = pos + h * (a21 * k1_pos)
            k2_abc, k2_ang, k2_pos = eval_v(y2_abc, y2_ang, y2_pos, t2)
            nfe += 1

            # Stage 3
            t3 = t + c3 * h
            y3_abc = abc + h * (a31 * k1_abc + a32 * k2_abc)
            y3_ang = angles + h * (a31 * k1_ang + a32 * k2_ang)
            y3_pos = pos + h * (a31 * k1_pos + a32 * k2_pos)
            k3_abc, k3_ang, k3_pos = eval_v(y3_abc, y3_ang, y3_pos, t3)
            nfe += 1

            # Stage 4
            t4 = t + c4 * h
            y4_abc = abc + h * (a41 * k1_abc + a42 * k2_abc + a43 * k3_abc)
            y4_ang = angles + h * (a41 * k1_ang + a42 * k2_ang + a43 * k3_ang)
            y4_pos = pos + h * (a41 * k1_pos + a42 * k2_pos + a43 * k3_pos)
            k4_abc, k4_ang, k4_pos = eval_v(y4_abc, y4_ang, y4_pos, t4)
            nfe += 1

            # Stage 5
            t5 = t + c5 * h
            y5_abc = abc + h * (
                a51 * k1_abc + a52 * k2_abc + a53 * k3_abc + a54 * k4_abc
            )
            y5_ang = angles + h * (
                a51 * k1_ang + a52 * k2_ang + a53 * k3_ang + a54 * k4_ang
            )
            y5_pos = pos + h * (
                a51 * k1_pos + a52 * k2_pos + a53 * k3_pos + a54 * k4_pos
            )
            k5_abc, k5_ang, k5_pos = eval_v(y5_abc, y5_ang, y5_pos, t5)
            nfe += 1

            # Stage 6
            t6 = t + c6 * h
            y6_abc = abc + h * (
                a61 * k1_abc + a62 * k2_abc + a63 * k3_abc +
                a64 * k4_abc + a65 * k5_abc
            )
            y6_ang = angles + h * (
                a61 * k1_ang + a62 * k2_ang + a63 * k3_ang +
                a64 * k4_ang + a65 * k5_ang
            )
            y6_pos = pos + h * (
                a61 * k1_pos + a62 * k2_pos + a63 * k3_pos +
                a64 * k4_pos + a65 * k5_pos
            )
            k6_abc, k6_ang, k6_pos = eval_v(y6_abc, y6_ang, y6_pos, t6)
            nfe += 1

            # Stage 7 (for embedded 4th-order)
            t7 = t + c7 * h
            y7_abc = abc + h * (
                a71 * k1_abc + a72 * k2_abc + a73 * k3_abc +
                a74 * k4_abc + a75 * k5_abc + a76 * k6_abc
            )
            y7_ang = angles + h * (
                a71 * k1_ang + a72 * k2_ang + a73 * k3_ang +
                a74 * k4_ang + a75 * k5_ang + a76 * k6_ang
            )
            y7_pos = pos + h * (
                a71 * k1_pos + a72 * k2_pos + a73 * k3_pos +
                a74 * k4_pos + a75 * k5_pos + a76 * k6_pos
            )
            k7_abc, k7_ang, k7_pos = eval_v(y7_abc, y7_ang, y7_pos, t7)
            nfe += 1

            # 5th-order solution
            new_abc_5 = abc + h * (
                b1 * k1_abc + b2 * k2_abc + b3 * k3_abc +
                b4 * k4_abc + b5 * k5_abc + b6 * k6_abc + b7 * k7_abc
            )
            new_ang_5 = angles + h * (
                b1 * k1_ang + b2 * k2_ang + b3 * k3_ang +
                b4 * k4_ang + b5 * k5_ang + b6 * k6_ang + b7 * k7_ang
            )
            new_pos_5 = pos + h * (
                b1 * k1_pos + b2 * k2_pos + b3 * k3_pos +
                b4 * k4_pos + b5 * k5_pos + b6 * k6_pos + b7 * k7_pos
            )

            # 4th-order embedded
            new_abc_4 = abc + h * (
                b1_hat * k1_abc + b2_hat * k2_abc + b3_hat * k3_abc +
                b4_hat * k4_abc + b5_hat * k5_abc + b6_hat * k6_abc + b7_hat * k7_abc
            )
            new_ang_4 = angles + h * (
                b1_hat * k1_ang + b2_hat * k2_ang + b3_hat * k3_ang +
                b4_hat * k4_ang + b5_hat * k5_ang + b6_hat * k6_ang + b7_hat * k7_ang
            )
            new_pos_4 = pos + h * (
                b1_hat * k1_pos + b2_hat * k2_pos + b3_hat * k3_pos +
                b4_hat * k4_pos + b5_hat * k5_pos + b6_hat * k6_pos + b7_hat * k7_pos
            )

            # Error estimate
            err_abc = new_abc_5 - new_abc_4
            err_ang = new_ang_5 - new_ang_4
            err_pos = new_pos_5 - new_pos_4

            # scale for all components
            scale_abc = atol + rtol * torch.max(abc.abs(), new_abc_5.abs())
            scale_ang = atol + rtol * torch.max(angles.abs(), new_ang_5.abs())
            scale_pos = atol + rtol * torch.max(pos.abs(), new_pos_5.abs())

            err_abc_norm = (err_abc.abs() / scale_abc).amax()
            err_ang_norm = (err_ang.abs() / scale_ang).amax()
            err_pos_norm = (err_pos.abs() / scale_pos).amax()

            err_norm = torch.stack([err_abc_norm, err_ang_norm, err_pos_norm]).amax()
            err_norm = float(err_norm)

            # Accept / reject
            if err_norm <= 1.0 or h <= min_h:
                # accept
                t = t + h
                abc, angles, pos = new_abc_5, new_ang_5, new_pos_5
                abc, angles, pos = clamp_state(abc, angles, pos)

                # propose new step size
                if err_norm == 0.0:
                    factor = max_factor
                else:
                    # p = 4 for embedded (error ~ h^(p+1)), so exponent = -1/(p+1) = -1/5
                    factor = safety * (err_norm ** (-1.0 / 5.0))
                    factor = max(min_factor, min(max_factor, factor))
                h = max(min_h, h * factor)
            else:
                # reject: shrink h, retry
                factor = safety * (err_norm ** (-1.0 / 5.0))
                factor = max(min_factor, min(1.0, factor))
                h = max(min_h, h * factor)

        tools.report_progress(max_nfe, max_nfe, progress_callback, cancel_check)
        # Final projection to physical domain
        cell_lengths = (torch.exp(abc) - EPS).clamp_min(0.0)
        angles_out = angles.clamp(math.pi / 3, 2 * math.pi / 3 - EPS)
        pos_out = pos.clamp(0.0, 1.0)

        return cell_lengths, angles_out, pos_out

    @torch.no_grad()
    def sample_rk4(self, z, atomic, mask, n_steps: int = 250, *, progress_callback=None, cancel_check=None):

        device = atomic.device
        B = atomic.shape[0]

        # Initial state (log-lengths, angles, positions)
        abc, angles, pos = self.initial_noise(mask)

        # Time and step size
        t = torch.zeros(B, device=device, dtype=atomic.dtype)  # shape (B,)
        h = 1.0 / float(n_steps)

        def eval_f(abc, angles, pos, t_vec):
            """Velocity field with inputs clamped to physical domain."""
            angles_in = angles.clamp(0.0, math.pi - EPS)
            pos_in = pos.clamp(0.0, 1.0)

            v_abc, v_angles, v_pos = self(
                z, t_vec, abc, angles_in, atomic, pos_in, mask
            )
            return v_abc, v_angles, v_pos

        for step in range(n_steps):
            tools.report_progress(step, n_steps, progress_callback, cancel_check)
            # k1
            k1_abc, k1_ang, k1_pos = eval_f(abc, angles, pos, t)

            # k2
            t2 = t + 0.5 * h
            abc2 = abc + 0.5 * h * k1_abc
            ang2 = angles + 0.5 * h * k1_ang
            pos2 = pos + 0.5 * h * k1_pos
            k2_abc, k2_ang, k2_pos = eval_f(abc2, ang2, pos2, t2)

            # k3
            t3 = t + 0.5 * h
            abc3 = abc + 0.5 * h * k2_abc
            ang3 = angles + 0.5 * h * k2_ang
            pos3 = pos + 0.5 * h * k2_pos
            k3_abc, k3_ang, k3_pos = eval_f(abc3, ang3, pos3, t3)

            # k4
            t4 = t + h
            abc4 = abc + h * k3_abc
            ang4 = angles + h * k3_ang
            pos4 = pos + h * k3_pos
            k4_abc, k4_ang, k4_pos = eval_f(abc4, ang4, pos4, t4)

            # RK4 update
            abc = abc + (h / 6.0) * (k1_abc + 2.0 * k2_abc + 2.0 * k3_abc + k4_abc)
            angles = angles + (h / 6.0) * (k1_ang + 2.0 * k2_ang + 2.0 * k3_ang + k4_ang)
            pos = pos + (h / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)

            angles = angles.clamp(math.pi / 3 + EPS, 2 * math.pi / 3- EPS)
            pos = pos.clamp(EPS, 1-EPS)

            # Advance time (shared across batch)
            t = t + h

        tools.report_progress(n_steps, n_steps, progress_callback, cancel_check)
        # Final projection to physical domain
        cell_lengths = (torch.exp(abc) - EPS).clamp_min(0.0)
        angles_out = angles.clamp(math.pi / 3 + EPS, 2 * math.pi / 3- EPS)
        pos_out = pos.clamp(EPS, 1-EPS)

        return cell_lengths, angles_out, pos_out


#
# FlatFlow was used in Ablations
#
class FlatFlow(Flow):

    def __init__(self, transformer_dim, n_blocks, n_heads, n_cliques, clique_dim, knot_dim, n_registers=0, dropout_rate=0.1, submodule=ops.SwiGLU, act=nn.GELU(), initial_length_dist=None, offset_by_atoms=True):

        super().__init__(transformer_dim, n_blocks, n_heads, n_cliques, clique_dim, knot_dim, n_registers=0, dropout_rate=0.1, submodule=ops.SwiGLU, act=nn.GELU(), initial_length_dist=None, offset_by_atoms=True)

        self.transformer = backbones.Transformer(transformer_dim, n_blocks, n_heads, dropout_rate, submodule, act)


    def forward(self, z, t, abc, angles, atomic, pos, mask):

        #
        # Make the time into HD space
        #
        t_emb = self.time_emb(t)
        t = t.view(-1, 1, 1)

        #
        # Create the mask of valid atoms and filter out the invalid ones
        #
        keep = tools.true_atom_mask(mask).unsqueeze(-1)
        pos = pos * keep 

        #
        # Embed molecular information in HD spaces
        #
        abc = self.abc_emb(abc)
        angles = self.angle_emb(angles)
        pos = self.pos_emb(pos) 

        #
        # Merge the embeddings of geometry information 
        #
        x = tools.into_structure_tensor(abc, angles, pos) 
        mask =  F.pad(mask, (2, 0), value=1)
        
        #
        # Merge type and time embeddings
        #
        emb = torch.cat([self.lattice_emb.repeat(abc.shape[0], 1, 1), atomic], dim=1)
        t_emb = t_emb.repeat(1, emb.size(1), 1)

        #
        # Process with a transformer (THIS IS THE KEY DIFFERENCE BETWEEN THE ABOVE FLOW MODEL)
        #
        x = self.transformer(x + z, emb, t_emb, mask = mask)

        #
        # Separate the geometry information and process independently
        #
        abc, angles, pos = tools.from_structure_tensor(x)

        abc = self.abc_from_model(abc)
        angles = self.angles_from_model(angles)
        pos = self.pos_from_model(pos)

        return abc, angles, pos
