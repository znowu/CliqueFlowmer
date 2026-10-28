import torch
import torch.nn.functional as F
from data.constants import atomic_numbers, atomic_symbols


def report_progress(step, total, progress_callback=None, cancel_check=None, every=10):
    """Report bounded work and cooperatively cancel without touching RNG state."""
    if step == 0 or step >= total or step % every == 0:
        if cancel_check is not None:
            cancel_check()
        if progress_callback is not None:
            progress_callback(min(step, total), total)


def get_device(model):
    return list(model.parameters())[0].device


def rank(x, dim):

    x = torch.argsort(x, dim=dim)
    x = torch.argsort(x, dim=dim)

    return x.float()


def standardize(x, dim):

    x = x - torch.mean(x, dim=dim, keepdim=True)
    x = x / torch.std(x, dim=dim, keepdim=True)
    return x 


def center(x, dim):
    x_range = x.shape[dim] - 1
    x = x/x_range
    x -= 0.5
    return x 


def masked_sum(x, mask, dim=-1):

    if len(mask.shape) < len(x.shape):
        
        more = len(x.shape) - len(mask.shape) 
        mask = mask.reshape(mask.shape + more * (1,))

    return (x * mask).sum(dim)


def masked_mean(x, mask, dim=-1, sqrt=False):

    if len(mask.shape) < len(x.shape):
        
        more = len(x.shape) - len(mask.shape) 
        mask = mask.reshape(mask.shape + more * (1,))

    sums = mask.sum(dim=dim, keepdim=True)
    sums = torch.sqrt(sums) if sqrt else sums

    return (x * mask / sums).sum(dim)


def masked_geo_sum(x, mask, decay=0.95, dim=-1):

    if len(mask.shape) < len(x.shape):
    
        more = len(x.shape) - len(mask.shape) 
        mask = mask.reshape(mask.shape + more * (1,))

    weights = (decay * mask).cumprod(dim=dim) / decay

    return (x * weights).sum(dim)


def from_structure_tensor(x):

    abc = x[:, 0, :]
    angles = x[:, 1, :]
    pos = x[:, 2:, :]

    return abc, angles, pos 


def into_structure_tensor(abc, angles, pos):

    abc = abc.unsqueeze(-2)
    angles = angles.unsqueeze(-2)

    return torch.cat([abc, angles, pos], dim=-2)


def masked_into_structure_tensor(abc, angles, pos, mask):

    B, N = mask.shape
    device = pos.device
    dtype  = pos.dtype

    #
    # Identify valid (true) atoms: mask minus start (0) and end (N_atoms-1)
    #
    N_atoms = mask.sum(dim=-1).long().clamp_min(0)       

    start_oh = F.one_hot(torch.zeros(B, dtype=torch.long, device=device), num_classes=N).to(dtype)  
    end_idx  = (N_atoms - 1).clamp_min(0)                  
    end_oh   = F.one_hot(end_idx, num_classes=N).to(dtype) 

    valid = mask * (1.0 - start_oh) * (1.0 - end_oh)      

    #
    # Zero-out non-true atoms in positions (broadcast on last dim)
    #
    pos_masked = pos * valid.unsqueeze(-1)                 

    #
    # Pack back into structure tensor: (B, 1+1+N, 3)
    #
    abc = abc.unsqueeze(-2)                           
    angles = angles.unsqueeze(-2)                        
    return torch.cat([abc, angles, pos_masked], dim=-2)


def r2(preds, targets, eps=1e-8):

    mse = ((preds.view(-1) - targets.view(-1))**2).mean()
    var = ((targets.view(-1) - targets.mean())**2).mean()
    
    return 1 - mse / (var + eps)


def fast_polyak(updatable_model, new_model, tau=0.005):

    one = torch.ones(1, requires_grad=False).to(get_device(updatable_model))

    for param, target_param in zip(
        new_model.parameters(), updatable_model.parameters()
    ):
        target_param.data.mul_(1 - tau)
        target_param.data.addcmul_(param.data, one, value=tau)


def beam_search(model, z, beam_width=5, max_atoms=4, min_atoms=6, temperature=1.0):
    """
    Beam search for generating atomic structures with CliqueFlowmer model.
    """
    device = z.device
    
    #
    # Initialize beam with start token
    #
    start_token = torch.zeros(1, 1, dtype=torch.long, device=device)
    start_emb = model.atomic_emb(start_token)
    
    #
    # Get vocabulary size from atomic embedding layer
    #
    vocab_size = len(atomic_numbers)
    
    #
    # Create end token
    #
    end = vocab_size - 1  # Assuming end token is last in vocabulary
    end_token = end * torch.ones(1, 1, dtype=torch.long, device=device)
    
    #
    # Active beam contains: (tokens, embeddings, log_prob)
    #
    active_beam = [(start_token, start_emb, 0.0)]
    best_completed = None
    best_score = float('-inf')
    
    for step in range(max_atoms - 1):
        candidates = []
        
        for tokens, token_embs, score in active_beam:
            #
            # Create mask for current sequence
            #
            mask = torch.ones((1, tokens.size(1)), device=device)
            
            #
            # Get predictions from decoder
            #
            log_probs = model.decoder(z, token_embs, mask)[0, -1] / temperature
            
            #
            # Option 1: End sequence if allowed
            #
            if tokens.size(1) >= min_atoms:
                end_score = score + log_probs[end].item()
                if end_score > best_score:
                    best_completed = torch.cat([tokens, end_token], dim=1)
                    best_score = end_score
            
            #
            # Option 2: Continue sequence
            #
            if tokens.size(1) < max_atoms:
                #
                # Get top-k tokens (excluding end token)
                #
                topk_log_probs, topk_tokens = torch.topk(
                    log_probs[:-1],
                    k=min(beam_width, log_probs.size(-1)-1)
                )
                
                for token_log_prob, token in zip(topk_log_probs, topk_tokens):
                    new_token = token.unsqueeze(0).unsqueeze(0)
                    new_tokens = torch.cat([tokens, new_token], dim=1)
                    new_token_emb = model.atomic_emb(new_token)
                    new_token_embs = torch.cat([token_embs, new_token_emb], dim=1)
                    new_score = score + token_log_prob.item()
                    candidates.append((new_tokens, new_token_embs, new_score))
        
        if not candidates:
            break
        
        #
        # Keep top beam_width candidates
        #
        candidates.sort(key=lambda x: x[2], reverse=True)
        active_beam = candidates[:beam_width]
    
    #
    # If no sequence completed, take best active sequence and end it
    #
    if best_completed is None and active_beam:
        tokens, token_embs, _ = active_beam[0]
        best_completed = torch.cat([tokens, end_token], dim=1)
    
    return best_completed.squeeze(0).long() if best_completed is not None else None


@torch.no_grad()
def batched_beam_search(
    model,
    z,
    beam_width=6,
    max_atoms=100,
    min_atoms=3,
    temperature=1.0,
    *,
    progress_callback=None,
    cancel_check=None,
    early_stop=True,
):
    """
    Batched beam search for generating atomic structures with CliqueFlowmer model.
    z: conditioning variable of shape [batch_size, ...]
    Returns: list of 1D LongTensors (token sequences) for each batch element, or None.
    """
    if not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer.")
    if not isinstance(min_atoms, int) or not isinstance(max_atoms, int) or not 1 <= min_atoms <= max_atoms:
        raise ValueError("Atom limits must satisfy 1 <= min_atoms <= max_atoms.")
    if not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive.")
    device = z.device
    batch_size = z.size(0)

    # Vocab / special tokens
    vocab_size = len(atomic_numbers)
    start_id = 0
    end_id = vocab_size - 1

    # Beam state per sample: lists of (tokens, score), tokens are 1D LongTensors
    active_tokens = [[torch.tensor([start_id], device=device, dtype=torch.long)]
                     for _ in range(batch_size)]
    active_scores = [[0.0] for _ in range(batch_size)]

    # Best completed sequence per sample
    best_completed = [None for _ in range(batch_size)]
    best_scores = [float('-inf') for _ in range(batch_size)]

    # One final iteration allows EOS after exactly max_atoms real atoms.
    total_steps = max_atoms + 1
    for step in range(total_steps):
        report_progress(step, total_steps, progress_callback, cancel_check, every=1)
        any_valid = False

        next_active_tokens = [[] for _ in range(batch_size)]
        next_active_scores = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            beams_tokens_i = active_tokens[i]
            beams_scores_i = active_scores[i]

            # No active beams for this sample
            if not beams_tokens_i:
                continue

            # Every future log-probability increment is <= 0. Once no active
            # prefix can beat the completed sequence, further expansion cannot
            # change the answer (including ties, which never replace it).
            if early_stop and best_completed[i] is not None and max(beams_scores_i) <= best_scores[i]:
                continue

            num_beams = len(beams_tokens_i)
            cur_len = beams_tokens_i[0].size(0)  # all beams same length

            # Stack tokens: [num_beams, cur_len]
            tokens_tensor = torch.stack(beams_tokens_i, dim=0)  # device = z.device
            scores_tensor = torch.tensor(beams_scores_i, device=device)

            # Expand conditioning z for each beam: [num_beams, ...]
            z_i = z[i:i+1].expand(num_beams, *z.shape[1:])

            # Forward all beams at once
            token_embs = model.atomic_emb(tokens_tensor)            # [num_beams, cur_len, emb_dim]
            mask = torch.ones((num_beams, cur_len), device=device)  # [num_beams, cur_len]
            logits = model.decoder(z_i, token_embs, mask)           # [num_beams, cur_len, vocab]
            log_probs = logits[:, -1, :] / temperature              # [num_beams, vocab]

            # Option 1: end sequence (only if enough atoms)
            if cur_len >= 1 + min_atoms:
                end_log_probs = log_probs[:, end_id]                # [num_beams]
                end_scores = scores_tensor + end_log_probs          # [num_beams]
                best_end_score, best_idx = end_scores.max(dim=0)

                if best_end_score.item() > best_scores[i]:
                    best_scores[i] = best_end_score.item()
                    best_tokens = beams_tokens_i[best_idx.item()]
                    best_completed[i] = torch.cat(
                        [best_tokens,
                         torch.tensor([end_id], device=device, dtype=torch.long)],
                        dim=0
                    )

            # Option 2: continue sequence (if not too long)
            if cur_len - 1 < max_atoms:
                # BOS is never a real atom; EOS was handled explicitly above.
                log_probs_trunc = log_probs[:, 1:-1]

                k = min(beam_width, log_probs_trunc.size(-1))
                topk_log_probs, topk_tokens = torch.topk(
                    log_probs_trunc, k=k, dim=-1
                )  # both [num_beams, k]
                topk_tokens = topk_tokens + 1

                cand_tokens = []
                cand_scores = []

                # Build candidate beams: all parents × top-k children
                for b in range(num_beams):
                    base_tokens = beams_tokens_i[b]
                    base_score = beams_scores_i[b]
                    for j in range(k):
                        tok_id = topk_tokens[b, j].item()
                        new_token = torch.tensor([tok_id], device=device, dtype=torch.long)
                        new_seq = torch.cat([base_tokens, new_token], dim=0)
                        new_score = base_score + topk_log_probs[b, j].item()
                        cand_tokens.append(new_seq)
                        cand_scores.append(new_score)

                if cand_tokens:
                    any_valid = True
                    cand_scores_tensor = torch.tensor(cand_scores, device=device)
                    k_keep = min(beam_width, len(cand_tokens))
                    _, top_idx = torch.topk(cand_scores_tensor, k=k_keep, dim=0)
                    chosen = top_idx.tolist()

                    next_active_tokens[i] = [cand_tokens[idx] for idx in chosen]
                    next_active_scores[i] = [cand_scores[idx] for idx in chosen]
                else:
                    # Strict equivalence with original: keep current beams
                    next_active_tokens[i] = beams_tokens_i
                    next_active_scores[i] = beams_scores_i
            else:
                # Cannot extend further; keep current beams as in original implementation
                next_active_tokens[i] = beams_tokens_i
                next_active_scores[i] = beams_scores_i

        if not any_valid:
            break

        active_tokens = next_active_tokens
        active_scores = next_active_scores

    # Finalize sequences
    results = []
    for i in range(batch_size):
        seq = best_completed[i]

        # If no completed sequence, fall back to best active beam (if any)
        if seq is None:
            beams_tokens_i = active_tokens[i]
            beams_scores_i = active_scores[i]

            if beams_tokens_i:
                best_idx = max(range(len(beams_scores_i)), key=lambda j: beams_scores_i[j])
                best_seq = beams_tokens_i[best_idx]
                seq = torch.cat(
                    [best_seq, torch.tensor([end_id], device=device, dtype=torch.long)],
                    dim=0
                )
            else:
                results.append(None)
                continue

        results.append(seq.long())  # 1D LongTensor

    report_progress(total_steps, total_steps, progress_callback, cancel_check, every=1)
    if z.is_cuda:
        torch.cuda.empty_cache()

    return results


"""
@torch.no_grad()
def batched_beam_search(model, z, beam_width=6, max_atoms=100, min_atoms=3, temperature=1.0):

    device = z.device
    batch_size = z.size(0)
    
    #
    # Initialize beams for each sample in batch with start tokens
    #
    start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
    start_emb = model.atomic_emb(start_token)
    
    #
    # Get vocabulary size and create end tokens
    #
    vocab_size = len(atomic_numbers)
    end = vocab_size - 1
    end_token = end * torch.ones(1, 1, dtype=torch.long, device=device)
    
    #
    # Initialize separate beam state for each sample in batch
    # Each beam contains (tokens, embeddings, score)
    #
    active_beams = []
    for i in range(batch_size):
        active_beams.append([(
            start_token[i:i+1], 
            start_emb[i:i+1], 
            0.0
        )])
    
    #
    # Track best completed sequence and score for each sample
    #
    best_completed = [None for _ in range(batch_size)]
    best_scores = [float('-inf') for _ in range(batch_size)]
    
    for step in range(max_atoms):
        #
        # Collect candidates for each sample
        #
        all_candidates = [[] for _ in range(batch_size)]
        any_valid = False
        
        for i in range(batch_size):
            for tokens, token_embs, score in active_beams[i]:
                #
                # Create mask and get predictions
                #
                mask = torch.ones((1, tokens.size(1)), device=device)
                log_probs = model.decoder(z[i:i+1], token_embs, mask)[0, -1] / temperature
                
                #
                # Try ending sequence if long enough
                #
                if tokens.size(1) >= 1 + min_atoms:
                    end_score = score + log_probs[end].item()
                    if end_score > best_scores[i]:
                        best_completed[i] = torch.cat([tokens, end_token], dim=1)
                        best_scores[i] = end_score
                
                #
                # Continue sequence if not too long
                #
                if tokens.size(1) - 1 < max_atoms:
                    #
                    # Get top-k tokens (excluding start & end token)
                    #
                    topk_log_probs, topk_tokens = torch.topk(
                        log_probs[:-1],
                        k=min(beam_width, log_probs.size(-1)-1)
                    )
                    
                    for token_log_prob, token in zip(topk_log_probs, topk_tokens):
                        new_token = token.view(1, 1)
                        new_tokens = torch.cat([tokens, new_token], dim=1)
                        new_token_emb = model.atomic_emb(new_token)
                        new_token_embs = torch.cat([token_embs, new_token_emb], dim=1)
                        new_score = score + token_log_prob.item()
                        all_candidates[i].append((new_tokens, new_token_embs, new_score))
                        any_valid = True
        
        #
        # Break if no valid candidates for any sample
        #
        if not any_valid:
            break
        
        #
        # Update active beams with top candidates for each sample
        #
        for i in range(batch_size):
            if all_candidates[i]:
                all_candidates[i].sort(key=lambda x: x[2], reverse=True)
                active_beams[i] = all_candidates[i][:beam_width]
    
    #
    # Finalize any samples that haven't completed
    #
    for i in range(batch_size):
        if best_completed[i] is None and active_beams[i]:
            tokens, _, _ = active_beams[i][0]
            best_completed[i] = torch.cat([tokens, end_token], dim=1)
    
    #
    # Return list of sequences, removing batch dimension
    #
    best_completed = [seq.squeeze(0).long() if seq is not None else None 
            for seq in best_completed]

    torch.cuda.empty_cache()

    return best_completed
"""

def true_atom_mask(mask: torch.Tensor):
    # mask: (B, L) with 1 for tokens that exist (incl. special), 0 for padding
    B, L = mask.shape
    device = mask.device

    N_atoms = mask.sum(-1).int()  # number of tokens present per example

    start_oh = torch.zeros(B, L, device=device)
    start_oh[:, 0] = 1.0

    end_idx = (N_atoms - 1).clamp_min(0)
    end_oh = torch.zeros(B, L, device=device)
    end_oh[torch.arange(B, device=device), end_idx] = 1.0

    keep = mask * (1.0 - start_oh) * (1.0 - end_oh)
    return keep



