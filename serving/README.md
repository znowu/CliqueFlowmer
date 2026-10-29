# Standalone research inference

The Python helpers encode crystals, reconstruct structures, and optimize their
latent representations. They run independently of the hosted
[interactive demo](https://cliqueflowmer.znowu.workers.dev).

| Module | Purpose |
| --- | --- |
| [pretrained.py](pretrained.py) | Download or reuse the checksum-pinned public checkpoint |
| [inference.py](inference.py) | Encoding, decoding, reconstruction, and latent optimization |
| [data_tools.py](data_tools.py) | Structure conversion and length-prior helpers |
| [relaxation.py](relaxation.py) | Optional research relaxation using a separate MatGL/ASE stack |

Install the core inference environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Pretrained checkpoint

`load_pretrained_model()` uses the exact Hugging Face revision and SHA-256 in
[checkpoint.json](checkpoint.json), with the matching bundled
[length_prior.json](length_prior.json). Its default device is CPU. For offline
reuse, pass `checkpoint_path="/path/to/checkpoint.pth"` or set
`CLIQUEFLOWMER_CHECKPOINT`; local copies must pass the same checksum check.

This convenience checkpoint was pretrained on **Alexandria and fine-tuned on
MP20 for formation-energy optimization**. It is distinct from the paper's
MP20-only experimental checkpoints. The released artifact lacks target-scaling
metadata, so predictor outputs are **raw model scores, not calibrated eV/atom**
or independently evaluated energies of decoded crystals.

## Encode, reconstruct, and optimize

Run this example from the repository root. The three example CIFs are in
[`examples/`](../examples/). Small CPU workloads often benefit from a small
thread pool; the library leaves that setting to the caller.

```python
from pathlib import Path

import numpy as np
import torch

from serving.pretrained import load_pretrained_model
from serving.inference import (
    structure_from_cif_text, structure_to_cif_text,
    encode_structure, reconstruct_structure, optimize_es, decode_latent,
)

torch.set_num_threads(2)
torch.manual_seed(42)
model = load_pretrained_model(device="cpu")
crystal = structure_from_cif_text(Path("examples/silicon.cif").read_text())

latent = encode_structure(model, crystal)
np.save("representation.npy", latent.detach().cpu().numpy())

decode_settings = dict(
    relax=False, n_steps=1000, beam_width=10, min_atoms=1, max_atoms=20,
)
_, reconstruction = reconstruct_structure(model, crystal, **decode_settings)
Path("reconstruction.cif").write_text(structure_to_cif_text(reconstruction))

optimized_latent, raw_score = optimize_es(
    model, latent, steps=100, weight_decay=0.4,
)
candidate = decode_latent(model, optimized_latent, **decode_settings)
Path("optimized.cif").write_text(structure_to_cif_text(candidate))
print("Optimized latent objective (raw model score):", raw_score.item())
```

Encoding is the quickest operation; each reconstruction or optimized decoding
uses 1,000 flow-integration steps and may take minutes on CPU. The 100 ES steps
above provide a short example, not a reproduction of the paper's full search.
Generated structures are unrelaxed and may change composition; they require
independent physical evaluation. Some latents may fail to decode a valid
structure, in which case the helper raises `ValueError`.

For another published model or a research checkpoint, use
`load_model_from_hf()` or `load_model_from_checkpoint()` from `inference.py`
with the matching model configuration and length prior. Optional relaxation
requires the separate oracle environment described in
[research setup](../docs/research.md).
