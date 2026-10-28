# CliqueFlowmer CPU demo

A single-CIF Streamlit application: upload or paste an ordered crystal, inspect
its latent representation, reconstruct it, or optimize its latent and view the
generated candidate. It runs on CPU and uses no paid inference API.

## Run locally

Use Python **3.11**, from the repository root:

```bash
python3.11 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r serving/web_demo/requirements.txt
python -m pip check
python -m streamlit run serving/web_demo/app.py
```

The first operation downloads a 65 MB published checkpoint from Hugging Face.
The revision and SHA-256 are pinned in `checkpoint.json`; the checkpoint is
verified before loading with `weights_only=True`. The original length prior is
distributed here as JSON, so the public demo does not unpickle a prior.

For offline testing, set `CLIQUEFLOWMER_CHECKPOINT` to the downloaded `.pth`.
The same checksum is required. `CLIQUEFLOWMER_THREADS=1` or `2` controls CPU
thread count. Model selection is intentionally fixed for this small demo.

## Free public deployment

Deploy on [Streamlit Community Cloud](https://share.streamlit.io/):

1. Sign in with the GitHub account that administers this repository.
2. Create an app, select the repository and branch containing these files.
3. Set the entrypoint to **`serving/web_demo/app.py`**.
4. In Advanced settings, choose **Python 3.11**. No secrets, API keys, GPU or
   paid account are required for this public app.
5. Deploy and run one example through optimization and download to check the
   actual host's available memory and performance.

Community Cloud finds the `requirements.txt` beside the entrypoint before the
root training requirements. Do not install the root requirements for this app.
The service binds through Streamlit's default configuration; `.streamlit/config.toml`
sets the theme, disables telemetry, and limits uploads at the HTTP layer.

Community Cloud is shared free hosting. Its resource allocation is variable;
apps sleep after inactivity. Local benchmark times are not a hosted-service
latency guarantee. No paid fallback or automatic resource upgrade is configured.
See [deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
and [resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

## Computation and interpretation

- Inputs: one CIF data block, at most 100 KB and **1–20 fully occupied sites**
  after symmetry expansion. The uploaded cell is retained; supercells are not
  silently reduced. Partial occupancy and disordered sites are rejected.
- Encoding returns the posterior mean: 121 coordinates arranged into eight
  16-coordinate cliques, with neighboring cliques sharing one coordinate.
- Quick uses **100 ES steps**; Standard uses **500**. Both retain the existing
  perturbation count (20 antithetic pairs), scale (0.05), learning-rate setting
  (3e-4), rank transform and weight decay (0.4).
- Decoding retains **1,000 CFG steps, guidance 2, and beam width 10**. The demo
  bounds generated structures to 1–20 sites. Exact beam-search early stopping
  avoids expanding prefixes that cannot beat the completed answer.
- Only the final latent is decoded. If its first geometry fails validation, the
  demo samples **one** additional geometry at that same latent, continuing the
  seeded RNG stream. The first valid candidate is returned. Both failures
  produce an honest error, with no substituted example or fabricated result.
- Generated structures are **unrelaxed**. MatGL, DGL and M3GNet are absent from
  the demo runtime. Optional research relaxation remains available separately
  through `serving.inference` with its extra dependencies.
- Scores come from the checkpoint's **latent target regressor**, not an
  independent evaluation of the decoded crystal. The published formation-energy
  checkpoint lacks target-scaling metadata, so scores are labeled **model score**,
  not eV/atom. `checkpoint.json` records this limitation. A lower score does not
  establish physical stability. Composition and site count can change.
- JSON downloads include the full latents, score history, checkpoint hash,
  seed, decoder settings, retry count and the input hash. NPY downloads contain
  numeric arrays only, with no pickle serialization.

## Resource bounds and privacy

One model is cached per app process and only one job runs at a time. Another
visitor receives a busy message instead of growing a queue. A five-minute
cooperative timeout and cancellation checks cover optimization and decoding.
Model download/loading and individual parser/tensor operations cannot be
interrupted mid-call. There is no background training or user job registry.

Uploads and outputs are held in the visitor's server-side Streamlit session;
the application does not write them to disk or publish them. Results expire
from the UI after 30 minutes when the page next updates, or when input changes.
This is session cleanup, not a guarantee of immediate physical memory erasure.
The embedded crystal viewer downloads a pinned 3Dmol.js script from jsDelivr;
it renders validated atom coordinates in the browser. Paper and model links
are external links, not additional inference services.

## Validation

```bash
python -m pip install pytest==9.2.0
python -m pytest tests -q
python -m serving.web_demo.benchmark
python -m serving.web_demo.benchmark --operation reconstruct --output /tmp/reconstruction.json
```

The benchmark uses real published weights and the three selected example
crystals (Si, SrTiO3, and Cs2AgBiBr6), recording per-case success/failure, elapsed time, process peak RSS,
objective values, retry count, package versions and checkpoint provenance.
Example CIFs are constructed from conventional textbook geometries; they are
demonstration inputs, not new experimental or DFT results.

The historical local benchmark below used Si, SrTiO3, and Cu before Cs2AgBiBr6
replaced Cu in the UI. Local validation on 2026-10-27 used the pinned CPU-only stack, two PyTorch threads,
seed 42, 100 ES steps and 1,000 CFG steps:

| Input | Optimization + decoding | Decode attempts | Process peak RSS |
|---|---:|---:|---:|
| SrTiO3, 5 sites | 12.3 s | 1 | 494 MiB |
| Si, 2 sites | 23.0 s | 2 | 494 MiB |
| Cu, 4 sites | 12.3 s | 1 | 494 MiB |

All three also completed reconstruction. A 20-site SrTiO3 supercell completed
the 500-step Standard optimization and returned a valid 20-site candidate in
19.4 s. These are individual local smoke tests, not a success-rate study or a
measurement on Streamlit's shared host; RSS excludes the separate browser.
