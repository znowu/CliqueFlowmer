# CliqueFlowmer crystal workspace

A custom React website for a single uploaded CIF: encode its 121-dimensional
representation, reconstruct a structure, or optimize the latent model objective.
The interface includes two interactive 3D viewers, original/optimized clique
heatmaps, a score trajectory, cancellation, and CIF, NumPy, and JSON downloads.

The dark navy, cyan, and copper design is responsive and supports keyboard
navigation and reduced motion. The 3Dmol viewer is bundled locally with its
license; uploaded structures are never sent to a third-party viewer.

## Architecture and cost

`site/` is the React/Vinext website, hosted on Sites. Its server-side `/api/*`
proxy forwards requests to `api.py`, a single-process Starlette CPU service.
The proxy key stays on the server. The API reuses `serving/web_demo/service.py`
and the same revision- and SHA-pinned public checkpoint. No GPU, MatGL, DGL,
training dataset, or paid inference API is needed.

The initial hosted preview runs the CPU service on **grace**, reached through a
free Cloudflare Quick Tunnel. This is a demonstration deployment: grace, the
API process, and the tunnel must remain running. Quick Tunnel URLs change on
restart and have no uptime guarantee. They are not a permanent hosting service.
The website's URL remains stable; its backend origin must be updated and the
website redeployed when the tunnel URL changes. A stable public deployment can
use a named tunnel or another CPU host with a stable HTTPS origin.

Site ID: `appgprj_6ae1967d9c0081919c8ea55b2d2fc68f`

Website: <https://cliqueflowmer-lab.loyalcow26.chatgpt.site>

The Site starts as an owner-private preview. Publishing a version does not
change its audience. Public access is a separate Sites sharing setting.

## CPU service

Use Python 3.11 and install the website requirements from the repository root:

```bash
python3.11 -m venv .venv-website
.venv-website/bin/python -m pip install -r serving/website/requirements.txt
```

Set `CLIQUEFLOWMER_PROXY_KEY` to a strong random secret and
`CLIQUEFLOWMER_PUBLIC_ORIGIN` to the website's exact HTTPS origin. Then run:

```bash
.venv-website/bin/python -m serving.website.run
```

The default listener is `127.0.0.1:7860`; `HOST` and `PORT` override it.
Alternatively, pass `--config /path/to/private-runtime.json` with these string
environment values. Keep that file outside source control with mode 0600.
`CLIQUEFLOWMER_CHECKPOINT` can select an already downloaded copy of the pinned
checkpoint. Otherwise, the first operation downloads it from Hugging Face.

The current machine's private configuration is in the ignored
`.cache-demo/website-runtime.json`. It can be restarted with:

```bash
.venv-demo-clean/bin/python -m serving.website.run --config .cache-demo/website-runtime.json
.cache-demo/bin/cloudflared tunnel --url http://127.0.0.1:7860 --protocol http2
```

Do not expose a service launched without the proxy key. `run.py` requires it.
The API binds to loopback by default and does not log uploaded CIFs or cookies.

## Website

Use Node.js 22.13 or later, then from `serving/website/site`:

```bash
npm ci
npm run dev
```

Put local settings in ignored `.dev.vars` and production settings in Sites:

| Variable | Meaning |
| --- | --- |
| `CLIQUEFLOWMER_API_ORIGIN` | HTTPS CPU-service origin, without a path |
| `CLIQUEFLOWMER_API_KEY` | Secret matching the API's `CLIQUEFLOWMER_PROXY_KEY` |
| `CLIQUEFLOWMER_PUBLIC_ORIGIN` | Same exact origin configured on the API |

The local proxy validates the browser's local origin and supplies the configured
public origin upstream. It forwards only the demo session cookie, never Sites
authentication cookies. Redirects from the CPU service are rejected.

On grace's older Ubuntu installation, the current Cloudflare preview runtime
needs a newer glibc. An isolated copy is available in `.cache-demo/glibc`; set
`MINIFLARE_WORKERD_PATH` to `.cache-demo/bin/workerd-local` (absolute path).
Node 22 is also available in `.venv-demo/lib/python3.11/site-packages/playwright/driver`.
These are local compatibility aids, not production dependencies.

The canonical website source stays in `site/` within this GitHub repository.
The Sites publishing helper uses a separate checkout at
`.cache-demo/website-publish/` so it cannot alter the main repository's Git
history. Copy changed source into that checkout, excluding Git metadata,
dependencies, builds, secrets, and local tool state; then follow the Sites
source/build/package/deploy workflow with the existing Site ID.

## Scientific and resource limits

- One ordered CIF, up to 100 KB and 20 sites in the uploaded cell.
- One CPU job at a time; concurrent submissions receive a clear busy message.
- Quick/Extended optimization uses 100/500 ES steps, followed by 1,000 flow
  steps, with at most one retry for invalid generated geometry.
- The checkpoint lacks target scaling metadata. Displayed scores are **raw
  latent predictor outputs**, not calibrated eV/atom or evaluated candidate
  energies. Generated structures are unrelaxed and may change composition.
- Jobs have a cooperative five-minute deadline. Cancellation reaches the ES
  and decoding loops. Results are session-private, memory-only, and expire
  after 30 minutes; at most 32 sessions retain results.
- Download JSON records the checkpoint identity, seed, parameters, scores,
  trajectory, input/output CIFs, and representations. NumPy downloads contain
  float32 arrays and can be loaded with `allow_pickle=False`.

## Verification

The new HTTP tests cover session ownership, signature tampering, origin/key
checks, invalid input, body limits, busy responses, cancellation, expiration,
and bounded retention. Install `httpx==0.28.1` for the test client and run:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The complete model/demo/API suite passes 56 tests. Browser checks use the real
CPU checkpoint, including upload, encode, reconstruction, optimization, JSON /
NumPy / CIF downloads, cancellation, invalid input, and mobile layout.

Third-party hosting details: [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).
