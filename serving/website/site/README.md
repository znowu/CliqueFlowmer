# CliqueFlowmer website

The public crystal workspace runs at
**https://cliqueflowmer.znowu.workers.dev**. This directory contains its React
interface, Vite build, and Cloudflare Worker API proxy. Model inference runs in
the separate [CPU service](../README.md).

Users can upload a CIF or choose an example, inspect its representation,
reconstruct or optimize a crystal, and download the candidate CIF and latent
vectors as CSV, NumPy, or JSON. Both crystal viewers support rotation, zoom,
reset, and automatic rotation. Decorative liquid motion can be paused and
respects reduced-motion preferences by default.

## Install and check

Use Node.js 22.13 or later:

```bash
npm ci
npm run typecheck
npm run build
npm run check:cloudflare
```

The build produces `dist/cloudflare/public/`; the last command bundles the Worker
locally without publishing. `npm run build:cloudflare` remains an alias for
`npm run build`.

## Local preview

Start the [CPU service](../README.md), then create the ignored
`cloudflare/.dev.vars`:

```dotenv
CLIQUEFLOWMER_API_ORIGIN="http://127.0.0.1:7860"
CLIQUEFLOWMER_API_KEY="<same proxy key as the CPU service>"
CLIQUEFLOWMER_PUBLIC_ORIGIN="<exact origin configured on the CPU service>"
```

Run `npm run dev` and open `http://127.0.0.1:5173`. This builds the frontend and
starts a local Worker so browser requests follow the same path as production.
After editing, run `npm run build` again; Wrangler watches the generated assets.
`npm run preview:cloudflare` previews an existing build.

## Source map

- `app/page.tsx`: crystal workspace and viewer controls.
- `app/globals.css`: responsive visual design.
- `components/`: latent comparison and liquid background.
- `lib/`: representation exports and liquid rendering.
- `cloudflare/`: entry point, static HTML, build and deployment configuration.
- `worker/model-proxy.ts`: bounded authenticated API proxy.
- `public/`: bundled 3Dmol.js, icons, and third-party notices.

Crystal rendering uses bundled 3Dmol.js 2.4.2; see
`public/3Dmol-LICENSE.txt`. Brand icon sources are recorded in
`public/brand/SOURCES.txt`.

## Publish

See [Cloudflare deployment](cloudflare/README.md). Runtime credentials belong in
Cloudflare secrets and local ignored files, never in browser code or Git.
`npm run deploy:cloudflare` builds and publishes; ordinary build/check commands
do not publish.
