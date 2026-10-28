# CliqueFlowmer · Computational Alchemy

Custom crystal-design workspace: upload a CIF, inspect its latent representation,
reconstruct it, optimize the learned objective, and download generated candidates.
Upload and Example structures are visible together. The representation view
compares original and optimized vectors as aligned rows on one color scale;
each vector can be downloaded as CSV, NumPy, or JSON.

## Run

Use Node.js >=22.13, `npm ci`, and `npm run dev`.
The UI requires a running CliqueFlowmer CPU API. Configure `.dev.vars` locally
or Sites environment variables in production:

- `CLIQUEFLOWMER_API_ORIGIN`: CPU service HTTPS origin.
- `CLIQUEFLOWMER_API_KEY`: server-only shared secret.
- `CLIQUEFLOWMER_PUBLIC_ORIGIN`: website origin configured on the CPU API.

The CPU backend and deployment instructions live in the main repository's
`serving/website/README.md`. This frontend's canonical source is
`serving/website/site/` in https://github.com/znowu/CliqueFlowmer.

`app/page.tsx` implements the workspace, `app/globals.css` its visual system,
and `worker/model-proxy.ts` the bounded authenticated API proxy. Crystal
visualization uses bundled 3Dmol.js 2.4.2 (see `public/3Dmol-LICENSE.txt`).

## Publish

Keep `.openai/hosting.json` and its existing Site ID. Use the Sites plugin's
source/build/package/deploy workflow. Runtime secrets belong in Sites, never
in the manifest or browser bundle. The production dispatcher enforces the
Site audience; the app supports private and anonymous-compatible public use.

The Vinext starter's `sites()` build integration, connector context, and
portable build helpers are retained. Local tool state and secrets are ignored.
