# Public Cloudflare website

This deploys the React UI as static HTML,
CSS, JavaScript, and the bundled crystal viewer, plus a small `/api/*` Worker
proxy. PyTorch stays on **grace**.

The approved public address is **https://cliqueflowmer.znowu.workers.dev**.
A `workers.dev` address needs no domain purchase. Keep the account on the **Workers Free** plan;
this target needs no database, storage subscription, or paid compute binding.

## Build and preview

From `serving/website/site`, using Node.js 22.13 or later and its existing npm
dependencies:

```bash
npm run build:cloudflare
npm run check:cloudflare
```

The second command bundles a local Worker dry run; it does not deploy.
The generated website is in `dist/cloudflare/public/`.

For an end-to-end local preview, put the following in the ignored
`cloudflare/.dev.vars`, using values from the CPU service's private configuration:

```dotenv
CLIQUEFLOWMER_API_ORIGIN="http://127.0.0.1:7860"
CLIQUEFLOWMER_API_KEY="<CPU service proxy key>"
CLIQUEFLOWMER_PUBLIC_ORIGIN="<exact origin configured on the CPU service>"
```

Then run `npm run preview:cloudflare` and visit `http://127.0.0.1:5173`.
On grace, the isolated runtime wrapper described in `../../README.md` is needed
for `MINIFLARE_WORKERD_PATH`. A current Linux installation needs no wrapper.

## Publish after the address is approved

1. Authenticate through `npx wrangler login`, then inspect `npx wrangler whoami`.
   For SSH/remote use, `npx --yes wrangler@4.119.0 login --device --browser=false`
   lets the owner sign in from another machine without forwarding a callback port.
   Confirm the chosen account is on the free plan. Set its `workers.dev`
   subdomain in the Cloudflare dashboard if it has none.
2. Set `name` in `wrangler.jsonc` to the agreed website name. The final hostname
   combines this name and the account subdomain.
3. Keep the CPU API listening on loopback on grace. Connect it through a stable
   HTTPS tunnel and keep both processes running. A Cloudflare named tunnel can
   use a domain already on the account; a Quick Tunnel is temporary and changes
   address when restarted. The browser never contacts that origin directly.
4. Configure these three runtime secrets using the dashboard or Wrangler's
   interactive prompts. Do not put their values in source code or command-line
   arguments:

   ```bash
   npx wrangler secret put CLIQUEFLOWMER_API_ORIGIN --config cloudflare/wrangler.jsonc
   npx wrangler secret put CLIQUEFLOWMER_API_KEY --config cloudflare/wrangler.jsonc
   npx wrangler secret put CLIQUEFLOWMER_PUBLIC_ORIGIN --config cloudflare/wrangler.jsonc
   ```

   `API_ORIGIN` is the tunnel's HTTPS origin. `API_KEY` must match
   `CLIQUEFLOWMER_PROXY_KEY` on grace. `PUBLIC_ORIGIN` must match the exact
   `CLIQUEFLOWMER_PUBLIC_ORIGIN` configured on the CPU service. The browser's
   origin is checked separately by the proxy. Missing settings fail closed.
   These settings are read only by the Worker, never embedded in the website.
5. Run `npm run deploy:cloudflare`, then verify an example optimization and its
   downloads through the returned public URL. Keep grace, its API, and the
   tunnel running. If the tunnel URL changes, update the `API_ORIGIN` secret.

There is no automatic publication in the ordinary `build` or `build:cloudflare`
commands. `secret put` may create the Worker if it does not exist, so the address
approval also precedes those remote configuration commands.

## Free-plan limits

Cloudflare's free Worker allowance is currently 100,000 requests per day and
10 ms of CPU per request. Static assets are served directly; only the API proxy
executes Worker code. Model computation runs on grace, so it does not consume
the Worker's CPU time. Polling a job uses multiple requests. The backend admits
one CPU job at a time and retains its existing input and runtime bounds.

Official references:

- [Static asset routing](https://developers.cloudflare.com/workers/static-assets/binding/)
- [Free Worker limits](https://developers.cloudflare.com/workers/platform/limits/)
- [workers.dev addresses](https://developers.cloudflare.com/workers/configuration/routing/workers-dev/)
- [Secrets](https://developers.cloudflare.com/workers/configuration/secrets/)

Limits and hosting policy were checked on 2026-10-28.

## Current Grace connection

The proxy supplies the CPU API's configured trusted origin upstream and checks
that browser POSTs originate from the public website. The configured API origin
is an exact string; keep it consistent on both sides when changing deployment
settings.

The Grace process uses `.cache-demo/website-runtime.json` at the repository root.
Its current backend connection is a Quick Tunnel, so a Grace or tunnel restart
requires starting the API/tunnel again and updating `CLIQUEFLOWMER_API_ORIGIN`.
The public `cliqueflowmer.znowu.workers.dev` address does not change. No paid
subscription or domain was purchased. Keep Grace and both processes running.
