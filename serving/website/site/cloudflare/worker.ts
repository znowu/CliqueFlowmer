import { modelProxy, type ModelEnv } from "../worker/model-proxy";

type Env = ModelEnv & { ASSETS: Fetcher };

export default {
  fetch(request: Request, env: Env): Promise<Response> {
    if (new URL(request.url).pathname.startsWith("/api/")) {
      return modelProxy(request, env);
    }
    return env.ASSETS.fetch(request);
  },
};
