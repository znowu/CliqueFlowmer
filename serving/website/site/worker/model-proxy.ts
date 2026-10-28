/** Same-origin transport; the CPU service credential never reaches the browser. */
export type ModelEnv = {
  CLIQUEFLOWMER_API_ORIGIN?: string;
  CLIQUEFLOWMER_API_KEY?: string;
  CLIQUEFLOWMER_PUBLIC_ORIGIN?: string;
};
const LIMIT = 110_000;
const SESSION = "cliqueflowmer_session";
function error(message: string, status: number) {
  return Response.json(
    { error: message },
    { status, headers: { "Cache-Control": "no-store" } },
  );
}
export async function modelProxy(
  request: Request,
  env: ModelEnv,
): Promise<Response> {
  const url = new URL(request.url);
  const allowed =
    request.method === "GET"
      ? /^\/api\/(config|jobs\/[A-Za-z0-9_-]{32})$/.test(url.pathname)
      : request.method === "POST" &&
        /^\/api\/(preview|jobs|jobs\/[A-Za-z0-9_-]{32}\/cancel)$/.test(
          url.pathname,
        );
  if (!allowed) return error("Request not allowed.", 404);
  if (
    request.method === "POST" &&
    request.headers.get("Origin") !== url.origin
  ) {
    return error("This request must come from the demo website.", 403);
  }
  if (
    !env.CLIQUEFLOWMER_API_ORIGIN ||
    !env.CLIQUEFLOWMER_API_KEY ||
    !env.CLIQUEFLOWMER_PUBLIC_ORIGIN
  ) {
    return error(
      "The model service is not connected yet. Please try again shortly.",
      503,
    );
  }
  let body: Uint8Array | undefined;
  if (request.method === "POST") {
    if (Number(request.headers.get("Content-Length")) > LIMIT)
      return error("Requests must be smaller than 110 KB.", 413);
    if (!request.headers.get("Content-Type")?.startsWith("application/json"))
      return error("Send the request as JSON.", 415);
    const reader = request.body?.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    if (reader)
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.length;
        if (size > LIMIT) {
          await reader.cancel();
          return error("Requests must be smaller than 110 KB.", 413);
        }
        chunks.push(value);
      }
    body = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      body.set(chunk, offset);
      offset += chunk.length;
    }
  }
  const headers = new Headers({
    "Content-Type": "application/json",
    Origin: env.CLIQUEFLOWMER_PUBLIC_ORIGIN,
    "x-cliqueflowmer-key": env.CLIQUEFLOWMER_API_KEY,
  });
  // Forward only the model session; unrelated browser cookies stay private.
  const cookie = request.headers
    .get("Cookie")
    ?.split(";")
    .map((x) => x.trim())
    .find((x) => x.startsWith(`${SESSION}=`));
  if (cookie) headers.set("Cookie", cookie);
  try {
    const upstream = await fetch(
      new URL(url.pathname, env.CLIQUEFLOWMER_API_ORIGIN),
      {
        method: request.method,
        headers,
        body: body as BodyInit | undefined,
        redirect: "manual",
        signal: AbortSignal.timeout(20_000),
      },
    );
    if (
      (upstream.status >= 300 && upstream.status < 400) ||
      !upstream.headers.get("Content-Type")?.includes("application/json")
    )
      return error("The CPU service is offline. Please try again later.", 503);
    const responseHeaders = new Headers({
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      Vary: "Cookie",
    });
    const session = upstream.headers.get("Set-Cookie");
    if (session?.startsWith(`${SESSION}=`))
      responseHeaders.set("Set-Cookie", session);
    const retry = upstream.headers.get("Retry-After");
    if (retry) responseHeaders.set("Retry-After", retry);
    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch (cause) {
    if (import.meta.env.DEV)
      console.error(
        "CPU proxy:",
        cause instanceof Error ? cause.message : "Request failed",
      );
    return error(
      "The CPU service is temporarily unavailable. Please try again shortly.",
      503,
    );
  }
}
