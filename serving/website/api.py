"""Single-process website API; run with ``python -m serving.website.api``.

Model jobs run on the existing bounded CPU worker. The API retains at most one
job per signed browser session and 32 jobs overall, with expiring in-memory
results. Uploaded structures are never written to disk by this transport.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from serving.web_demo.service import (
    BUDGETS, EXAMPLES, MANIFEST, MAX_CIF_BYTES, MAX_JOB_SECONDS, MAX_SITES,
    RESULT_TTL_SECONDS, DemoService, Job, structure_summary, validate_cif,
)
from serving.web_demo.viewer import structure_payload

BODY_LIMIT = 110_000
MAX_RETAINED_JOBS = 32
COOKIE_NAME = "cliqueflowmer_session"
TERMINAL = {"completed", "failed", "cancelled"}
log = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


@dataclass
class Entry:
    owner: str
    job: Job
    created: float


class JobRegistry:
    """Bounded registry; an opaque ID alone never grants access to a job."""

    def __init__(self, worker, *, limit=MAX_RETAINED_JOBS, ttl=RESULT_TTL_SECONDS,
                 clock=time.monotonic):
        self.worker = worker
        self.limit = limit
        self.ttl = ttl
        self.clock = clock
        self._entries: dict[str, Entry] = {}
        self._latest: dict[str, str] = {}
        self._lock = threading.RLock()

    def _remove(self, job_id):
        entry = self._entries.pop(job_id)
        entry.job.cancel()
        if self._latest.get(entry.owner) == job_id:
            del self._latest[entry.owner]

    def prune(self):
        with self._lock:
            now = self.clock()
            for job_id, entry in list(self._entries.items()):
                with entry.job._lock:
                    finished = entry.job._finished
                deadline = (finished if finished is not None else entry.created + MAX_JOB_SECONDS) + self.ttl
                if now >= deadline:
                    self._remove(job_id)

    def submit(self, owner, cif, operation, seed, budget):
        with self._lock:
            self.prune()
            previous = self._latest.get(owner)
            if len(self._entries) >= self.limit and previous is None:
                raise ApiError(429, "Result storage is full. Please try again after earlier results expire.")
            try:
                job = self.worker.submit(cif, operation=operation, seed=seed, budget=budget)
            except ValueError as exc:
                if "worker is busy" in str(exc):
                    raise ApiError(429, "The CPU worker is busy. Please try again when the current job finishes.") from exc
                raise ApiError(422, str(exc)) from exc
            # Preserve the previous result if submitting a new job was rejected.
            if previous is not None:
                self._remove(previous)
            job_id = secrets.token_urlsafe(24)
            self._entries[job_id] = Entry(owner=owner, job=job, created=self.clock())
            self._latest[owner] = job_id
            return job_id

    def get(self, owner, job_id):
        with self._lock:
            self.prune()
            entry = self._entries.get(job_id)
            if entry is None or not secrets.compare_digest(entry.owner, owner):
                raise ApiError(404, "This job is unavailable or has expired.")
            return entry.job

    def close(self):
        with self._lock:
            for job_id in list(self._entries):
                self._remove(job_id)
        self.worker.close()


class Sessions:
    """Stateless signed cookies prevent caller-chosen session fixation."""

    def __init__(self):
        self._key = secrets.token_bytes(32)

    def _signature(self, session):
        return hmac.new(self._key, session.encode("ascii"), hashlib.sha256).hexdigest()

    def read(self, cookie):
        if cookie:
            session, separator, signature = cookie.partition(".")
            if (separator and len(session) == 64 and len(signature) == 64
                    and all(char in "0123456789abcdef" for char in session + signature)
                    and hmac.compare_digest(signature, self._signature(session))):
                return session
        return secrets.token_hex(32)

    def encode(self, session):
        return session + "." + self._signature(session)


def _origin(value):
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            return None
        default_port = 443 if parsed.scheme == "https" else 80
        return parsed.scheme, parsed.hostname.lower(), parsed.port or default_port
    except (ValueError, TypeError):
        return None


async def _json_body(request, *, allow_empty=False):
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            size = int(declared)
        except ValueError:
            raise ApiError(422, "Invalid request length.") from None
        if size < 0:
            raise ApiError(422, "Invalid request length.")
        if size > BODY_LIMIT:
            raise ApiError(413, "Requests must be smaller than 110 KB.")
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > BODY_LIMIT:
            raise ApiError(413, "Requests must be smaller than 110 KB.")
        chunks.extend(chunk)
    if allow_empty and not chunks:
        return {}
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise ApiError(415, "Send the request as JSON.")
    try:
        data = json.loads(chunks)
    except (ValueError, UnicodeDecodeError):
        raise ApiError(422, "Could not read the JSON request.") from None
    if not isinstance(data, dict):
        raise ApiError(422, "The request must be a JSON object.")
    return data


def _cif(data):
    cif = data.get("cif")
    if not isinstance(cif, str) or not cif.strip():
        raise ApiError(422, "Upload, paste, or choose a CIF first.")
    try:
        size = len(cif.encode("utf-8"))
    except UnicodeEncodeError:
        raise ApiError(422, "The CIF must be valid UTF-8 text.") from None
    if size > MAX_CIF_BYTES:
        raise ApiError(422, "The demo accepts CIF files up to 100 KB.")
    return cif


def _result_with_viewers(state):
    # Job.snapshot supplies a deep copy. Enrichment never changes the stored job.
    result = state.get("result")
    if result is not None:
        for prefix in ("input", "output"):
            cif = result.get(prefix + "_cif")
            result[prefix + "_viewer"] = structure_payload(validate_cif(cif)) if cif else None
    return state


def create_app(*, worker=None, static_dir=None, public_origin=None,
               proxy_key=None,
               registry_limit=MAX_RETAINED_JOBS, result_ttl=RESULT_TTL_SECONDS,
               clock=time.monotonic):
    """Create an application; dependency injection keeps transport tests offline."""
    configured_origin = public_origin if public_origin is not None else os.environ.get("CLIQUEFLOWMER_PUBLIC_ORIGIN")
    proxy_key = proxy_key if proxy_key is not None else os.environ.get("CLIQUEFLOWMER_PROXY_KEY")
    if configured_origin and _origin(configured_origin) is None:
        raise ValueError("CLIQUEFLOWMER_PUBLIC_ORIGIN must be one HTTP(S) origin.")
    worker = worker if worker is not None else DemoService()
    registry = JobRegistry(worker, limit=registry_limit, ttl=result_ttl, clock=clock)
    sessions = Sessions()
    validation_slot = threading.BoundedSemaphore(1)
    # Derive labels and summaries from the same CIFs served to the browser.
    # Compute once at startup rather than parsing structures on every request.
    example_catalog = []
    for name, cif in EXAMPLES.items():
        structure = validate_cif(cif)
        example_catalog.append({
            "name": name, "label": name.partition(" · ")[0], "cif": cif,
            "summary": structure_summary(structure),
            "element_count": len(structure.composition.elements),
        })

    async def validate(cif):
        # Bound expensive CIF parsing across previews and job submissions too.
        if not validation_slot.acquire(blocking=False):
            raise ApiError(429, "Another structure is being checked. Please try again shortly.")
        try:
            return await run_in_threadpool(validate_cif, cif)
        except ValueError as exc:
            raise ApiError(422, str(exc)) from exc
        finally:
            validation_slot.release()

    async def config(request):
        registry.prune()
        return JSONResponse({
            "examples": example_catalog,
            "limits": {"max_cif_bytes": MAX_CIF_BYTES, "max_sites": MAX_SITES,
                       "max_job_seconds": MAX_JOB_SECONDS, "result_ttl_seconds": result_ttl},
            "budgets": BUDGETS, "model": MANIFEST,
        })

    async def preview(request):
        data = await _json_body(request)
        structure = await validate(_cif(data))
        return JSONResponse({"cif": structure.to(fmt="cif"), "summary": structure_summary(structure),
                             "viewer": structure_payload(structure)})

    async def submit(request):
        data = await _json_body(request)
        cif = _cif(data)
        operation = data.get("operation", "optimize")
        budget = data.get("budget", "quick")
        seed = data.get("seed", 42)
        if not isinstance(operation, str) or operation not in {"optimize", "encode", "reconstruct"}:
            raise ApiError(422, "Choose encode, reconstruct, or optimize.")
        if not isinstance(budget, str) or budget not in BUDGETS:
            raise ApiError(422, "Choose a supported computation budget.")
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ApiError(422, "Seed must be an integer between 0 and 2147483647.")
        await validate(cif)
        job_id = registry.submit(request.state.session_id, cif, operation, seed, budget)
        return JSONResponse({"job_id": job_id}, status_code=202)

    async def status(request):
        job = registry.get(request.state.session_id, request.path_params["job_id"])
        state = await run_in_threadpool(_result_with_viewers, job.snapshot())
        return JSONResponse(state)

    async def cancel(request):
        await _json_body(request, allow_empty=True)
        job = registry.get(request.state.session_id, request.path_params["job_id"])
        state = job.snapshot()
        if state["status"] not in TERMINAL:
            job.cancel()
        return JSONResponse({"status": state["status"] if state["status"] in TERMINAL else "cancelling"})

    async def api_error(request, exc):
        headers = {"Retry-After": "5"} if exc.status == 429 else None
        return JSONResponse({"error": exc.message}, status_code=exc.status, headers=headers)

    async def http_error(request, exc):
        return JSONResponse({"error": "Not found." if exc.status_code == 404 else "Request not allowed."},
                            status_code=exc.status_code)

    async def unexpected_error(request, exc):
        # Do not place CIF contents, session cookies, or exception text in logs.
        log.error("Website API failed (%s)", type(exc).__name__)
        return JSONResponse({"error": "This request could not be completed. Please try again."}, status_code=500)

    async def expire_results():
        while True:
            await asyncio.sleep(min(60, result_ttl))
            registry.prune()

    @asynccontextmanager
    async def lifespan(app):
        cleanup = asyncio.create_task(expire_results())
        try:
            yield
        finally:
            cleanup.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup
            await run_in_threadpool(registry.close)

    app = Starlette(debug=False, routes=[
        Route("/api/config", config, methods=["GET"]),
        Route("/api/preview", preview, methods=["POST"]),
        Route("/api/jobs", submit, methods=["POST"]),
        Route("/api/jobs/{job_id}", status, methods=["GET"]),
        Route("/api/jobs/{job_id}/cancel", cancel, methods=["POST"]),
        *([Mount("/", StaticFiles(directory=static_dir, html=True))] if static_dir else []),
    ], exception_handlers={ApiError: api_error, HTTPException: http_error, Exception: unexpected_error},
        lifespan=lifespan)
    app.state.registry = registry

    async def session_and_origin(request: Request, call_next):
        if proxy_key and not secrets.compare_digest(request.headers.get("x-cliqueflowmer-key", ""), proxy_key):
            return JSONResponse({"error": "Not authorized."}, status_code=403)
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        owner = sessions.read(request.cookies.get(COOKIE_NAME))
        request.state.session_id = owner
        expected = _origin(configured_origin or str(request.base_url))
        if request.method == "POST" and _origin(request.headers.get("origin")) != expected:
            response = JSONResponse({"error": "This request must come from the demo website."}, status_code=403)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Vary"] = "Cookie"
        response.set_cookie(
            COOKIE_NAME, sessions.encode(owner), httponly=True, samesite="strict",
            secure=bool(expected and expected[0] == "https"),
            max_age=RESULT_TTL_SECONDS + MAX_JOB_SECONDS, path="/",
        )
        return response

    app.add_middleware(BaseHTTPMiddleware, dispatch=session_and_origin)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.website.api:app", host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "7860")), workers=1,
                access_log=False, log_level="warning")
