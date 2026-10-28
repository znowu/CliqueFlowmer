"""Bounded CPU inference shared by the Streamlit UI and benchmark CLI.

No uploaded file is written to disk. One worker owns the model and RNG at a time.
Results live in session memory; snapshots discard results older than 30 minutes.
Expiration is checked on access, without a background deletion timer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable
import warnings

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifFile, CifParser

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "checkpoint.json").read_text())
MAX_CIF_BYTES = 100_000
MAX_SITES = 20
MAX_JOB_SECONDS = 300
RESULT_TTL_SECONDS = 1800
BUDGETS = {
    "quick": {"es_steps": 100, "flow_steps": 1000},
    "standard": {"es_steps": 500, "flow_steps": 1000},
}
EXAMPLES = {
    "Strontium titanate · 5 sites": (HERE / "examples/strontium_titanate.cif").read_text(),
    "Silicon · 2 sites": (HERE / "examples/silicon.cif").read_text(),
    "Cs₂AgBiBr₆ · 10 sites": (HERE / "examples/cesium_silver_bismuth_bromide.cif").read_text(),
}
log = logging.getLogger(__name__)


def validate_structure(structure: Structure) -> Structure:
    if not 1 <= len(structure) <= MAX_SITES:
        raise ValueError(f"Use a structure with 1–{MAX_SITES} sites in its uploaded cell.")
    if not structure.is_ordered:
        raise ValueError("Use an ordered CIF with fully occupied sites; partial occupancies are unsupported.")
    try:
        numbers = [site.specie.Z for site in structure]
    except (AttributeError, ValueError) as exc:
        raise ValueError("Every site must contain a supported chemical element.") from exc
    if any(not 1 <= z <= 118 for z in numbers):
        raise ValueError("Every site must contain a supported chemical element.")
    lengths = np.asarray(structure.lattice.abc)
    coords = np.asarray(structure.frac_coords)
    if (not np.isfinite(structure.lattice.matrix).all()
            or not np.isfinite(coords).all()
            or not np.isfinite(structure.volume)
            or not 0.1 <= structure.volume <= 1e6
            or np.any(lengths < 0.1) or np.any(lengths > 100)
            or np.any(np.abs(coords) > 1e4)):
        raise ValueError("The CIF must have a finite, nondegenerate cell and finite atomic coordinates.")
    # A tiny intersite separation can make generated-structure comparison and
    # visualization misleading. This is a geometric check, not chemical validity.
    if len(structure) > 1:
        distances = structure.distance_matrix
        np.fill_diagonal(distances, np.inf)
        if np.min(distances) < 0.5:
            raise ValueError("The structure contains sites less than 0.5 Å apart.")
    return structure


def validate_cif(cif_text: str) -> Structure:
    if not isinstance(cif_text, str) or not cif_text.strip():
        raise ValueError("Upload, paste, or choose a CIF first.")
    if len(cif_text.encode("utf-8")) > MAX_CIF_BYTES:
        raise ValueError("The demo accepts CIF files up to 100 KB.")
    # CifFile stores blocks by header and silently overwrites repeated names.
    # Count the headers it recognizes before parsing, so repeated names cannot
    # hide additional blocks (including blocks the parser might otherwise skip).
    if len(re.findall(r"^\s*data_", cif_text, flags=re.MULTILINE)) != 1:
        raise ValueError("Upload exactly one CIF data block containing one structure.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = CifFile.from_str(cif_text)
            if len(parsed.data) != 1:
                raise ValueError("Upload exactly one CIF data block containing one structure.")
            block = next(iter(parsed.data.values())).data
            # Bound work before symmetry expansion, not only after parsing.
            for key, value in block.items():
                key = key.lower()
                if isinstance(value, list):
                    if key.startswith("_atom_site_") and len(value) > MAX_SITES:
                        raise ValueError("The demo accepts at most 20 atom-site rows.")
                    if "symop" in key or "equiv_pos" in key:
                        if len(value) > 192:
                            raise ValueError("This CIF contains too many symmetry operations.")
            parser = CifParser.from_str(cif_text, occupancy_tolerance=1.0, check_cif=False)
            structures = parser.parse_structures(primitive=False, on_error="raise")
    except ValueError as exc:
        if str(exc).startswith(("Upload exactly", "The demo", "This CIF")):
            raise
        raise ValueError("Could not read a valid crystal structure from this CIF.") from exc
    except Exception as exc:
        raise ValueError("Could not read a valid crystal structure from this CIF.") from exc
    if len(structures) != 1:
        raise ValueError("Upload exactly one crystal structure.")
    return validate_structure(structures[0])


def structure_summary(structure: Structure) -> dict:
    return {
        "formula": structure.composition.reduced_formula,
        "n_sites": len(structure),
        "volume": float(structure.volume),
        "lattice_lengths": list(structure.lattice.abc),
        "lattice_angles": list(structure.lattice.angles),
    }


def load_pinned_model():
    import torch
    from huggingface_hub import hf_hub_download
    from serving.data_tools import normal_lengths_from_mle
    from serving.inference import load_model_from_checkpoint

    # Local overrides are useful for offline tests, but must still be the same
    # hash-pinned published model. Never load a user-uploaded checkpoint.
    path = os.environ.get("CLIQUEFLOWMER_CHECKPOINT")
    if not path:
        path = hf_hub_download(
            repo_id=MANIFEST["repo_id"], filename=MANIFEST["filename"],
            revision=MANIFEST["revision"],
        )
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != MANIFEST["sha256"]:
        raise RuntimeError("Published checkpoint checksum verification failed.")
    prior = json.loads((HERE / "length_prior.json").read_text())
    # Large thread pools are slower for small structures on shared CPU hosts.
    torch.set_num_threads(max(1, min(2, int(os.environ.get("CLIQUEFLOWMER_THREADS", "2")))))
    model = load_model_from_checkpoint(
        path, model_kwargs={"initial_length_dist": normal_lengths_from_mle(prior, "cpu")},
        device="cpu",
    )
    model.eval()
    model.requires_grad_(False)
    return model


def run_pipeline(model, cif_text: str, *, operation="optimize", seed=42,
                 budget="quick", progress=lambda fraction, message: None,
                 cancel_check=lambda: None) -> dict:
    import torch
    from models.graphops import separate_latents
    from serving.inference import encode_structure, decode_latent, optimize_es

    if operation not in {"encode", "reconstruct", "optimize"}:
        raise ValueError("Unknown operation.")
    if budget not in BUDGETS:
        raise ValueError("Unknown computation budget.")
    if not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("Seed must be an integer between 0 and 2147483647.")
    settings = BUDGETS[budget]
    cancel_check()
    structure = validate_cif(cif_text)
    progress(0.05, "Encoding the crystal")
    started = time.monotonic()
    # The service serializes jobs; restore RNG after each request so user seeds
    # do not leak across sessions or perturb surrounding research code.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        z = encode_structure(model, structure).detach()
        cancel_check()
        with torch.no_grad():
            initial = float(model.target_regressor(separate_latents(
                z.reshape(1, -1), model.index_matrix.to(z.device))).item())
        if not np.isfinite(initial):
            raise ValueError("The predictor produced a nonfinite score for this structure.")
        z_output = z.clone()
        final = initial
        history = [{"step": 0, "score": initial}]
        if operation == "optimize":
            progress(0.12, "Optimizing the latent representation")
            z_output, score, records = optimize_es(
                model, z, steps=settings["es_steps"], weight_decay=0.4,
                progress_callback=lambda step, total: progress(
                    0.12 + 0.28 * step / max(1, total), f"Optimizing · {step}/{total} steps"),
                cancel_check=cancel_check, return_history=True,
            )
            z_output = z_output.detach().reshape(-1)
            final = float(score.detach().item())
            history = [{"step": h["step"], "score": h["objective"]} for h in records]
        if not np.isfinite(final) or not torch.isfinite(z_output).all():
            raise ValueError("Optimization produced a nonfinite result. Try another seed or structure.")
        output = None
        decode_attempts = 0
        if operation != "encode":
            # Flow decoding is stochastic. Retry geometric rejection once, at
            # the same latent and continuing the seeded RNG stream. This never
            # selects candidates by a property score or repeats optimization.
            for attempt in range(2):
                decode_attempts = attempt + 1
                message = "Decoding the candidate crystal" if attempt == 0 else "Resampling an invalid geometry · final attempt"
                progress(0.4 if attempt == 0 else 0.95, message)
                try:
                    output = decode_latent(
                        model, z_output, integration="cfg", relax=False,
                        n_steps=settings["flow_steps"], beam_width=10, min_atoms=1, max_atoms=MAX_SITES,
                        progress_callback=lambda step, total: progress(
                            (0.4 + 0.55 * step / max(1, total)) if attempt == 0 else 0.95,
                            message),
                        cancel_check=cancel_check,
                    )
                    validate_structure(output)
                    break
                except ValueError as exc:
                    # Configuration/shape errors should surface immediately;
                    # only invalid decoded structures are eligible for retry.
                    geometric_failure = str(exc).startswith((
                        "No valid structure produced", "The structure contains",
                        "The CIF must have a finite", "Use a structure with",
                    ))
                    if not geometric_failure:
                        raise
                    if attempt == 1:
                        raise ValueError("Two decoded geometries failed validation. Try a different seed or another structure.") from exc
                    cancel_check()
    cancel_check()
    result = {
        "operation": operation,
        "input_cif": structure.to(fmt="cif"),
        "output_cif": output.to(fmt="cif") if output is not None else None,
        "input_summary": structure_summary(structure),
        "output_summary": structure_summary(output) if output is not None else None,
        "latent": z.cpu().tolist(),
        "optimized_latent": z_output.cpu().tolist() if operation == "optimize" else None,
        "initial_score": initial, "final_score": final,
        "best_observed_score": min(h["score"] for h in history),
        "history": history,
        "metadata": {
            **MANIFEST, "checkpoint": MANIFEST["filename"], "seed": seed,
            "budget": budget, "es_steps": settings["es_steps"] if operation == "optimize" else 0,
            "flow_steps": settings["flow_steps"] if output is not None else 0,
            "beam_width": 10, "guidance": 2.0, "min_atoms": 1, "max_atoms": MAX_SITES,
            "relaxed": False, "elapsed_seconds": time.monotonic() - started,
            "decode_attempts": decode_attempts,
            "input_sha256": hashlib.sha256(cif_text.encode()).hexdigest(),
            "preprocessing": "Uploaded cell retained; ordered fully occupied sites only.",
            "score_note": "Scores are evaluated on latents, not independently on decoded structures.",
        },
    }
    if output is not None:
        result["composition_preserved"] = structure.composition == output.composition
        if operation == "reconstruct":
            from pymatgen.analysis.structure_matcher import StructureMatcher
            matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
            result["structure_matched"] = bool(matcher.fit(structure, output))
    progress(1.0, "Complete")
    return result


class Cancelled(Exception):
    """Cooperative cancellation; handled without displaying a traceback."""


@dataclass
class Job:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _state: dict = field(default_factory=lambda: {
        "status": "queued", "progress": 0.0, "message": "Preparing the model",
        "result": None, "error": None,
    })
    _finished: float | None = None

    def update(self, **values):
        with self._lock:
            self._state.update(values)
            if values.get("status") in {"completed", "failed", "cancelled"}:
                self._finished = time.monotonic()

    def cancel(self):
        self._cancel.set()

    def snapshot(self) -> dict:
        with self._lock:
            if self._finished is not None and time.monotonic() - self._finished > RESULT_TTL_SECONDS:
                self._state.update(status="failed", result=None, message="Result expired",
                                   error="This result expired. Run the demo again.")
            return deepcopy(self._state)


class DemoService:
    """One model, one active job, no unbounded queue or retained job registry."""

    def __init__(self, model_loader: Callable = load_pinned_model,
                 runner: Callable = run_pipeline, timeout_seconds=MAX_JOB_SECONDS):
        self._model_loader = model_loader
        self._runner = runner
        self._model = None
        self._slot = threading.BoundedSemaphore(1)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cliqueflowmer")
        self.timeout_seconds = timeout_seconds

    def submit(self, cif_text: str, operation="optimize", seed=42, budget="quick") -> Job:
        if operation not in {"encode", "reconstruct", "optimize"} or budget not in BUDGETS:
            raise ValueError("Unsupported operation or budget.")
        if not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ValueError("Seed must be an integer between 0 and 2147483647.")
        if not isinstance(cif_text, str) or len(cif_text.encode()) > MAX_CIF_BYTES:
            raise ValueError("The demo accepts CIF files up to 100 KB.")
        if not self._slot.acquire(blocking=False):
            raise ValueError("The free CPU worker is busy. Please try again when the current job finishes.")
        job = Job()
        try:
            self._executor.submit(self._run, job, cif_text, operation, seed, budget)
        except Exception:
            self._slot.release()
            raise
        return job

    def _run(self, job, cif_text, operation, seed, budget):
        started = time.monotonic()

        def check():
            if job._cancel.is_set():
                raise Cancelled()
            if time.monotonic() - started > self.timeout_seconds:
                raise TimeoutError("This job exceeded the demo time limit. Try the Quick budget or a smaller structure.")

        def progress(fraction, message):
            check()
            job.update(progress=float(fraction), message=message)

        try:
            job.update(status="running")
            check()
            validate_cif(cif_text)
            if self._model is None:
                self._model = self._model_loader()
            check()
            result = self._runner(self._model, cif_text, operation=operation, seed=seed,
                                  budget=budget, progress=progress, cancel_check=check)
            check()
            job.update(status="completed", progress=1.0, message="Complete", result=result)
        except Cancelled:
            job.update(status="cancelled", message="Cancelled")
        except (ValueError, TimeoutError) as exc:
            job.update(status="failed", error=str(exc), message="Could not finish this job")
        except Exception:
            log.exception("Demo inference failed")
            job.update(status="failed", error="The model could not complete this request. Try again or choose an example.")
        finally:
            self._slot.release()

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
