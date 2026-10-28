#!/usr/bin/env python3
"""Build and locally validate the NeurIPS PDF and arXiv source ZIP.

Uses only Python's standard library plus the TeX/Poppler command-line tools.
Select the requested TeX Live installation through PATH before running.
This does not upload anything or run arXiv's server-side submission checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


PAPER = Path(__file__).resolve().parent
PDF_NAME = "CliqueFlowmer_NeurIPS2026.pdf"
ZIP_NAME = "CliqueFlowmer_arxiv.zip"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", name)
        or path.is_absolute()
        or any(part in (".", "..") for part in name.split("/"))
        or "//" in name
        or str(path) != name
    ):
        raise ValueError(f"Unsafe or unsupported archive filename: {name!r}")


def output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        command, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=180
    )


def run(command: list[str], *, cwd: Path, log) -> None:
    log.write("\n$ " + " ".join(command) + "\n")
    log.flush()
    subprocess.run(
        command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
        check=True, timeout=180,
    )


def pdf_checks(path: Path) -> dict:
    info = {}
    for line in output(["pdfinfo", str(path)]).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    for key in ("Encrypted", "JavaScript"):
        if info.get(key) != "no":
            raise ValueError(f"{path.name}: {key} must be no, got {info.get(key)!r}")
    if not re.match(r"612(?:\.0+)? x 792(?:\.0+)? pts", info.get("Page size", "")):
        raise ValueError(f"{path.name}: expected US Letter page size")

    fonts = output(["pdffonts", str(path)]).splitlines()[2:]
    if not fonts:
        raise ValueError(f"{path.name}: no embedded text fonts detected")
    bad_fonts = [line for line in fonts if "Type 3" in line or line.split()[-5] != "yes"]
    if bad_fonts:
        raise ValueError(f"{path.name}: unembedded or Type 3 fonts: {bad_fonts}")

    return {
        "pages": int(info["Pages"]),
        "page_size": info["Page size"],
        "pdf_version": info["PDF version"],
        "embedded_font_count": len(fonts),
        "all_fonts_embedded": True,
        "type3_fonts": 0,
        "encrypted": False,
        "javascript": False,
        "title": info.get("Title"),
        "authors": info.get("Author"),
    }


def tex_log_checks(path: Path) -> dict:
    text = path.read_text(errors="replace")
    errors = re.findall(
        r"(?m)^!.*|^LaTeX Warning:.*|^Package \S+ Warning:.*|^Overfull .*", text
    )
    if errors:
        raise ValueError(f"{path.name}: unresolved TeX diagnostics: {errors[:8]}")
    return {
        "latex_warnings": 0,
        "overfull_boxes": 0,
        "underfull_boxes": len(re.findall(r"(?m)^Underfull ", text)),
    }


def prepare(args) -> None:
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    compiler = output(["pdflatex", "--version"]).splitlines()[0]
    version = re.search(r"TeX Live (\d{4})", compiler)
    if not version or int(version.group(1)) != args.texlive_version:
        raise ValueError(
            f"Expected TeX Live {args.texlive_version} on PATH, found {compiler}. "
            "Use the matching installation before preparing submission files."
        )

    manifest = json.loads((PAPER / "manifest.json").read_text())
    names = sorted(manifest["source_sha256"])
    if not {"final.tex", "final.bib", "math_commands.tex", "neurips_2026.sty"} <= set(names):
        raise ValueError("The source manifest omits a required manuscript file")
    sources = {}
    for name in names:
        safe_name(name)
        path = PAPER / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or non-regular source file: {name}")
        sources[name] = path.read_bytes()
    mismatches = [
        name for name, expected in manifest["source_sha256"].items()
        if sha256(sources[name]) != expected
    ]
    if mismatches:
        raise ValueError(f"Source files differ from the reviewed manifest: {mismatches}")
    top_levels = [
        name for name, data in sources.items()
        if name.endswith(".tex") and re.search(rb"(?m)^\s*\\documentclass", data)
    ]
    if top_levels != ["final.tex"]:
        raise ValueError(f"Expected only final.tex as the top-level document: {top_levels}")

    log_path = destination / "submission-build.log"
    with tempfile.TemporaryDirectory(prefix="cliqueflowmer-submission-") as temporary:
        work = Path(temporary)
        built = work / "build"
        with log_path.open("w") as log:
            run(["bash", str(PAPER / "build.sh"), str(built)], cwd=PAPER, log=log)
            reference_pdf = built / "final.pdf"
            if reference_pdf.stat().st_size >= 50_000_000:
                raise ValueError("NeurIPS PDF must be smaller than 50 MB")
            pdf_report = pdf_checks(reference_pdf)
            latex_report = tex_log_checks(built / "final.log")
            biblog = (built / "final.blg").read_text(errors="replace")
            if re.search(r"Warning--|error message|couldn't open|I found no", biblog):
                raise ValueError("BibTeX reported a warning or error; inspect submission-build.log")
            bbl = (built / "final.bbl").read_bytes()
            if b"\\begin{thebibliography}" not in bbl or b"\\bibitem" not in bbl:
                raise ValueError("The compiled bibliography is missing or empty")

            control = {
                "spec_version": 1,
                "process": {"compiler": "pdflatex"},
                "sources": [{"filename": "final.tex", "usage": "toplevel"}],
                "texlive_version": args.texlive_version,
            }
            payload = dict(sources)
            payload["final.bbl"] = bbl
            payload["00README.json"] = (json.dumps(control, indent=2) + "\n").encode()
            archive = work / ZIP_NAME
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
                for name, data in sorted(payload.items()):
                    safe_name(name)
                    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.external_attr = 0o100644 << 16
                    package.writestr(entry, data)

            extracted = work / "extracted"
            extracted.mkdir()
            with zipfile.ZipFile(archive) as package:
                archived_names = package.namelist()
                if len(archived_names) != len(set(archived_names)) or set(archived_names) != set(payload):
                    raise ValueError("Archive file list differs from the validated payload")
                for name in archived_names:
                    safe_name(name)
                    if package.read(name) != payload[name]:
                        raise ValueError(f"Archive content mismatch: {name}")
                if package.testzip() is not None:
                    raise ValueError("Archive CRC validation failed")
                package.extractall(extracted)

            # Mimic source-only submission: no .aux/.toc/.out, no BibTeX run.
            for _ in range(4):
                run(
                    ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                     "-halt-on-error", "final.tex"], cwd=extracted, log=log,
                )
            extracted_pdf = extracted / "final.pdf"
            extracted_report = pdf_checks(extracted_pdf)
            tex_log_checks(extracted / "final.log")
            reference_text = output(["pdftotext", "-layout", str(reference_pdf), "-"])
            extracted_text = output(["pdftotext", "-layout", str(extracted_pdf), "-"])
            if reference_text != extracted_text or pdf_report["pages"] != extracted_report["pages"]:
                raise ValueError("The extracted source ZIP does not reproduce the built PDF text/pages")
            pages = reference_text.split("\f")
            acknowledgments = next(
                (i + 1 for i, page in enumerate(pages)
                 if re.search(r"(?m)^\s*Acknowledg(?:e)?ments\b", page)), None
            )
            if acknowledgments != 11:
                raise ValueError(f"Expected 10 main pages and acknowledgments on page 11, found {acknowledgments}")

            report = {
                "manuscript": manifest["manuscript"],
                "compiler": compiler,
                "texlive_version": args.texlive_version,
                "shell_escape": False,
                "pdf": {**pdf_report, **latex_report, "main_pages": 10,
                        "acknowledgments_start_page": acknowledgments,
                        "sha256": sha256(reference_pdf.read_bytes()),
                        "bytes": reference_pdf.stat().st_size},
                "archive": {"sha256": sha256(archive.read_bytes()),
                            "bytes": archive.stat().st_size,
                            "uncompressed_bytes": sum(len(x) for x in payload.values()),
                            "crc_valid": True, "files": archived_names,
                            "source_sha256": {name: sha256(data) for name, data in sorted(payload.items())}},
                "source_manifest_hashes_match": all(
                    sha256(sources[name]) == expected
                    for name, expected in manifest["source_sha256"].items()
                ),
                "fresh_zip_build": {"passes": 4, "bibtex_run": False,
                                    "bundled_bbl": True, "text_and_page_count_match": True},
                "arxiv_server_checks_run": False,
                "validation_scope": "Local compiler, source archive, PDF, bibliography, and layout checks. No upload or moderation check.",
            }
            shutil.copyfile(reference_pdf, destination / PDF_NAME)
            shutil.copyfile(archive, destination / ZIP_NAME)
            (destination / "submission-report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"NeurIPS PDF: {destination / PDF_NAME}")
    print(f"arXiv source ZIP: {destination / ZIP_NAME}")
    print(f"Validation report: {destination / 'submission-report.json'}")
    print(f"Build log: {log_path}")
    print("Passed local submission checks; arXiv server-side checks have not been run.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texlive-version", type=int, choices=(2023, 2025), default=2025)
    parser.add_argument("--output", type=Path, default=PAPER / "dist")
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print(f"Submission preparation failed: {error}", file=sys.stderr)
        print(f"Build log, if available: {args.output.resolve() / 'submission-build.log'}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
