# CliqueFlowmer paper

Camera-ready sources for **Offline Materials Optimization with CliqueFlowmer**,
accepted at NeurIPS 2026. `final.tex` is the sole main document and uses the
current, unmodified official NeurIPS style with `main,final` options. The same
accepted-paper version is suitable for the arXiv replacement.

- [arXiv record](https://arxiv.org/abs/2603.06082)
- [Interactive crystal workspace](https://cliqueflowmer.znowu.workers.dev)

## Submission files

Run the preparation command below to produce these files in `paper/dist/`:

| File | Use |
| --- | --- |
| `CliqueFlowmer_NeurIPS2026.pdf` | Upload this PDF for the NeurIPS camera-ready revision |
| `CliqueFlowmer_arxiv.zip` | Upload this complete source project for the arXiv replacement; also importable in Overleaf |
| `submission-report.json` | File hashes and local compilation/PDF/archive validation results |
| `submission-build.log` | Full build log, kept outside the upload ZIP |

[submission-metadata.json](submission-metadata.json) contains the title, author
list, updated abstract, categories, and suggested comments for the replacement
form. Replace **arXiv:2603.06082**, rather than creating a new identifier. The
abstract draft points to the current website and removes the old Colab link.
Retain the existing paper's CC BY 4.0 license and cs.AI/cs.CE categories unless
the authors deliberately change them. The repository's MIT code license is
separate. A journal reference and DOI should only be added when available.

The ZIP has `final.tex` at its root, all 54 required figure assets, custom macros,
the official style, `final.bib`, a freshly generated `final.bbl`, and
`00README.json`. The control file selects pdfLaTeX and TeX Live 2025. It has no
compiled paper PDF, logs, auxiliary files, old drafts, build scripts, or unused
images. Figures remain in their original PDF/PNG formats; no conversion or
shell escape is required.

After uploading, select `final.tex` and pdfLaTeX/TeX Live 2025 if prompted, inspect
arXiv's generated PDF, and apply the replacement metadata. These files are
prepared and locally validated; no NeurIPS or arXiv submission has been made.
Account checks, moderation, and the mandatory arXiv preview occur on the server.

## Build and prepare

Install TeX Live 2025, Python 3, and the Poppler tools (`pdfinfo`, `pdffonts`, and
`pdftotext`). A standard TeX Live installation with the recommended and extra
LaTeX packages and fonts contains the manuscript dependencies.

Select that TeX Live installation through `PATH`, then run from the repository
root:

```bash
python3 paper/prepare_submission.py
```

For an isolated installation, the command is:

```bash
PATH=/path/to/texlive/2025/bin/x86_64-linux:$PATH python3 paper/prepare_submission.py
```

The script verifies source hashes, builds with shell escape disabled, checks
fonts and references, creates the ZIP, extracts it into a clean directory, and
recompiles using only its bundled bibliography. It rejects mismatched compilers,
missing dependencies, unsafe filenames, unresolved references, overflow boxes,
unembedded/Type 3 fonts, or a main-text limit violation. It checks that the
extracted project reproduces the PDF's text and page count. It does not contact
arXiv or run its account/moderation checks.

For ordinary editing and PDF builds:

```bash
bash paper/build.sh
```

The output is `paper/build/final.pdf`. An optional argument selects a different
build directory, with relative paths resolved from `paper/`:

```bash
bash paper/build.sh /tmp/cliqueflowmer-paper-build
```

For Overleaf, upload the source ZIP, select `final.tex`, pdfLaTeX, and TeX Live
2025. All required figures and the bibliography are included.

## Validation and provenance

Validated on 2026-10-28 with pdfTeX 1.40.28 / TeX Live 2025:

- 10 main-text pages, 37 total; acknowledgments begin on page 11.
- US Letter, all 37 fonts embedded, no Type 3 fonts, JavaScript, or encryption.
- No final-pass LaTeX/BibTeX warnings, undefined references, or overfull boxes.
  One pre-existing underfull paragraph is harmless and visually inspected.
- Clean source-only ZIP rebuild matches the built PDF's text and page count.
- arXiv's open-source preflight parser passes with **zero issues**, detects all
  54 figures, and accepts the compiler control file. The final ZIP was checked
  using [arXiv/submission-tools](https://github.com/arXiv/submission-tools) commit
  `b093524ee453e8d10e82f3f5905551f8503d0263`, with source and JavaScript checks
  enabled. This is a local run of the official tool, not an arXiv server upload.

Scientific content and figures are unchanged. Submission preparation makes
image filenames explicit for source discovery, disables unused EPS conversion,
and updates the official style with its Education-track option. These source
changes preserve every word's rendered position under TeX Live 2025. Modern
TeX improves some mathematical Unicode text extraction and slightly changes
appendix pagination relative to the older TeX Live 2019 PDF; the main-text
page limit and all cross-references remain correct.

`manifest.json` records the reviewed source hashes and original PDF provenance.
Generated PDFs, ZIPs, reports, and logs stay outside Git. Earlier submissions,
design proposals, and revision backups remain in the local working archive.

## Official requirements consulted

- [NeurIPS 2026 main-track handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)
- [Official NeurIPS 2026 author kit](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
- [arXiv TeX submission requirements](https://info.arxiv.org/help/submit_tex.html)
- [arXiv TeX Live versions](https://info.arxiv.org/help/faq/texlive.html)
- [arXiv compiler control file](https://info.arxiv.org/help/00README.html)
- [Replacing an arXiv submission](https://info.arxiv.org/help/replace.html)
