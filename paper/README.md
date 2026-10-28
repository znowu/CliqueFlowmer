# CliqueFlowmer paper

Camera-ready sources for **Offline Materials Optimization with CliqueFlowmer**,
accepted at NeurIPS 2026. The root document is `final.tex`, using the supplied
NeurIPS style with the `main,final` options.

- [arXiv preprint](https://arxiv.org/abs/2603.06082)
- [Interactive crystal workspace](https://cliqueflowmer.znowu.workers.dev)

## Build

Install a standard TeX Live distribution with pdfLaTeX, BibTeX, and the LaTeX
recommended/extra packages, then run:

```bash
bash paper/build.sh
```

The output is `paper/build/final.pdf`. An optional argument selects a different
build directory; relative paths are resolved from `paper/`:

```bash
bash paper/build.sh /tmp/cliqueflowmer-paper-build
```

For Overleaf, upload this directory, select `final.tex` as the main document,
and use pdfLaTeX. All required figures are included; no plot-generation step is
needed.

## Contents and provenance

This directory contains the approved manuscript, bibliography, mathematical
macros, NeurIPS style, and the 54 figure assets used by the manuscript. These
files were copied without content changes from the camera-ready working copy.
`manifest.json` records their SHA-256 checksums and the reference PDF checksum.

The verified layout has 10 main-text pages and 37 pages in total, with
acknowledgments starting on page 11. Generated PDFs and LaTeX build files stay
outside version control. Earlier submissions, unused figure variants, design
proposals, and local revision backups are not part of this source package.
