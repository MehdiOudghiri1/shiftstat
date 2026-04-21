# Preprint Workspace

This folder contains a theory-first manuscript draft for a scientific paper built around the ShiftStat codebase.

## Current manuscript focus

Working title:

`Weighted Worst-Group Reliability under Covariate Shift: A Theory-First Study of Hidden Deployment Failures`

Scientific goal:

- study target reliability under covariate shift
- formalize worst-group reliability, not only aggregate reliability
- derive transport and estimation results using importance weighting
- connect those results to subgroup and selective-deployment benchmarks already present in the repository

## Layout

- `main.tex`: manuscript entry point
- `style/preamble.tex`: packages, theorem environments, layout
- `style/macros.tex`: mathematical notation and helper commands
- `sections/`: modular paper sections
- `refs.bib`: bibliography
- `build.ps1`: PowerShell build helper

## Build

If you have a LaTeX toolchain installed, run:

```powershell
cd preprint
./build.ps1
```

The script tries `latexmk`, then `tectonic`, then a basic `pdflatex` + `bibtex` sequence.

## Provenance of current figures and numbers

The draft uses committed manuscript figures under `figures/`. The scripts in
`research/` can regenerate the study summaries used to build those figures.
Generated CSV/JSON/log outputs are intentionally ignored by git under:

- `research/results/`

The broader package benchmark artifacts can be regenerated from:

- `../paper_assets/configs/publication_suite.yaml`

These results are used as a strong initial draft, not as the final experimental campaign. A final paper should expand the seed count, add ablations, and likely add at least one real dataset or semi-real benchmark.
