# Experiment Configuration Guide

ShiftStat experiments are config-driven wrappers around the benchmark framework. The goal is reproducible scientific execution, not a generic orchestration platform.

## Supported formats

- JSON
- YAML

## Minimal YAML example

```yaml
name: publication_sweep
output_dir: paper_assets/generated/publication_sweep
figure_format: png
scenario:
  preset: covariate_shift_sweep
  baseline_names: [raw_model, weighting_only, confidence_abstention]
  seeds: [7, 19]
  parameters:
    severities: [0.2, 0.8, 1.4]
    n_samples_ref: 220
    n_samples_target: 220
```

## Running the config

```bash
shiftstat-experiment path/to/config.yaml
```

or

```bash
python -m shiftstat.experiments path/to/config.yaml
```

## What gets written

- per-scenario run CSVs
- per-scenario aggregated summary CSVs
- LaTeX-ready tables
- publication-friendly figures
- markdown summaries
- experiment manifest and run log

## Design notes

- Scenarios stay explicit through named presets and parameter blocks.
- Seed lists live in the config so reruns are exact.
- Artifacts are written to ordinary directories so they can be versioned, attached to papers, or used in appendices.
