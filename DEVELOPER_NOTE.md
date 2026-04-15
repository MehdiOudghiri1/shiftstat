# Developer note

## Architectural choices

- `src/` layout with typed, estimator-style modules and a small public surface
- Shared tabular validation and schema inference in `shiftstat.utils`
- Clear separation between statistical computation, plotting, and reporting
- Dataclass-based result containers for inspectable, lightweight outputs
- Synthetic datasets and benchmarks designed to support future research experiments

## Public API decisions

- `ShiftDetector` is the main entry point for feature-wise and dataset-level drift analysis
- `ImportanceWeighter` estimates target-over-reference importance weights while keeping downstream evaluation separate
- Weighted metric utilities are available as standalone functions for easy integration with scikit-learn workflows
- Reports expose `to_markdown()`, `to_dict()`, and `to_frame()` to keep outputs notebook- and paper-friendly

## Deferred to V2

- Calibration-aware evaluation under shift
- Selective prediction and abstention diagnostics
- Rich subgroup robustness reporting
- Advanced benchmark suites and experiment orchestration
- Deep learning and streaming-monitoring integrations

