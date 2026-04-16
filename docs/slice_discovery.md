# Slice Discovery Methodology

ShiftStat V3 includes an interpretable baseline for locating concentrated target failures.

Implementation choices:

- objective: binary failure indicator such as prediction error or high-confidence error
- model: shallow decision tree
- rationale: simple, reproducible, and readable slice definitions

Why this approach:

- it produces human-auditable rules
- it keeps claims modest
- it is easy to rerun with different seeds or thresholds

Current limitations:

- slices are descriptive partitions, not causal explanations
- the target sample is used for discovery, so optimism is possible
- small failure counts can make slices unstable

Use `SliceDiscoverer` directly when you want explicit control over discovery, or use `ReliabilityAuditor` when you want discovery integrated into a broader audit workflow.
