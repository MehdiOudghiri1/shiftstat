# API Stability

ShiftStat exposes a small stable surface through package modules and selected
top-level names in `shiftstat.__all__`. Anything imported from a private module,
private function, or name beginning with `_` is not part of the compatibility
contract.

## Stability Levels

**Stable**

- public metrics and result containers used in the documentation
- public estimator-like classes such as detectors, auditors, calibrators,
  weighters, analyzers, benchmark runners, and experiment runners
- documented CLI behavior for `shiftstat-experiment`

**Experimental**

- benchmark presets and generated artifact schemas before version `1.0`
- certification and adaptive subgroup-audit APIs while the corresponding
  statistical guarantees are still being expanded
- plotting defaults, figure titles, and visual styling

**Internal**

- helper functions in private modules
- dataclass fields not documented in the API reference
- benchmark implementation details used only by tests or paper scripts

## Deprecation Policy

Before `1.0`, incompatible changes may still happen, but they should be rare
and documented in `CHANGELOG.md`. After `1.0`, public API removals should emit a
deprecation warning for at least one minor release before removal.

Deprecation notes should include:

- the affected name
- the replacement path
- the first version that emits the warning
- the earliest version where removal can occur

## Versioning

ShiftStat follows semantic-versioning intent:

- patch releases fix bugs without changing public behavior
- minor releases add APIs or documented capabilities
- major releases may change public contracts

The package version is read from `src/shiftstat/_version.py` and injected into
package metadata during the build.
