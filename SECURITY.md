# Security Policy

## Supported Versions

ShiftStat is pre-1.0 research software. Security fixes are applied to the active
development line and to the latest published release when a release has been
published to PyPI.

## Reporting A Vulnerability

Please report suspected vulnerabilities privately to the maintainer listed in
`pyproject.toml`. Include:

- the affected ShiftStat version or commit
- the Python version and operating system
- a minimal reproduction, if possible
- whether the issue affects local files, generated artifacts, dependency
  handling, or arbitrary code execution

Do not open a public issue for a vulnerability until a fix or mitigation is
available.

## Scope

ShiftStat is a local scientific Python library. The most relevant security risks
are dependency supply-chain issues, unsafe artifact paths, unsafe loading of
user-provided configuration files, and accidental disclosure of local paths in
generated manifests.

Experiment configuration uses JSON and `yaml.safe_load`; configuration files
should still be treated as trusted local inputs because they control output
paths and benchmark execution.
