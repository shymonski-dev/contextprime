# Security Policy

## Supported Versions

Security fixes are applied to the default branch and the latest tagged release.

## Release Security Gates

Run from repository root before release:

- `./run_security_release_gate.sh`
- `./run_european_union_artificial_intelligence_act_readiness.sh`
- `./run_full_tests_stable.sh`

Repository automation:

- `.github/workflows/security-gate.yml` runs secret scanning and dependency checks.
- `.github/dependabot.yml` keeps dependency and workflow updates active.

## Reporting A Vulnerability

Please do not open public issues for security reports.

Send a private report with:

- clear reproduction steps
- impact assessment
- affected versions or commit hashes
- any proof of concept material

If private reporting is not yet enabled on the repository host, contact the maintainer directly and request a private disclosure channel before sharing exploit details.

## Disclosure Process

1. report is acknowledged
2. impact is validated and triaged
3. fix is prepared and tested
4. advisory and patched release are published

## Scope

In scope:

- credential exposure
- authentication and authorization bypass
- request limiting bypass
- dependency and supply chain vulnerabilities
- data leakage from logs or responses
- default or weak password usage in documentation or examples

Out of scope:

- issues requiring physical host access
- unsupported historical snapshots
