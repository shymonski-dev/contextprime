# Marketplace Release Checklist

Use this checklist before publishing the action in GitHub Marketplace.

## Required repository files

1. `action.yml`
2. `README.md`
3. `LICENSE`
4. `DUAL_LICENSE.md`
5. `LICENSES/`
6. `SECURITY.md`
7. `CODE_OF_CONDUCT.md`
8. `CONTRIBUTING.md`
9. `SUPPORT.md`

## Required checks

1. `./run_security_release_gate.sh`
2. `./run_european_union_artificial_intelligence_act_readiness.sh`
3. `./run_full_tests_stable.sh --skip-build` (recommended before release tag)

## Publish sequence

1. Commit and push `main`.
2. Create and push a version tag:
   - `git tag -a v1.0.0 -m "Marketplace release v1.0.0"`
   - `git push origin v1.0.0`
3. Open GitHub Releases and publish the tag.
4. In repository settings, publish the action to Marketplace.

## Post publish

1. Verify the listing text and branding icon.
2. Verify the usage example in Marketplace.
3. Verify pull request and push checks still pass.
