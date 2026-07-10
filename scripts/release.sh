#!/usr/bin/env bash
# Release chore — the manual side. Bumps the crate version and checks the
# CHANGELOG entry exists; you then open a PR with the result. Merging that PR
# to main triggers .github/workflows/release.yml, which creates the git tag
# and the GitHub release (notes = the matching CHANGELOG section) automatically.
# The tag + release are a *side effect of merging the version bump*, so they
# can't be forgotten (which is how v1.2.20–v1.2.29 shipped without releases).
#
# Usage:
#   1. Add a `## vX.Y.Z (YYYY-MM-DD)` section to CHANGELOG.md with the notes.
#   2. scripts/release.sh X.Y.Z          # bumps Cargo.toml + Cargo.lock
#   3. Open a PR with the CHANGELOG + version bump; merge it.
#      → the release workflow tags vX.Y.Z and publishes the GitHub release.
set -euo pipefail

VER="${1:?usage: scripts/release.sh X.Y.Z}"
VER="${VER#v}"
if ! [[ "$VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: version '$VER' is not X.Y.Z" >&2; exit 1
fi
cd "$(git rev-parse --show-toplevel)"

# 1. The CHANGELOG entry is the release-notes source — require it up front.
if ! grep -qE "^## v${VER} \(" CHANGELOG.md; then
  echo "error: no '## v${VER} (YYYY-MM-DD)' section in CHANGELOG.md — add the release notes first." >&2
  exit 1
fi

# 2. Bump Cargo.toml (crate version) + Cargo.lock (the tlaplusplus package entry).
sed -i -E "0,/^version = \"[0-9.]+\"/s//version = \"${VER}\"/" Cargo.toml
if [ -f Cargo.lock ]; then
  sed -i -E "/^name = \"tlaplusplus\"$/{n;s/^version = \"[0-9.]+\"/version = \"${VER}\"/}" Cargo.lock
fi

echo "Bumped crate version to ${VER}."
git --no-pager diff --stat Cargo.toml Cargo.lock || true
echo
echo "Next: commit (e.g. 'chore: release v${VER}'), open a PR with the CHANGELOG"
echo "entry + this bump, and merge it. The release workflow will tag v${VER} and"
echo "publish the GitHub release from the CHANGELOG section."
