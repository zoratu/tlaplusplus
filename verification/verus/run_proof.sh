#!/bin/bash
# Run Verus on the seqlock_resize.rs proof artifact.
#
# Requires:
#   - Verus built from source (see https://github.com/verus-lang/verus/blob/main/BUILD.md)
#   - Z3 4.12.5 OR a newer Z3 with --no-solver-version-check
#   - Environment variable VERUS_DIR pointing at the verus repo root, OR
#     `verus` already on PATH.
#
# This script intentionally does NOT install Verus — see ../README.md
# (root project) for setup notes. Verus has a multi-hundred-MB toolchain
# footprint and is built only on the verification spot, never CI.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

if [ -n "${VERUS_DIR:-}" ]; then
  VERUS_BIN="$VERUS_DIR/source/target-verus/release/verus"
elif command -v verus >/dev/null 2>&1; then
  VERUS_BIN=verus
else
  echo "error: cannot find a verus binary." >&2
  echo "       Set VERUS_DIR=/path/to/verus or put 'verus' on PATH." >&2
  exit 2
fi

if [ ! -x "$VERUS_BIN" ]; then
  echo "error: $VERUS_BIN does not exist or is not executable" >&2
  exit 2
fi

# Verus needs `rustup` and the matching pinned toolchain (1.95.0 at the
# time of writing, see verus/rust-toolchain.toml). If rustup is installed
# via rustup-init, source ~/.cargo/env so the wrapper finds it.
if ! command -v rustup >/dev/null 2>&1; then
  if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    . "$HOME/.cargo/env"
  fi
fi

echo "Using verus: $VERUS_BIN"
echo

# Verus's solver-version check expects Z3 4.12.5 exactly. We commit to
# the option even if the bundled Z3 is the right version: tlaplusplus
# users who run this from an aarch64 spot need Z3 from apt (currently
# 4.13.3) since upstream Verus does not yet ship an aarch64 Linux Z3.
exec "$VERUS_BIN" \
  -V no-solver-version-check \
  seqlock_resize.rs
