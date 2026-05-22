#!/usr/bin/env bash
# Bootstrap Verus on a fresh aarch64 (or x86_64) spot and run
# `cargo verus check --features verus` against this crate.
#
# Designed for the manual Phase 2 verification workflow:
#   1. Provision spot, ssh in
#   2. rsync this repo to ~/tlaplusplus
#   3. scp this script to /tmp/bootstrap.sh and run it
#   4. Subsequent `cargo verus check` calls reuse the cached Verus build
#
# Workaround applied to vstd
# ===========================
# vstd's `axiom_u64_trailing_zeros` in `vstd/std_specs/bits.rs` triggers
# a Z3 4.13.3 reader-thread crash on aarch64 (the proof body's bit-
# vector assertions produce output Verus's parser rejects). Setting
# `#[verifier::external_body]` on JUST that one function skips the
# proof body but keeps the public spec / API visible — including the
# rest of vstd's Vec/Seq/Map specs, which would otherwise be erased
# under a blanket `verify = false` patch.
#
# The patch is applied to whatever cargo-git-checkout copy of vstd
# cargo has fetched (so we re-apply on every fresh spot or after
# `cargo clean`).
set -euo pipefail
cd "$HOME"

if [ -x "$HOME/verus/source/target-verus/release/verus" ]; then
  echo "verus binary already built; skipping toolchain build"
else
  echo "=== installing rustup + apt deps ==="
  if ! command -v rustc >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
  fi
  . "$HOME/.cargo/env"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential git z3 unzip cmake clang pkg-config libssl-dev \
    libz3-dev libclang-dev default-jre-headless python3

  echo "=== cloning Verus (shallow) ==="
  if [ ! -d "$HOME/verus" ]; then
    git clone --depth 1 https://github.com/verus-lang/verus.git "$HOME/verus"
  fi

  echo "=== Z3 wrapper ==="
  cd "$HOME/verus/source"
  cp /usr/bin/z3 ./z3
  chmod +x ./z3

  echo "=== building Verus from source (--vstd-no-verify) ==="
  . ../tools/activate
  vargo --no-solver-version-check build --release --vstd-no-verify 2>&1 | tail -5
fi

. "$HOME/.cargo/env"

echo
echo "=== ensuring rustup default 1.95.0 ==="
rustup toolchain install 1.95.0 --component rustc-dev --component llvm-tools 2>&1 | tail -3
rustup default 1.95.0 2>&1 | tail -2

export PATH="$HOME/verus/source/target-verus/release:$PATH"

# Trigger a cargo fetch so the vstd git checkout exists, then patch.
echo
echo "=== fetching vstd via cargo (this populates ~/.cargo/git/checkouts) ==="
cd "$HOME/tlaplusplus"
cargo fetch --features verus 2>&1 | tail -3 || true

echo
echo "=== patching vstd bits.rs::axiom_u64_trailing_zeros to external_body ==="
for f in $(ls /home/ubuntu/.cargo/git/checkouts/verus-*/*/source/vstd/std_specs/bits.rs 2>/dev/null); do
  python3 - <<PY
import re, sys
path = "$f"
with open(path) as fh:
    src = fh.read()
if '#[verifier::external_body]\npub broadcast proof fn axiom_u64_trailing_zeros' in src:
    print(f"already patched: {path}")
else:
    src2 = re.sub(
        r'(pub broadcast proof fn axiom_u64_trailing_zeros)',
        r'#[verifier::external_body]\n\1',
        src,
        count=1,
    )
    if src2 != src:
        with open(path, 'w') as fh:
            fh.write(src2)
        print(f"patched: {path}")
    else:
        print(f"no match in: {path}")
PY
done

echo
echo "=== ensuring vstd verify = true (so spec items are exposed) ==="
for f in $(ls /home/ubuntu/.cargo/git/checkouts/verus-*/*/source/vstd/Cargo.toml 2>/dev/null); do
  sed -i 's/^verify = false$/verify = true/' "$f"
done

echo
echo "=== running cargo verus check on tlaplusplus ==="
cargo verus check --features verus -- -V no-solver-version-check 2>&1 | tail -10
