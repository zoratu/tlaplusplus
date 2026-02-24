#!/usr/bin/env bash
# Local CI check script - runs all quality checks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          tlaplusplus Local CI Check                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Track failures
FAILED_CHECKS=()

# Function to run a check
run_check() {
    local name="$1"
    local command="$2"

    echo -e "${BLUE}▶ Running: ${name}${NC}"

    if eval "$command"; then
        echo -e "${GREEN}✓ ${name} passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${name} failed${NC}"
        echo ""
        FAILED_CHECKS+=("$name")
        return 1
    fi
}

# 1. Check formatting
run_check "cargo fmt" "cargo fmt --all -- --check" || true

# 2. Clippy lints
run_check "cargo clippy" "cargo clippy --all-targets --all-features -- -D warnings" || true

# 3. Build (release)
run_check "cargo build --release" "cargo build --release"

# 4. Tests
run_check "cargo test" "cargo test --release"

# 5. Check documentation builds
run_check "cargo doc" "cargo doc --no-deps --document-private-items" || true

# 6. TLA corpus analysis
echo -e "${BLUE}▶ Running: TLA Corpus Analysis${NC}"
if ./target/release/tlaplusplus analyze-tla \
    --module corpus/language_coverage/LanguageFeatureMatrix.tla \
    --config corpus/language_coverage/LanguageFeatureMatrix.cfg \
    > .ci/last-analysis.txt 2>&1; then

    # Check for improvement metrics
    SUPPORTED=$(grep "next_branch_probe_supported=" .ci/last-analysis.txt | cut -d= -f2)
    TOTAL=$(grep "next_branch_probe_total=" .ci/last-analysis.txt | cut -d= -f2)
    EXPR_OK=$(grep "expr_probe_ok=" .ci/last-analysis.txt | cut -d= -f2)
    EXPR_TOTAL=$(grep "expr_probe_total=" .ci/last-analysis.txt | cut -d= -f2)

    echo -e "${GREEN}  Branch support: ${SUPPORTED}/${TOTAL}${NC}"
    echo -e "${GREEN}  Expression eval: ${EXPR_OK}/${EXPR_TOTAL}${NC}"
    echo -e "${GREEN}✓ TLA Corpus Analysis passed${NC}"
    echo ""
else
    echo -e "${RED}✗ TLA Corpus Analysis failed${NC}"
    FAILED_CHECKS+=("TLA Corpus Analysis")
    echo ""
fi

# 7. Run simple TLA model check (if exists)
if [ -f "corpus/SimpleCounter.tla" ] && [ -f "corpus/SimpleCounter.cfg" ]; then
    run_check "SimpleCounter model check" \
        "./target/release/tlaplusplus run-tla \
            --module corpus/SimpleCounter.tla \
            --config corpus/SimpleCounter.cfg \
            --workers 2 \
            --work-dir ./.tlapp-ci-test \
            --clean-work-dir=true" || true
fi

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${RED}✗ ${#FAILED_CHECKS[@]} check(s) failed:${NC}"
    for check in "${FAILED_CHECKS[@]}"; do
        echo -e "${RED}  - ${check}${NC}"
    done
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    exit 1
fi
