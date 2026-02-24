#!/usr/bin/env bash
# Install git hooks for local CI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

cd "$PROJECT_ROOT"

# Check if .git exists
if [ ! -d ".git" ]; then
    echo "❌ Not a git repository. Run 'git init' first."
    exit 1
fi

echo "📦 Installing git hooks..."
echo ""

# Create hooks directory if it doesn't exist
mkdir -p "$HOOKS_DIR"

# Install pre-commit hook
echo "▶ Installing pre-commit hook..."
ln -sf "../../.ci/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"
echo "✅ pre-commit hook installed"

# Install pre-push hook
echo "▶ Installing pre-push hook..."
ln -sf "../../.ci/pre-push" "$HOOKS_DIR/pre-push"
chmod +x "$HOOKS_DIR/pre-push"
echo "✅ pre-push hook installed"

echo ""
echo "✅ Git hooks installed successfully!"
echo ""
echo "Hooks installed:"
echo "  • pre-commit: Fast checks (format, build, quick tests)"
echo "  • pre-push: Full CI checks (clippy, tests, corpus analysis)"
echo ""
echo "To skip hooks: git commit --no-verify or git push --no-verify"
