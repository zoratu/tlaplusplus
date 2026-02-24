# Local CI System

This directory contains the local Continuous Integration (CI) system for tlaplusplus.

## Quick Start

### Install Git Hooks

```bash
.ci/install-hooks.sh
```

This installs:
- **pre-commit**: Fast checks before each commit (format, build, quick tests)
- **pre-push**: Full checks before push (clippy, tests, corpus analysis)

### Run Full CI Check

```bash
.ci/check.sh
```

Runs all quality checks:
1. `cargo fmt` - Code formatting
2. `cargo clippy` - Linter
3. `cargo build --release` - Release build
4. `cargo test` - Full test suite
5. `cargo doc` - Documentation build
6. TLA corpus analysis - Language feature coverage
7. SimpleCounter model check - End-to-end validation

### Run Benchmarks

```bash
.ci/bench.sh
```

Runs performance benchmarks on synthetic models.

## CI Checks

### Pre-Commit (Fast - ~10s)
- Code formatting check (auto-fixes if needed)
- Build check
- Quick unit tests

### Pre-Push (Comprehensive - ~30s)
- All pre-commit checks
- Clippy lints
- Full release build
- Complete test suite
- Documentation build
- Corpus analysis
- Model validation

## Skipping Hooks

```bash
# Skip pre-commit
git commit --no-verify

# Skip pre-push
git push --no-verify
```

## CI Artifacts

- `.ci/last-analysis.txt` - Latest corpus analysis output
- `.tlapp-ci-test/` - Temporary directory for CI model checks
- `.tlapp-bench/` - Temporary directory for benchmarks

## Adding New Checks

Edit `.ci/check.sh` and add your check using the `run_check` function:

```bash
run_check "My Check Name" "command to run"
```

## Local Development Workflow

1. **Install hooks once**: `.ci/install-hooks.sh`
2. **Make changes**: Edit code
3. **Commit**: Hooks run automatically
4. **Push**: Full CI runs automatically

For quick iteration without hooks:
```bash
# Development with fast feedback
cargo check && cargo test

# Pre-push dry run
.ci/check.sh
```

## Remote Testing

For many-core remote systems:
```bash
scripts/remote_bench.sh --max-x 20000 --max-y 20000 --max-sum 40000 --workers 96
```

## CI Philosophy

- **Fast feedback**: Pre-commit checks are quick (<10s)
- **Comprehensive validation**: Pre-push catches all issues
- **Local-first**: All checks run locally before pushing
- **No surprises**: Same checks that will run in GitHub Actions (when added)
