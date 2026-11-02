#!/bin/sh
# Lint, type-check, and test the package and documentation examples.
set -eu
project_dir="$(dirname "$0")/.."

uv run --directory "$project_dir" ruff check src tests
uv run --directory "$project_dir" pyrefly check
uv run --directory "$project_dir" python -m pytest
