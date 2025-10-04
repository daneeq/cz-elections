#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs) || true
poetry run capture
