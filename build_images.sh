#!/usr/bin/env bash
set -euo pipefail
exec python3 "$(dirname "$0")/scripts/build_images.py" "$@"
