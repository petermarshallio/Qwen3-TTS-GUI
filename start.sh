#!/usr/bin/env bash
# Activates .venv and runs Qwen3-TTS-GUI. Run ./setup.sh first if .venv doesn't exist yet.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -d .venv ]]; then
  echo "Error: .venv not found. Run ./setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

exec python src/qwen_tts_gui.py
