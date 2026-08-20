#!/usr/bin/env bash
# Sets up a local .venv for Qwen3-TTS-GUI on macOS/Linux.
# Linux/Windows CUDA users: this installs CPU-only torch by default. See the
# README's "Install PyTorch for your machine" section to swap in a CUDA wheel.
# Intel macOS: torch is pinned to the last version with an x86_64 macOS wheel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OS="$(uname -s)"
ARCH="$(uname -m)"
MIN_PY_MINOR=10
WITH_FLASH_ATTN=0

# PyTorch has shipped no macosx_x86_64 wheel since 2.2.2 (Feb 2024) — Intel Macs
# are stuck on that pin, which in turn caps Python at 3.12. Not a bug, a ceiling.
INTEL_MAC=0
if [[ "$OS" == "Darwin" && "$ARCH" == "x86_64" ]]; then
  INTEL_MAC=1
fi
PIN_TORCH=2.2.2
PIN_TORCHVISION=0.17.2
PIN_TORCHAUDIO=2.2.2

for arg in "$@"; do
  case "$arg" in
    --with-flash-attn) WITH_FLASH_ATTN=1 ;;
    -h|--help)
      echo "Usage: $0 [--with-flash-attn]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

log() { printf '%s\n' "$*"; }
err() { printf 'Error: %s\n' "$*" >&2; exit 1; }

if [[ "$OS" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    err "Homebrew is required but not found. Install it from https://brew.sh and re-run this script."
  fi
fi

# Discover python3.X binaries actually on PATH rather than naming versions
# explicitly, so this doesn't go stale when a new Python is released.
PYTHON_BIN=""
PYTHON_VER=""
BEST_MINOR=-1
while IFS= read -r candidate; do
  [[ -n "$candidate" ]] || continue
  command -v "$candidate" >/dev/null 2>&1 || continue
  ver="$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)" || continue
  major="${ver%%.*}"
  minor="${ver##*.}"
  [[ "$major" -eq 3 && "$minor" -ge "$MIN_PY_MINOR" ]] || continue
  if [[ "$INTEL_MAC" -eq 1 ]]; then
    if [[ "$minor" -gt 12 ]]; then
      log "Skipping $candidate (Python $ver): Intel macOS has no PyTorch wheel newer than ${PIN_TORCH}, which tops out at Python 3.12."
      continue
    fi
  else
    # 3.13 is skipped, not just warned on: compiled CUDA extensions (e.g. flash-attn)
    # commonly don't have wheels for it yet. 3.10-3.12 and 3.14+ are fine.
    if [[ "$minor" -eq 13 ]]; then
      log "Skipping $candidate (Python $ver): known compiled-extension gaps, e.g. FlashAttention."
      continue
    fi
  fi
  if [[ "$minor" -gt "$BEST_MINOR" ]]; then
    BEST_MINOR="$minor"
    PYTHON_BIN="$candidate"
    PYTHON_VER="$ver"
  fi
done < <(compgen -c python3 | sort -u)

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ "$INTEL_MAC" -eq 1 ]]; then
    err "No usable Python found (need 3.10-3.12 on Intel macOS — PyTorch has no wheel newer than ${PIN_TORCH} for this architecture). Install one with: brew install python@3.11 — then re-run this script."
  elif [[ "$OS" == "Darwin" ]]; then
    err "No usable Python found (need 3.$MIN_PY_MINOR+, other than 3.13). Install one with: brew install python@3.11 — then re-run this script."
  else
    err "No usable Python found (need 3.$MIN_PY_MINOR+, other than 3.13). Install Python 3.10-3.12 or 3.14+ (e.g. via your distro's package manager) and re-run this script."
  fi
fi

log "Using $PYTHON_BIN (Python $PYTHON_VER)"
if [[ "$INTEL_MAC" -eq 1 ]]; then
  log "Intel macOS detected: PyTorch has no wheel newer than ${PIN_TORCH} for this architecture (Apple Silicon-only since then)."
  log "This project's models are validated against much newer torch — generation may fail at runtime even though install succeeds."
fi

# Homebrew's python@3.x formulas don't bundle tkinter (unlike the system Python),
# and several Linux distros split it into a separate package too. Checked against
# the base interpreter, before creating the venv, since tkinter is a compiled
# extension tied to the interpreter build rather than something pip can install.
attempt_tkinter_install() {
  if [[ "$OS" == "Darwin" ]]; then
    log "Attempting: brew install python-tk@${PYTHON_VER}"
    brew install "python-tk@${PYTHON_VER}" || true
  elif command -v apt-get >/dev/null 2>&1; then
    log "Attempting: sudo apt-get install -y python3-tk"
    sudo apt-get install -y python3-tk || true
  elif command -v dnf >/dev/null 2>&1; then
    log "Attempting: sudo dnf install -y python3-tkinter"
    sudo dnf install -y python3-tkinter || true
  elif command -v yum >/dev/null 2>&1; then
    log "Attempting: sudo yum install -y python3-tkinter"
    sudo yum install -y python3-tkinter || true
  elif command -v pacman >/dev/null 2>&1; then
    log "Attempting: sudo pacman -S --noconfirm tk"
    sudo pacman -S --noconfirm tk || true
  elif command -v zypper >/dev/null 2>&1; then
    log "Attempting: sudo zypper install -y python3-tk"
    sudo zypper install -y python3-tk || true
  elif command -v apk >/dev/null 2>&1; then
    log "Attempting: sudo apk add py3-tkinter"
    sudo apk add py3-tkinter || true
  else
    log "No supported package manager found for an automatic install."
  fi
}

if ! "$PYTHON_BIN" -c "import tkinter" >/dev/null 2>&1; then
  log "tkinter not found for $PYTHON_BIN — attempting to install it automatically ..."
  attempt_tkinter_install

  if ! "$PYTHON_BIN" -c "import tkinter" >/dev/null 2>&1; then
    if [[ "$OS" == "Darwin" ]]; then
      err "tkinter is still not available in $PYTHON_BIN after 'brew install python-tk@${PYTHON_VER}'. Install it manually and re-run this script."
    else
      err "tkinter is still not available in $PYTHON_BIN. Install your distro's tkinter package manually (e.g. python3-tk) and re-run this script."
    fi
  fi
  log "tkinter installed successfully."
fi

# The Python "sox" package (pulled in by qwen-tts) is just a wrapper around the
# system sox binary — pip installing it doesn't install the binary itself. Not a
# hard requirement for the GUI to start, so this warns rather than erroring out.
install_system_package() {
  local pkg="$1"
  if [[ "$OS" == "Darwin" ]]; then
    log "Attempting: brew install $pkg"
    brew install "$pkg" || true
  elif command -v apt-get >/dev/null 2>&1; then
    log "Attempting: sudo apt-get install -y $pkg"
    sudo apt-get install -y "$pkg" || true
  elif command -v dnf >/dev/null 2>&1; then
    log "Attempting: sudo dnf install -y $pkg"
    sudo dnf install -y "$pkg" || true
  elif command -v yum >/dev/null 2>&1; then
    log "Attempting: sudo yum install -y $pkg"
    sudo yum install -y "$pkg" || true
  elif command -v pacman >/dev/null 2>&1; then
    log "Attempting: sudo pacman -S --noconfirm $pkg"
    sudo pacman -S --noconfirm "$pkg" || true
  elif command -v zypper >/dev/null 2>&1; then
    log "Attempting: sudo zypper install -y $pkg"
    sudo zypper install -y "$pkg" || true
  elif command -v apk >/dev/null 2>&1; then
    log "Attempting: sudo apk add $pkg"
    sudo apk add "$pkg" || true
  else
    log "No supported package manager found for an automatic install."
  fi
}

if ! command -v sox >/dev/null 2>&1; then
  log "sox (system binary) not found — attempting to install it automatically ..."
  install_system_package sox
  if command -v sox >/dev/null 2>&1; then
    log "sox installed successfully."
  else
    log "Warning: sox is still not available. Install it manually if you hit audio-processing errors (e.g. brew install sox)."
  fi
fi

if [[ -d .venv ]]; then
  EXISTING_VER="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "")"
  if [[ "$EXISTING_VER" != "$PYTHON_VER" ]]; then
    log ".venv was built with Python ${EXISTING_VER:-unknown}, but ${PYTHON_VER} was selected this run — recreating .venv."
    rm -rf .venv
    "$PYTHON_BIN" -m venv .venv
  else
    log ".venv already exists (Python $EXISTING_VER), reusing it."
  fi
else
  log "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "Upgrading pip ..."
python -m pip install --upgrade pip

PIN_NUMBA=0.62.1
PIN_NUMPY=1.26.4

if [[ "$INTEL_MAC" -eq 1 ]]; then
  # torch 2.2.2 was built against NumPy 1.x's ABI and crashes on import under
  # NumPy 2.x ("_ARRAY_API not found"). requirements.txt installs unpinned numpy,
  # which resolves to 2.x, so this has to be pinned before anything imports torch.
  # Installed first so later installs see it already satisfied and don't upgrade it.
  log "Pinning numpy==${PIN_NUMPY} (torch ${PIN_TORCH} is not compatible with NumPy 2.x) ..."
  pip install "numpy==${PIN_NUMPY}"

  log "Installing PyTorch ${PIN_TORCH} (last Intel-macOS build; plain PyPI, no CUDA index) ..."
  pip install "torch==${PIN_TORCH}" "torchvision==${PIN_TORCHVISION}" "torchaudio==${PIN_TORCHAUDIO}"

  # librosa (pulled in by qwen-tts) requires numba, which requires llvmlite. Newer
  # releases of both dropped Intel-macOS wheels the same way torch did, which would
  # otherwise force a from-source llvmlite build needing cmake + a matching system
  # LLVM. Pinning numba here makes pip settle on the last compatible wheel pair
  # (numba 0.62.1 <-> llvmlite 0.45.1) instead of trying to build the latest.
  log "Pinning numba==${PIN_NUMBA} (last Intel-macOS wheel; avoids a from-source llvmlite build) ..."
  pip install "numba==${PIN_NUMBA}"
elif [[ "$OS" == "Darwin" ]]; then
  # macOS wheels (CPU/MPS, no CUDA variant exists) live on plain PyPI, not the
  # download.pytorch.org/whl/cpu index — that index is for Linux/Windows and its
  # macOS wheels stop at ancient versions.
  log "Installing PyTorch (macOS build) ..."
  pip install torch torchvision torchaudio
else
  log "Installing PyTorch (CPU build) ..."
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

log "Installing remaining dependencies ..."
pip install -r requirements.txt

if [[ "$WITH_FLASH_ATTN" -eq 1 ]]; then
  log "Installing FlashAttention (optional, GPU only, commonly fails on Windows) ..."
  pip install ninja packaging
  pip install flash-attn --no-build-isolation
fi

log ""
log "Setup complete."
if [[ "$INTEL_MAC" -eq 1 ]]; then
  log "Reminder: torch is pinned to ${PIN_TORCH} (Intel-macOS ceiling). If ./start.sh fails while loading"
  log "the model, that old torch is the likely cause — Apple Silicon or Linux/Windows won't have this issue."
elif [[ "$OS" != "Darwin" ]]; then
  log "If you have an NVIDIA GPU, replace the CPU torch build with a CUDA build:"
  log "  see https://pytorch.org/get-started/locally/"
fi
log ""
log "To run the app:"
log "  ./start.sh"
