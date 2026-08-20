# Qwen3-TTS-GUI

An intuitive GUI app for Qwen3's TTS model

## Features

The app is built around a screenplay metaphor: you write a **script** — a
stack of dialogue lines, each spoken by an **actor** (a voice) — and generate
it as one audio file.

- **Script canvas**: build a script as dialogue blocks (actor, optional tone,
  the line itself), or paste a plain-text script — your own, or one written
  by an LLM — via **Import Script**
- **Cast sidebar**: every available actor (a preset voice, a cloned
  recording, or a designed voice), with a play button to preview each one
- **Casting a new actor**: bring an existing recording, record one live with
  a guided script, or design a voice from a natural-language description
- **Settings**: device (CUDA/CPU), and model size/dtype for training and
  generation independently — set once, not per script

## Installation

This project is a local Python app (tkinter GUI) that uses `qwen-tts` + PyTorch.

### Prerequisites

- **Python**: Python **3.10+** recommended (64-bit).
  - For Windows, **Python 3.11 is the safest choice**. Some ML packages (especially compiled CUDA extensions like FlashAttention) may not support Python 3.13 yet.
- **Git**: only needed if you’re cloning this repo.
- **tkinter**: included with most Python installs.
  - On some Linux distros you may need to install it separately (e.g. `python3-tk`).
- **Audio I/O**: recording/playback uses `sounddevice` (PortAudio) and `soundfile`.
  - On Windows, wheels usually work out-of-the-box. If you see build errors, install **Microsoft C++ Build Tools** and retry.
- **GPU (optional)**: an NVIDIA GPU can be used via CUDA, but **you do not need CUDA** for CPU mode.

### Install (from source)

1. Get the code (skip if you already have it).

```bash
git clone <REPO_URL>
cd Qwen3-TTS-GUI
```

1. Create and activate a virtual environment (recommended).

Windows (CMD):

```cmd
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

1. Install **PyTorch** for your machine.

- **CPU-only** (works everywhere):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

- **CUDA (NVIDIA GPU)**: install the CUDA-enabled wheel that matches your setup. Follow the official selector:
  - PyTorch install page: [PyTorch “Get Started”](https://pytorch.org/get-started/locally/)
  - CUDA downloads (optional): [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

> [!IMPORTANT]
> If you’re using CUDA, your PyTorch wheel must match the CUDA runtime you intend to use (e.g. cu118 / cu121). If CUDA is installed but PyTorch is CPU-only, the app will run in CPU mode.

1. Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

### Optional: FlashAttention (GPU speed-up)

If you have a compatible GPU setup, FlashAttention can improve inference speed. It’s optional and the app will fall back automatically if it’s not installed.

```bash
pip install flash-attn --no-build-isolation
```

> [!NOTE]
> FlashAttention installation commonly fails on Windows. If it fails, skip it—everything still works (just slower on GPU).

If you *do* want to try installing FlashAttention on Windows, make sure all of the following are true:

- **You are using Python 3.10 or 3.11** (Python 3.13 is often unsupported for CUDA extensions).
- **MSVC build tools are installed** (so `cl.exe` exists):
  - Install “Visual Studio Build Tools” and select **Desktop development with C++**, **MSVC v143**, and the **Windows SDK**.
- **Your CUDA toolchain matches your PyTorch CUDA build**:
  - Check: `python -c "import torch; print(torch.version.cuda)"`
  - If you install a CUDA Toolkit, install the *same* major/minor version (e.g. 12.1 / 12.6). Mismatches commonly cause build failures.
  - Note: you usually **don’t need** the CUDA Toolkit just to *run* PyTorch; you mainly need it to *build* CUDA extensions.
- Install build helpers first:

```bash
pip install ninja packaging
```

### Optional: Build an executable (PyInstaller)

If you want to package the app into an `.exe`, install build requirements and then run PyInstaller.

```bash
pip install -r requirements_build.txt
pyinstaller --noconfirm --onefile --windowed src/qwen_tts_gui.py
```

### Troubleshooting

- **CUDA selected but not available**:
  - Confirm you installed a CUDA-enabled PyTorch wheel (see the PyTorch selector above).
  - Update your NVIDIA driver, then restart your terminal (so PATH changes take effect).
- **`CUDA error: no kernel image is available for execution on the device`**:
  - This almost always means **your GPU’s compute capability isn’t supported by the PyTorch CUDA build you installed** (often happens with very old or very new GPUs).
  - Check what PyTorch sees:

```bash
python -c "import torch; print('torch', torch.__version__); print('torch cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('cc', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
```

  - Fix options:
    - **If your GPU is old** (low compute capability): use **CPU mode** (recommended) or install an older PyTorch version that still supports your GPU.
    - **If your GPU is new** and you don’t see your compute capability in the “supported CUDA capabilities” list: install a newer PyTorch CUDA wheel (or nightly) that includes kernels for your GPU.
    - **RTX 50‑series / Blackwell (`sm_120`, compute capability 12.0)**: install a CUDA **12.8+** PyTorch wheel (example: `cu128`).

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

  - Debugging tip (makes CUDA errors point to the real call site):
    - PowerShell: `$env:CUDA_LAUNCH_BLOCKING=1`
    - bash: `export CUDA_LAUNCH_BLOCKING=1`
- **FlashAttention install fails on Windows** (common):
  - The simplest fix is to **skip FlashAttention**; the app will fall back automatically.
  - If you need it, use **Python 3.11**, install **MSVC Build Tools**, and align your CUDA Toolkit version with `torch.version.cuda`.
  - If you want the least pain, install and run in **WSL2 (Ubuntu)** instead of native Windows.
- **`HTTP Error 404` during FlashAttention install/build**:
  - This typically happens during a build-time download step. Retry the install, and ensure your network/proxy isn’t blocking GitHub/CDN downloads.
  - If it keeps happening, treat it as a sign to **skip FlashAttention on Windows** (or use WSL2).
- **Microphone / audio device errors**:
  - Try switching the device to **CPU** first (to confirm the model loads), then troubleshoot audio separately.
  - On Windows, ensure microphone permissions are enabled for your Python installation/app.
- **`ModuleNotFoundError: _tkinter` (Linux)**:
  - Install your distro’s tkinter package (often `python3-tk`), then re-run the app.

#### Known-good Windows setups

- **Most reliable (no CUDA build headaches)**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

- **GPU (NVIDIA) without FlashAttention**:
  - Pick a stable PyTorch CUDA wheel (commonly `cu121` or `cu126` from the PyTorch selector), then install project deps. If you don’t plan to compile CUDA extensions, you can usually skip installing the CUDA Toolkit.

## Usage

Run the GUI application:

```bash
python src/qwen_tts_gui.py
```

## How to Use

### Casting a new actor

1. Click **+ New Actor** in the Cast sidebar (or "+ New Actor…" from the actor
   picker when building a script)
2. Enter a name
3. Choose how to give it a voice:
   - **From Audio File + Transcript**: browse for a WAV file and enter the exact
     transcript (or check **Quick clone** to skip the transcript, faster but lower
     fidelity)
   - **Record Audio**: use the built-in recorder, reading one of the offered scripts
     aloud
   - **Design from Description**: describe the voice in plain language (e.g. "warm
     elderly British female voice") — no recording needed
4. Click **Train Voice**. The actor appears in the Cast sidebar once done,
   with a preview clip generated automatically.

### Building and generating a script

1. Click **+ Add Line**, pick an actor (or cast a new one on the spot), and
   type their line. Repeat for as many lines/actors as the script needs —
   presets and cloned/designed voices can be mixed freely.
2. Optionally give a preset-voiced line a **tone** (e.g. "cheerfully") —
   voice-clone lines don't support this.
3. Or skip steps 1–2 and use **Import Script** to paste a plain-text script
   instead — screenplay format: a `NAME` line (optionally `NAME (tone)`), the
   dialogue below it, a blank line before the next actor. Works well with
   scripts written by an LLM.
4. Pick a Language and Output Format, then click **Generate Speech** —
   you'll be asked where to save the result.

## Notes

- For CUDA/GPU usage, ensure you have CUDA-compatible PyTorch installed
- Flash Attention 2 will be used automatically if available, otherwise falls back to SDPA
- CPU mode uses float32 precision and eager attention implementation
- GPU mode uses bfloat16 precision for better performance

This readme and the comments in the code were created with the assistance of AI