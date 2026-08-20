import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import zlib
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import ClassVar

import av
import numpy as np
import sounddevice as sd
import soundfile as sf
import sv_ttk
import torch
from PIL import Image, ImageDraw, ImageTk
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # PyInstaller's bundled CUDA DLLs live under _MEIPASS, which isn't on PATH by
    # default — without this, torch.cuda.is_available() reports False in a frozen
    # build even though the DLLs are right there.
    torch_lib_path = os.path.join(sys._MEIPASS, "torch", "lib")
    current_path = os.environ.get("PATH", "")
    if os.path.exists(torch_lib_path):
        current_path = torch_lib_path + os.pathsep + current_path
    if sys._MEIPASS not in current_path:
        current_path = sys._MEIPASS + os.pathsep + current_path
    os.environ["PATH"] = current_path

    try:
        import torch._C

        _ = torch.cuda.is_available()  # re-check now that PATH includes the DLLs
    except Exception:
        pass

SAMPLE_RATE = 44100
CHANNELS = 1

# Default output location for trained voices, generated speech, and mic
# recordings — gitignored, so nothing written by the app lands in the repo.
if getattr(sys, "frozen", False):
    _base_dir = os.path.dirname(sys.executable)
else:
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_base_dir, "local")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cached "what does this actor sound like" clips for the Cast sidebar's play button —
# generated proactively (right after a voice is cast, and via a startup sweep that
# backfills anything missing, including the 9 presets) rather than on demand, so
# clicking play is always instant rather than triggering a model load.
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "previews")
os.makedirs(PREVIEW_DIR, exist_ok=True)
PREVIEW_TEXT = "Hello there — this is a quick preview of my voice."


def preview_path_for(key):
    return os.path.join(PREVIEW_DIR, f"{key}_preview.wav")


MODEL_SIZES = ("1.7B", "0.6B")
MODEL_REPOS_BASE = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
MODEL_REPOS_CUSTOM_VOICE = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}
# VoiceDesign has no 0.6B release yet per the model card's own table, and isn't wired
# up in this app at all yet (see TODO.md) — not offered as a size choice.

# Explicit dtype choices for Configure — CPU always forces float32 regardless (see
# get_model_config), so these only take effect when actually running on CUDA.
DTYPE_OPTIONS = ("bfloat16", "float32", "float16")
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}

# Fixed speaker list for MODEL_REPOS_CUSTOM_VOICE (from the model card). Not fetched
# dynamically via AutoConfig, which would need network/HF-cache access just to
# populate a dropdown before the user has done anything.
PRESET_VOICES = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Read-aloud reference scripts offered on the Train Voice > Record tab, ordered
# shortest-to-longest. "Original" is this app's very first script (kept for anyone
# already used to it); the other four grow in length, phonetic variety (plosives,
# fricatives, nasals, sibilants, vowel spread), and sentence-type variety (statement,
# question, exclamation) — "Comprehensive" is deliberately the longest and richest,
# repeating most sound classes more than once.
TRAINING_SCRIPTS = {
    "Original (~35s)": (
        "On a warm Saturday morning, the quick brown fox jumped over several lazy "
        "dogs while distant musicians played jazzy tunes near the quiet park. "
        "People checked their phones, argued about numbers, dates, and prices, and "
        "casually mentioned names like Alex, Jordan, and Taylor. A cyclist shouted "
        "warnings, a train horn echoed, and someone asked, ‘Why does this even "
        "matter?’ as rain began falling lightly at exactly 9:47 a.m., changing "
        "plans, moods, and expectations all at once."
    ),
    "Quick (~5s)": "The quick brown fox jumped swiftly over three lazy sleeping dogs.",
    "Short (~10s)": (
        "Good morning! I brewed a fresh pot of coffee and opened the windows. Could "
        "you check if it's still raining outside before we leave at nine?"
    ),
    "Standard (~18s)": (
        "Good morning! I just got back from a walk around the lake, and it's "
        "surprisingly warm for October. Did you know our flight leaves at 6:45, "
        "not 7:15? We should double-check the tickets before Thursday — just "
        "to be safe, and maybe grab a quick bite first."
    ),
    "Comprehensive (~55s)": (
        "Good afternoon! My name is Jonathan, and I'd like to tell you about a "
        "rather peculiar Thursday. It started at exactly 7:15, when a sudden "
        "thunderstorm rattled the windows and startled the neighbor's three dogs. "
        "‘Why does this always happen on my day off?’ I muttered, "
        "grabbing a thick jacket and rushing outside. By noon, the sky had cleared "
        "completely — sunshine, blue skies, and a gentle breeze replaced the "
        "chaos. We quickly rescheduled the picnic to 2:30, bought fresh "
        "strawberries, cheese, and sparkling lemonade, and invited twelve friends. "
        "Should we also bring umbrellas, just in case? Absolutely — better "
        "safe than soaked! By evening, laughter echoed through the garden as "
        "fireworks lit up the darkening sky, and everyone agreed: sometimes the "
        "strangest days become the best memories."
    ),
}

# VoiceDesign has no 0.6B release (see MODEL_REPOS_BASE/MODEL_REPOS_CUSTOM_VOICE) and
# isn't offered a size choice in Configure — this is the only checkpoint there is.
MODEL_REPO_VOICE_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# The reference clip qwen_tts's own docs recommend synthesizing once with VoiceDesign
# before feeding it into create_voice_clone_prompt, same as any other reference
# recording — reusing "Standard" rather than exposing a picker, since nobody has to
# read this aloud themselves.
VOICE_DESIGN_REFERENCE_TEXT = TRAINING_SCRIPTS["Standard (~18s)"]


def list_custom_voices():
    """Voice names available in the local cache (OUTPUT_DIR/*.pt)."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(".pt")
    )


def list_voice_recordings(voice_name):
    """Recorded takes for `voice_name`, sorted oldest to newest
    (OUTPUT_DIR/{voice_name}/mic_*.wav — the filename timestamp sorts lexically)."""
    if not voice_name:
        return []
    voice_dir = os.path.join(OUTPUT_DIR, voice_name)
    if not os.path.isdir(voice_dir):
        return []
    return sorted(
        os.path.join(voice_dir, f)
        for f in os.listdir(voice_dir)
        if f.lower().endswith(".wav")
    )


def build_voice_lookup():
    """Bare voice name (lowercased) -> (kind, key), for resolving an actor name to a
    voice. Presets are added first so a same-named custom voice takes precedence."""
    lookup = {}
    for speaker in PRESET_VOICES:
        lookup[speaker.lower()] = ("preset", speaker)
    for name in list_custom_voices():
        lookup[name.lower()] = ("custom", name)
    return lookup


# A colored dot + name is how the script canvas and Cast sidebar show "who's talking"
# without reading — every reference to a given actor uses the same color, for the life
# of the process. zlib.crc32 (not the builtin hash(), which is salted per-process) so
# the mapping is stable across runs, not just within one.
ACTOR_PALETTE = [
    "#4FB6D6",  # cyan
    "#9C8FE8",  # violet
    "#E0679E",  # magenta
    "#5FBE8B",  # green
    "#B0AE4A",  # olive
    "#E0645A",  # red
    "#E0973F",  # amber
    "#6E93EF",  # blue
]


def color_for_actor(key):
    """Deterministic display color for actor `key` — same actor, same color, always."""
    return ACTOR_PALETTE[zlib.crc32(key.lower().encode()) % len(ACTOR_PALETTE)]


# sv-ttk's light theme paints the window at #fafafa and its text at #1c1c1c. THEME_FG
# is that same foreground, for the places we set a widget's text color explicitly
# (rather than leaving it to the ttk style) and so must match the theme by hand.
THEME_FG = "#1c1c1c"
# These give each script block a visibly shaded "card" surface, so a block (and the
# text box inside it) reads as a distinct clickable area rather than blending into
# the canvas background.
SCRIPT_BLOCK_BG = "#eeeeee"
SCRIPT_BLOCK_BORDER = "#c9c9c9"
# Typewriter feel for the script itself — actor name, tone, and dialogue text all
# share this fixed-width family. "Courier" is one of Tk's generic logical font
# names, so it resolves to a real monospace font on any platform.
SCRIPT_FONT_FAMILY = "Courier"
SCRIPT_BLOCK_FOCUS_BORDER = "#005fb8"  # sv-ttk's accent color


# --- Cast sidebar icons (play/spinner, edit, delete): drawn as small fixed-size
# bitmaps rather than font glyphs. Text characters (▶, 🗑, ✎, and the arrows used
# for the spinner) don't render at a consistent size across fonts/platforms —
# notably emoji-presentation glyphs like 🗑, which come out visibly wider than
# plain symbol glyphs at the same font size — so no font/padding tuning can make
# a row of them look uniform. A hand-drawn bitmap has no such ambiguity: every
# icon and every spinner frame is exactly ICON_SIZE square, always. ---

ICON_SIZE = 16
_ICON_SUPERSAMPLE = 4  # draw this many times larger, then downsample for smooth
# (antialiased) edges — ImageDraw's own shapes are hard-edged at low resolution.


def _icon_canvas():
    size = ICON_SIZE * _ICON_SUPERSAMPLE
    return Image.new("RGBA", (size, size), (0, 0, 0, 0)), size


def _finish_icon(img):
    return img.resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS)


def _draw_play_icon(color):
    img, size = _icon_canvas()
    draw = ImageDraw.Draw(img)
    margin = size * 0.28
    draw.polygon(
        [(margin, size * 0.15), (margin, size * 0.85), (size - margin, size / 2)],
        fill=color,
    )
    return _finish_icon(img)


def _draw_edit_icon(color):
    img, size = _icon_canvas()
    draw = ImageDraw.Draw(img)
    width = int(size * 0.12)
    draw.line(
        [(size * 0.2, size * 0.8), (size * 0.7, size * 0.3)], fill=color, width=width
    )
    draw.polygon(
        [
            (size * 0.68, size * 0.32),
            (size * 0.85, size * 0.15),
            (size * 0.85, size * 0.32),
        ],
        fill=color,
    )
    return _finish_icon(img)


def _draw_delete_icon(color):
    img, size = _icon_canvas()
    draw = ImageDraw.Draw(img)
    width = int(size * 0.07)
    draw.line(
        [(size * 0.2, size * 0.28), (size * 0.8, size * 0.28)], fill=color, width=width
    )
    draw.line(
        [(size * 0.38, size * 0.16), (size * 0.62, size * 0.16)],
        fill=color,
        width=width,
    )
    draw.line(
        [
            (size * 0.26, size * 0.28),
            (size * 0.3, size * 0.85),
            (size * 0.7, size * 0.85),
            (size * 0.74, size * 0.28),
        ],
        fill=color,
        width=width,
        joint="curve",
    )
    for fx in (0.4, 0.5, 0.6):
        draw.line(
            [(size * fx, size * 0.38), (size * fx, size * 0.75)],
            fill=color,
            width=width,
        )
    return _finish_icon(img)


def _build_spinner_icons(color, frame_count=8):
    """One upward arrow, rotated into `frame_count` evenly-spaced frames — rotated
    before downsampling (not after), so every frame keeps the same antialiased
    quality as the base drawing."""
    img, size = _icon_canvas()
    draw = ImageDraw.Draw(img)
    width = int(size * 0.1)
    cx = size / 2
    draw.line([(cx, size * 0.8), (cx, size * 0.25)], fill=color, width=width)
    draw.polygon(
        [
            (cx - size * 0.18, size * 0.35),
            (cx + size * 0.18, size * 0.35),
            (cx, size * 0.15),
        ],
        fill=color,
    )
    return [
        _finish_icon(img.rotate(-i * (360 / frame_count), resample=Image.BICUBIC))
        for i in range(frame_count)
    ]


# Screenplay-format headers: "NAME" or "NAME (tone)".
_SCREENPLAY_HEADER_RE = re.compile(r"^([^(]+?)(?:\s*\(([^)]*)\))?$")


def parse_screenplay_text(text, lookup):
    """Parse a pasted plain-text script — screenplay format: a NAME (optionally
    "NAME (tone)"), one or more lines of dialogue, a blank line before the next
    actor — into the same `{"kind", "key", "tone", "text"}` shape the block editor
    produces, so an imported script drops straight into `self.script_lines`.

    This is the format an LLM naturally writes a script in when asked to, and reads
    naturally to a person too — no bracket-marker syntax to learn.

    Returns:
        (lines, errors). `errors` lists any actor name that doesn't resolve via
        `lookup` — import is blocked on any error, the same way the block editor is
        implicitly protected (it can only ever reference a real actor, being
        picker-driven).
    """
    lines = []
    errors = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        block_lines = block.splitlines()
        header = block_lines[0].strip()
        dialogue = "\n".join(line.strip() for line in block_lines[1:]).strip()

        match = _SCREENPLAY_HEADER_RE.match(header)
        if not match:
            errors.append(f'Could not read speaker line: "{header}"')
            continue

        name = match.group(1).strip()
        tone = (match.group(2) or "").strip()
        resolved = lookup.get(name.lower())
        if resolved is None:
            errors.append(f"Unknown actor: {name}")
            continue
        if not dialogue:
            continue

        kind, key = resolved
        lines.append({"kind": kind, "key": key, "tone": tone, "text": dialogue})

    return lines, errors


def group_segments_by_kind(segments):
    """Group parsed (kind, key, text, instruct) segments by kind, keeping each
    segment's original index (needed to reassemble in order later).

    Returns:
        (groups, kinds_needed): `groups[kind]` is a list of (index, key, text,
        instruct); `kinds_needed` orders "custom" (Base model) before "preset"
        (CustomVoice model) — the order those models get loaded in.
    """
    groups = {}
    for i, (kind, key, seg_text, instr) in enumerate(segments):
        groups.setdefault(kind, []).append((i, key, seg_text, instr))
    kinds_needed = [k for k in ("custom", "preset") if groups.get(k)]
    return groups, kinds_needed


def _is_cuda_compat_error(exc):
    """Whether `exc` looks like a CUDA kernel/device compatibility error (as opposed to
    some other RuntimeError) — the case worth silently falling back to CPU for."""
    error_str = str(exc).lower()
    return "cuda" in error_str and (
        "kernel" in error_str or "no kernel image" in error_str or "device" in error_str
    )


def encode_audio(samples, sr, output_file, output_format):
    """Encode a float32 numpy array to mp3/m4a via PyAV (bundles its own
    encoders, no system ffmpeg needed). samples is (n,) mono or (n, channels)."""
    if samples.ndim == 1:
        data = samples.reshape(1, -1)
        layout = "mono"
    else:
        data = samples.T
        layout = "stereo" if data.shape[0] == 2 else "mono"

    container_format = "mp3" if output_format == "mp3" else "mp4"
    codec_name = "libmp3lame" if output_format == "mp3" else "aac"

    container = av.open(output_file, mode="w", format=container_format)
    stream = container.add_stream(codec_name, rate=sr)
    stream.layout = layout
    stream.bit_rate = 192000

    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(data.astype(np.float32)), format="fltp", layout=layout
    )
    frame.sample_rate = sr

    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)

    container.close()


def reveal_in_file_manager(path):
    """Open the OS file manager with `path` selected (best-effort, ignores failures)."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", path], check=False)
        elif sys.platform.startswith("win"):
            subprocess.run(["explorer", "/select,", path], check=False)
        else:
            subprocess.run(["xdg-open", os.path.dirname(path)], check=False)
    except Exception:
        pass


class ProgressPanel:
    """A live step checklist + progress bar + elapsed timer for a background job
    whose overall shape (a known sequence of named steps) is known up front, but
    whose per-step duration isn't — there's no token-level progress hook to report,
    so each step's bar segment pulses (indeterminate) while active and only snaps to
    a real, filled fraction once that step actually finishes."""

    PENDING, ACTIVE, DONE, FAILED = "pending", "active", "done", "failed"
    GLYPHS: ClassVar[dict] = {PENDING: "○", ACTIVE: "●", DONE: "✓", FAILED: "✗"}
    COLORS: ClassVar[dict] = {
        PENDING: "gray",
        ACTIVE: "#2b6cb0",
        DONE: "#2f855a",
        FAILED: "#c53030",
    }

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)

        self.bar = ttk.Progressbar(self.frame, mode="determinate")
        self.bar.pack(fill=tk.X, pady=(0, 6))

        self.steps_frame = ttk.Frame(self.frame)
        self.steps_frame.pack(fill=tk.X)

        self.elapsed_label = ttk.Label(
            self.frame, foreground="gray", font=("TkDefaultFont", 9)
        )
        self.elapsed_label.pack(anchor=tk.E, pady=(4, 0))

        self._rows = []
        self._start_time = None
        self._timer_id = None

    def set_steps(self, descriptions):
        """(Re)build the checklist for a new run; every step starts pending."""
        for row in self._rows:
            row["frame"].destroy()
        self._rows = []

        self.bar.config(mode="determinate", maximum=max(len(descriptions), 1), value=0)
        for desc in descriptions:
            row_frame = ttk.Frame(self.steps_frame)
            row_frame.pack(fill=tk.X, anchor=tk.W)
            glyph = ttk.Label(
                row_frame,
                text=self.GLYPHS[self.PENDING],
                width=2,
                foreground=self.COLORS[self.PENDING],
            )
            glyph.pack(side=tk.LEFT)
            label = ttk.Label(
                row_frame, text=desc, foreground=self.COLORS[self.PENDING]
            )
            label.pack(side=tk.LEFT)
            self._rows.append({"frame": row_frame, "glyph": glyph, "label": label})

    def _set_state(self, index, state):
        row = self._rows[index]
        row["glyph"].config(text=self.GLYPHS[state], foreground=self.COLORS[state])
        row["label"].config(foreground=self.COLORS[state])

    def start_step(self, index):
        """Mark step `index` active — pulses the bar, since we don't know how long
        this particular step will take."""
        self._set_state(index, self.ACTIVE)
        self.bar.config(mode="indeterminate")
        self.bar.start(80)

    def complete_step(self, index):
        """Mark step `index` done and advance the bar to a real, filled fraction."""
        self.bar.stop()
        self.bar.config(mode="determinate", value=index + 1)
        self._set_state(index, self.DONE)

    def fail_step(self, index):
        self.bar.stop()
        self.bar.config(mode="determinate")
        self._set_state(index, self.FAILED)

    def show(self, before=None):
        self.frame.pack(fill=tk.X, padx=20, pady=(0, 10), before=before)
        self._start_time = time.monotonic()
        self._tick()

    def hide(self):
        if self._timer_id is not None:
            self.frame.after_cancel(self._timer_id)
            self._timer_id = None
        self.bar.stop()
        self.frame.pack_forget()

    def _tick(self):
        elapsed = int(time.monotonic() - self._start_time)
        self.elapsed_label.config(text=f"Elapsed: {elapsed // 60}:{elapsed % 60:02d}")
        self._timer_id = self.frame.after(1000, self._tick)


class CollapsibleSection:
    """A clickable "▸ Title" header that shows/hides a body frame — Tkinter has no
    built-in disclosure widget. Collapsed by default; the body is only packed once
    expanded, so it takes no space until opened."""

    def __init__(self, parent, title):
        self._title = title
        self.expanded = False
        self.header = ttk.Frame(parent)
        self.toggle_label = ttk.Label(
            self.header, text=f"▸ {title}", foreground="#2b6cb0", cursor="hand2"
        )
        self.toggle_label.pack(anchor=tk.W)
        self.toggle_label.bind("<Button-1>", lambda e: self.toggle())
        self.body = ttk.Frame(parent)

    def toggle(self):
        self.expanded = not self.expanded
        self.toggle_label.config(text=f"{'▾' if self.expanded else '▸'} {self._title}")
        if self.expanded:
            self.body.pack(fill=tk.X, pady=(5, 0), after=self.header)
        else:
            self.body.pack_forget()

    def pack(self, **kwargs):
        self.header.pack(**kwargs)


class ScrollableFrame:
    """A vertically-scrolling Frame — Tkinter has no built-in one. The standard
    recipe: a Canvas + Scrollbar, with an inner Frame (`self.body`) placed on the
    canvas; widgets are packed into `body` exactly as they would be into any other
    Frame. Backs both the Cast sidebar and the script canvas."""

    def __init__(self, parent):
        self.outer = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.outer, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(
            self.outer, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.body = ttk.Frame(self.canvas)
        self._body_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self._body_id, width=e.width),
        )
        # Bound directly on canvas + body (not via bind_all on Enter/Leave — Tk fires
        # <Leave> on the canvas the instant the pointer crosses onto any child widget
        # stacked on top of it, which would silently kill scrolling over almost
        # everything, and bind_all from two simultaneously-visible ScrollableFrames
        # would fight over the same global binding).
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.body.bind("<MouseWheel>", self._on_mousewheel)

    def bind_scroll(self, widget):
        """Opt a widget — and, recursively, every descendant it already has — into
        scrolling this frame directly. Call on a row/block's outer Frame after all
        of its children are built, so scrolling works with the pointer anywhere over
        the row (its labels, buttons, entries, text), not just gaps between them."""
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        for child in widget.winfo_children():
            self.bind_scroll(child)

    def _on_mousewheel(self, event):
        # macOS reports delta already in small per-notch units; Windows reports it in
        # multiples of 120 — dividing by 120 on macOS would round almost every
        # scroll down to zero.
        amount = event.delta if sys.platform == "darwin" else event.delta / 120
        self.canvas.yview_scroll(int(-1 * amount), "units")

    def pack(self, **kwargs):
        self.outer.pack(**kwargs)


class QwenTTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Studio — Qwen3 TTS")
        self.root.geometry("1280x860")
        self.root.minsize(960, 600)

        # "Block.*" styles give a script block's Frame/Label/Entry a matching lighter
        # background (SCRIPT_BLOCK_BG) — each unspecified option (font, foreground,
        # etc.) falls back to its base TFrame/TLabel/TEntry default since ttk treats
        # an unregistered "Block.TWidget" name as a variant of TWidget.
        style = ttk.Style()
        style.configure("Block.TFrame", background=SCRIPT_BLOCK_BG)
        style.configure("Block.TLabel", background=SCRIPT_BLOCK_BG)
        style.configure("Block.TEntry", fieldbackground=SCRIPT_BLOCK_BG)
        # Small, square-ish footprint for the Cast sidebar's per-row icon buttons —
        # the default TButton padding is sized for text labels, not a 16x16 image.
        style.configure("Icon.TButton", padding=3)

        # Built once a real Tk root exists (ImageTk.PhotoImage needs one) and kept
        # referenced on self for the widgets' lifetime — Tk drops an image the
        # instant nothing still refers to its PhotoImage.
        self.icon_play = ImageTk.PhotoImage(_draw_play_icon(THEME_FG))
        self.icon_edit = ImageTk.PhotoImage(_draw_edit_icon(THEME_FG))
        self.icon_delete = ImageTk.PhotoImage(_draw_delete_icon(THEME_FG))
        self.icon_spinner_frames = [
            ImageTk.PhotoImage(frame) for frame in _build_spinner_icons(THEME_FG)
        ]

        self.device_type = tk.StringVar(value="cuda")
        default_dtype = "bfloat16" if torch.cuda.is_available() else "float32"
        self.train_model_size = tk.StringVar(value="1.7B")
        self.train_dtype = tk.StringVar(value=default_dtype)
        self.generate_model_size = tk.StringVar(value="1.7B")
        self.generate_dtype = tk.StringVar(value=default_dtype)
        self.recording = False
        self.audio_chunks = []
        self.recording_map = {}  # filename -> full path, for the current voice's takes
        self.generate_cancel_event = threading.Event()

        # The script: an ordered list of {"kind", "key", "tone", "text"} dicts, one
        # per dialogue block. Generation converts this straight into the
        # (kind, key, text, instruct) tuples _generate_speech_thread already expects
        # — no text parsing needed for anything built through the block editor.
        self.script_lines = []
        # Set only while the New Actor dialog was opened *from* the actor picker (so
        # the freshly cast actor gets selected into whichever line asked for it),
        # rather than from the Cast sidebar's own "+ New Actor" button.
        self._new_actor_on_created = None
        # None in "new actor" mode; the actor's current name in "edit" mode (opened
        # via edit_voice) — train_voice checks this to allow retraining under the
        # same name without a collision error, and to rename-on-disk rather than
        # just create a second voice if the name's changed instead.
        self._editing_voice_key = None

        # Cast sidebar play buttons awaiting a preview show a rotating arrow instead
        # of ▶ (and are disabled meanwhile) — refresh_cast_sidebar repopulates this
        # list each time it rebuilds the rows; this just keeps every currently-shown
        # one animated in lockstep.
        self._preview_spinner_frame = 0
        self._preview_spinner_buttons = []
        self._tick_preview_spinner()

        self.build_top_bar(self.root)

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)
        self.build_cast_sidebar(body)

        # Both the script canvas and the generation bar live in this frame, so
        # set_use_busy's _set_frame_enabled(self.use_frame, ...) disables the whole
        # right-hand side (not the Cast sidebar) while a generation is in flight.
        self.use_frame = ttk.Frame(body)
        self.use_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.build_script_canvas(self.use_frame)
        self.build_generation_bar(self.use_frame)

        # Built once and shown/hidden (not torn down and rebuilt) on every open.
        self.build_new_actor_dialog()
        self.build_settings_dialog()
        self.build_import_dialog()

        self.refresh_cast_sidebar()
        # Backfill any voice (including the 9 presets, on first-ever run) missing a
        # cached preview, without delaying the window itself appearing.
        threading.Thread(target=self.sweep_missing_previews, daemon=True).start()

    # --- Placeholder text + live counters, shared by the Transcript / Text to
    # Generate / Instruction boxes ---

    def add_placeholder_text(self, widget, placeholder):
        """Grey hint text for a ScrolledText widget; cleared on focus, restored when empty."""
        widget.showing_placeholder = True
        widget.insert("1.0", placeholder)
        widget.config(foreground="grey")

        def on_focus_in(event):
            if widget.showing_placeholder:
                widget.delete("1.0", tk.END)
                widget.config(foreground=THEME_FG)
                widget.showing_placeholder = False

        def on_focus_out(event):
            if not widget.get("1.0", "end-1c").strip():
                widget.insert("1.0", placeholder)
                widget.config(foreground="grey")
                widget.showing_placeholder = True

        widget.bind("<FocusIn>", on_focus_in, add="+")
        widget.bind("<FocusOut>", on_focus_out, add="+")

    def add_placeholder_entry(self, widget, placeholder):
        """Grey hint text for a single-line Entry widget."""
        widget.showing_placeholder = True
        widget.insert(0, placeholder)
        widget.config(foreground="grey")

        def on_focus_in(event):
            if widget.showing_placeholder:
                widget.delete(0, tk.END)
                widget.config(foreground=THEME_FG)
                widget.showing_placeholder = False

        def on_focus_out(event):
            if not widget.get().strip():
                widget.insert(0, placeholder)
                widget.config(foreground="grey")
                widget.showing_placeholder = True

        widget.bind("<FocusIn>", on_focus_in, add="+")
        widget.bind("<FocusOut>", on_focus_out, add="+")

    def get_text_value(self, widget):
        """Content of a placeholder-aware ScrolledText widget, empty while the placeholder shows."""
        if getattr(widget, "showing_placeholder", False):
            return ""
        return widget.get("1.0", tk.END).strip()

    def get_entry_value(self, widget):
        """Content of a placeholder-aware Entry widget, empty while the placeholder shows."""
        if getattr(widget, "showing_placeholder", False):
            return ""
        return widget.get().strip()

    def bind_text_counter(self, widget, label):
        """Keep `label` showing a live character/word count for `widget` (placeholder-aware)."""

        def update(event=None):
            content = self.get_text_value(widget)
            words = len(content.split()) if content else 0
            label.config(text=f"{len(content)} characters, {words} words")

        widget.bind("<KeyRelease>", update, add="+")
        widget.bind("<<Paste>>", lambda e: widget.after(1, update), add="+")
        widget.bind("<FocusIn>", lambda e: widget.after(1, update), add="+")
        widget.bind("<FocusOut>", lambda e: widget.after(1, update), add="+")
        update()

    # --- Actor picker: a native dropdown menu for choosing an existing actor (or
    # casting a new one) for a script line, posted at whatever widget was clicked.
    # No type-to-filter search box — a tk.Menu can't embed one — but arrow keys and
    # first-letter jumps (both native menu behavior) cover a cast this small. ---

    def open_actor_picker(self, anchor_widget, on_pick):
        """Post a dropdown menu of every actor below `anchor_widget`.

        Args:
            anchor_widget: The widget the menu positions itself under.
            on_pick: Called with (kind, key) once an actor is chosen — either an
                existing one, or a freshly cast one (via "+ New Actor", which reopens
                the New Actor dialog and calls this once training succeeds).
        """
        lookup = build_voice_lookup()
        actors = sorted({v for v in lookup.values()}, key=lambda kv: kv[1].lower())

        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="+ New Actor…",
            command=lambda: self.open_new_actor_dialog(on_created=on_pick),
        )
        menu.add_separator()
        for kind, key in actors:
            menu.add_command(
                label=key, command=lambda kind=kind, key=key: on_pick(kind, key)
            )

        x = anchor_widget.winfo_rootx()
        y = anchor_widget.winfo_rooty() + anchor_widget.winfo_height()
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    # --- Busy state: disable every control on the tab + show a spinner while a
    # background thread is running. The library gives no generation-progress hook
    # (see modeling_qwen3_tts.py — kwargs never reach the inner streamer-capable
    # call), so this is deliberately indeterminate rather than a fake percentage. ---

    def _set_frame_enabled(self, widget, enabled):
        """Recursively enable/disable every interactive control under `widget`."""
        for child in widget.winfo_children():
            cls = child.winfo_class()
            if cls == "TCombobox":
                child.config(state="readonly" if enabled else "disabled")
            elif cls in ("TButton", "TRadiobutton", "TEntry", "Text"):
                child.config(state=tk.NORMAL if enabled else tk.DISABLED)
            self._set_frame_enabled(child, enabled)

    def set_train_busy(self, busy):
        if busy:
            steps = ["Load model", "Build voice prompt", "Save voice"]
            if self.train_method.get() == "design":
                steps = ["Design reference voice", *steps]
            self.train_progress_panel.set_steps(steps)
            self.train_progress_panel.show(before=self.train_status_label)
        else:
            self.train_progress_panel.hide()

        self._set_frame_enabled(self.train_frame, not busy)
        self.train_button.config(text="Training..." if busy else "Train Voice")

    def set_use_busy(self, busy):
        # Steps are set by generate_speech() before this is called with busy=True —
        # they depend on the parsed [VoiceName] segments, unlike Train's fixed steps.
        if busy:
            self.generate_cancel_event.clear()
            self.use_progress_panel.show(before=self.use_status_label)
        else:
            self.use_progress_panel.hide()

        self._set_frame_enabled(self.use_frame, not busy)
        self.generate_button.config(text="Generating..." if busy else "Generate Speech")
        # Cancel needs the opposite enabled state to everything else _set_frame_enabled
        # just touched: available only while busy.
        self.cancel_button.config(state=tk.NORMAL if busy else tk.DISABLED)

    def cancel_generation(self):
        """Request cancellation. Generation only checks this between discrete steps
        (model load, per-voice-group generate, stitch, save) — a step already running
        is a single blocking model call with no way to interrupt it mid-flight, so
        cancelling during one takes effect once that step finishes, not instantly."""
        self.generate_cancel_event.set()
        self.cancel_button.config(state=tk.DISABLED)
        self.use_status_label.config(
            text="Cancelling — finishing the current step (can't be interrupted "
            "mid-step) before stopping...",
            foreground="orange",
        )

    def show_success_with_link(self, title, message, file_path):
        """Success dialog with a clickable link that reveals the generated file."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        ttk.Label(dialog, text=message, wraplength=400, justify=tk.LEFT).pack(
            padx=20, pady=(20, 10)
        )

        link = ttk.Label(
            dialog,
            text=file_path,
            foreground="blue",
            cursor="hand2",
            font=("TkDefaultFont", 10, "underline"),
            wraplength=400,
            justify=tk.LEFT,
        )
        link.pack(padx=20, pady=(0, 10))
        link.bind("<Button-1>", lambda e: reveal_in_file_manager(file_path))

        ttk.Button(dialog, text="OK", command=dialog.destroy).pack(pady=(0, 20))
        dialog.grab_set()

    def build_new_actor_dialog(self):
        """Builds the "Cast a New Actor" dialog once, hidden until
        `open_new_actor_dialog` deiconifies it."""
        self.new_actor_window = tk.Toplevel(self.root)
        self.new_actor_window.title("Cast a New Actor")
        self.new_actor_window.geometry("560x780")
        self.new_actor_window.withdraw()
        self.new_actor_window.transient(self.root)
        self.new_actor_window.protocol("WM_DELETE_WINDOW", self.close_new_actor_dialog)

        self.train_frame = ttk.Frame(self.new_actor_window)
        self.train_frame.pack(fill=tk.BOTH, expand=True)

        self.new_actor_title_label = ttk.Label(
            self.train_frame, text="Cast a New Actor", font=("Arial", 16, "bold")
        )
        self.new_actor_title_label.pack(pady=10)

        name_frame = ttk.Frame(self.train_frame)
        name_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(name_frame, text="Voice Name:").pack(side=tk.LEFT, padx=5)
        self.voice_name_entry = ttk.Entry(name_frame, width=30)
        self.voice_name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        # Recordings are filed under this name (local/{name}/mic_*.wav), so the
        # recording picker below needs to follow it as the user types.
        self.voice_name_entry.bind(
            "<KeyRelease>", lambda e: self.refresh_recording_list(), add="+"
        )

        method_frame = ttk.LabelFrame(
            self.train_frame, text="Training Method", padding=10
        )
        method_frame.pack(fill=tk.X, padx=20, pady=10)

        self.train_method = tk.StringVar(value="file")
        ttk.Radiobutton(
            method_frame,
            text="From Audio File + Transcript",
            variable=self.train_method,
            value="file",
            command=self.update_train_method_visibility,
        ).pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(
            method_frame,
            text="Record Audio (Pre-made Script)",
            variable=self.train_method,
            value="record",
            command=self.update_train_method_visibility,
        ).pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(
            method_frame,
            text="Design from Description",
            variable=self.train_method,
            value="design",
            command=self.update_train_method_visibility,
        ).pack(anchor=tk.W, pady=5)

        # File input section (shown only for the "file" method)
        self.file_frame = ttk.LabelFrame(
            self.train_frame, text="File Input", padding=10
        )

        audio_file_frame = ttk.Frame(self.file_frame)
        audio_file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(audio_file_frame, text="Audio File:").pack(side=tk.LEFT, padx=5)
        self.audio_file_entry = ttk.Entry(audio_file_frame, width=40)
        self.audio_file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(
            audio_file_frame, text="Browse", command=self.browse_audio_file
        ).pack(side=tk.LEFT, padx=5)

        transcript_frame = ttk.Frame(self.file_frame)
        transcript_frame.pack(fill=tk.X, pady=5)
        ttk.Label(transcript_frame, text="Transcript:").pack(anchor=tk.W, padx=5)
        self.transcript_entry = scrolledtext.ScrolledText(
            transcript_frame, height=4, width=50, wrap=tk.WORD
        )
        self.transcript_entry.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.add_placeholder_text(
            self.transcript_entry,
            "Enter the exact transcript of the reference audio...",
        )
        self.transcript_counter_label = ttk.Label(
            transcript_frame, text="", foreground="gray"
        )
        self.transcript_counter_label.pack(anchor=tk.E, padx=5)
        self.bind_text_counter(self.transcript_entry, self.transcript_counter_label)

        # Recording section (shown only for the "record" method)
        self.record_frame = ttk.LabelFrame(
            self.train_frame, text="Recording", padding=10
        )

        script_picker_frame = ttk.Frame(self.record_frame)
        script_picker_frame.pack(fill=tk.X, pady=5)
        ttk.Label(
            script_picker_frame, text="Script to read:", font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 5))
        self.script_combo = ttk.Combobox(
            script_picker_frame,
            width=20,
            state="readonly",
            values=list(TRAINING_SCRIPTS.keys()),
        )
        self.script_combo.set(next(iter(TRAINING_SCRIPTS)))
        self.script_combo.pack(side=tk.LEFT)
        self.script_combo.bind(
            "<<ComboboxSelected>>", lambda e: self.on_script_selected()
        )

        self.script_display = scrolledtext.ScrolledText(
            self.record_frame, height=7, width=50, wrap=tk.WORD
        )
        self.script_display.pack(fill=tk.X, pady=5)
        self.on_script_selected()

        self.record_status_label = ttk.Label(
            self.record_frame, text="Ready to record", foreground="green"
        )
        self.record_status_label.pack(pady=5)

        self.record_button = ttk.Button(
            self.record_frame, text="Start Recording", command=self.toggle_recording
        )
        self.record_button.pack(pady=5)

        # Every take is kept (local/{voice}/mic_{timestamp}.wav) rather than
        # overwriting a single fixed file — this picks which one Train Voice uses.
        take_frame = ttk.Frame(self.record_frame)
        take_frame.pack(fill=tk.X, pady=5)
        ttk.Label(take_frame, text="Take to use:").pack(side=tk.LEFT, padx=5)
        self.recording_combo = ttk.Combobox(take_frame, width=30, state="disabled")
        self.recording_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Design section (shown only for the "design" method): no reference audio at
        # all — a natural-language description synthesizes a voice from scratch via
        # the VoiceDesign model, which then gets clone-prompted like any other voice.
        self.design_frame = ttk.LabelFrame(
            self.train_frame, text="Voice Design", padding=10
        )
        ttk.Label(
            self.design_frame, text="Voice Description:", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=5)
        self.voice_design_instruct_entry = ttk.Entry(self.design_frame, width=50)
        self.voice_design_instruct_entry.pack(fill=tk.X, pady=5)
        self.add_placeholder_entry(
            self.voice_design_instruct_entry,
            "e.g. warm elderly British female voice",
        )

        self.train_status_label = ttk.Label(
            self.train_frame, text="", foreground="blue"
        )
        self.train_status_label.pack(pady=10)

        self.train_progress_panel = ProgressPanel(self.train_frame)
        # Not shown here — shown only while training is in progress (set_train_busy).

        self.x_vector_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.train_frame,
            text="Quick clone (skip transcript matching — faster, lower fidelity)",
            variable=self.x_vector_only_var,
        ).pack(pady=(0, 5))

        self.train_button = ttk.Button(
            self.train_frame,
            text="Train Voice",
            command=self.train_voice,
            style="Accent.TButton",
        )
        self.train_button.pack(pady=(20, 5))
        ttk.Button(
            self.train_frame, text="Close", command=self.close_new_actor_dialog
        ).pack(pady=(0, 15))

        self.update_train_method_visibility()

    def open_new_actor_dialog(self, on_created=None, prefill_name=None):
        """Show the New Actor dialog, reset to a blank "new actor" form — or, for
        `edit_voice`, into "edit" mode: name pre-filled and title updated to say
        so, `self._editing_voice_key` set for `train_voice` to check. Nothing else
        should carry over from whatever was last open in this dialog.

        Args:
            on_created: If given, called with (kind, key) once training succeeds —
                for the actor picker's "+ New Actor" path, so the freshly cast actor
                gets selected into whichever line asked for it.
            prefill_name: If given, pre-filled into Voice Name and treated as an
                edit of that actor (still a plain editable Entry — typing a
                different name renames it rather than editing in place).
        """
        self._new_actor_on_created = on_created
        self._editing_voice_key = prefill_name or None
        self.voice_name_entry.delete(0, tk.END)
        if prefill_name:
            self.voice_name_entry.insert(0, prefill_name)
        title = f'Edit "{prefill_name}"' if prefill_name else "Cast a New Actor"
        self.new_actor_window.title(title)
        self.new_actor_title_label.config(text=title)
        self.refresh_recording_list()
        self.new_actor_window.deiconify()
        self.new_actor_window.lift()
        self.new_actor_window.focus_set()
        # Modal: the dialog holds the one pending `_new_actor_on_created` callback,
        # so a second picker's "+ New Actor" (or the Cast sidebar's own button)
        # can't reach it and silently overwrite/drop it while this is open.
        self.new_actor_window.grab_set()

    def close_new_actor_dialog(self):
        self.new_actor_window.grab_release()
        self._new_actor_on_created = None
        self._editing_voice_key = None
        self.new_actor_window.withdraw()

    def _finish_new_actor_dialog(self, voice_name, on_created):
        """Close the dialog after a successful train, and — if it was opened via
        the actor picker's "+ New Actor" path rather than the Cast sidebar's own
        button — select the freshly cast actor into whichever line asked for it."""
        self.close_new_actor_dialog()
        if on_created is not None:
            on_created("custom", voice_name)

    def on_script_selected(self):
        """Refresh the read-only preview to match the picked training script."""
        text = TRAINING_SCRIPTS[self.script_combo.get()]
        self.script_display.config(state=tk.NORMAL)
        self.script_display.delete("1.0", tk.END)
        self.script_display.insert("1.0", text)
        self.script_display.config(state=tk.DISABLED)

    def update_train_method_visibility(self):
        """Show only the File Input, Recording, or Voice Design section, matching
        the selected training method."""
        method = self.train_method.get()
        frame_by_method = {
            "file": self.file_frame,
            "record": self.record_frame,
            "design": self.design_frame,
        }
        for other_method, frame in frame_by_method.items():
            if other_method != method:
                frame.pack_forget()
        frame_by_method[method].pack(
            fill=tk.X, padx=20, pady=10, before=self.train_status_label
        )

    def refresh_recording_list(self):
        """Repopulate the "Take to use" picker for the current Voice Name. Called
        whenever that name changes, and after every new recording (which always
        becomes the selection)."""
        voice_name = self.voice_name_entry.get().strip()
        self.recording_map = {
            os.path.basename(p): p for p in list_voice_recordings(voice_name)
        }
        values = list(self.recording_map.keys())
        self.recording_combo["values"] = values
        if values:
            self.recording_combo.set(values[-1])
            self.recording_combo.config(state="readonly")
        else:
            self.recording_combo.set("")
            self.recording_combo.config(state="disabled")

    def refresh_voice_lists(self):
        """Repopulate the Cast sidebar — called after a voice is (re-)trained or
        deleted, so it shows up immediately without restarting the app."""
        self.refresh_cast_sidebar()

    def build_generation_bar(self, parent):
        """Bottom bar under the script canvas: Language, Output Format, Advanced
        sampling controls, Generate/Cancel, and the progress panel."""
        bar = ttk.Frame(parent)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(bar, orient=tk.HORIZONTAL).pack(fill=tk.X)

        controls_row = ttk.Frame(bar)
        controls_row.pack(fill=tk.X, padx=20, pady=(12, 4))

        ttk.Label(controls_row, text="Language:").pack(side=tk.LEFT, padx=(0, 5))
        self.use_language_combo = ttk.Combobox(
            controls_row, width=12, state="readonly", values=SUPPORTED_LANGUAGES
        )
        self.use_language_combo.set("Auto")
        self.use_language_combo.pack(side=tk.LEFT, padx=(0, 18))

        ttk.Label(controls_row, text="Format:").pack(side=tk.LEFT, padx=(0, 5))
        self.output_format = tk.StringVar(value="wav")
        for fmt in ("wav", "mp3", "m4a"):
            ttk.Radiobutton(
                controls_row, text=fmt.upper(), variable=self.output_format, value=fmt
            ).pack(side=tk.LEFT, padx=4)

        # Generate opens a native Save As dialog (pre-filled with a dated default
        # filename) before starting, rather than writing to a persistent folder path.
        self.last_save_dir = OUTPUT_DIR
        self.cancel_button = ttk.Button(
            controls_row,
            text="Cancel",
            command=self.cancel_generation,
            state=tk.DISABLED,
        )
        self.cancel_button.pack(side=tk.RIGHT)
        self.generate_button = ttk.Button(
            controls_row,
            text="Generate Speech",
            command=self.generate_speech,
            style="Accent.TButton",
        )
        self.generate_button.pack(side=tk.RIGHT, padx=(0, 10))

        self._setup_advanced_generation_section(bar)

        self.use_status_label = ttk.Label(bar, text="", foreground="blue")
        self.use_status_label.pack(padx=20, pady=(0, 4), anchor=tk.W)

        self.use_progress_panel = ProgressPanel(bar)
        # Not shown here — shown only while generation is in progress (set_use_busy).

    def _setup_advanced_generation_section(self, parent):
        """Collapsed by default; values default to the library's own hard defaults
        (qwen_tts's `_merge_generate_kwargs`), so leaving this closed generates
        identically to before it existed. subtalker_* variants are confirmed relevant
        for every model this app uses — see TODO.md's "Generation / sampling controls"
        entry."""
        # Defaults mirror qwen_tts.inference.qwen3_tts_model._merge_generate_kwargs's
        # hard_defaults exactly.
        self.adv_temperature = tk.DoubleVar(value=0.9)
        self.adv_top_k = tk.IntVar(value=50)
        self.adv_top_p = tk.DoubleVar(value=1.0)
        self.adv_repetition_penalty = tk.DoubleVar(value=1.05)
        self.adv_max_new_tokens = tk.IntVar(value=2048)
        self.adv_subtalker_dosample = tk.BooleanVar(value=True)
        self.adv_subtalker_top_k = tk.IntVar(value=50)
        self.adv_subtalker_top_p = tk.DoubleVar(value=1.0)
        self.adv_subtalker_temperature = tk.DoubleVar(value=0.9)

        section = CollapsibleSection(parent, "Advanced (sampling controls)")
        section.pack(fill=tk.X, padx=20, pady=(0, 10))
        body = section.body

        self._add_spinbox_row(
            body, "Temperature:", self.adv_temperature, 0.1, 2.0, 0.05
        )
        self._add_spinbox_row(body, "Top-k:", self.adv_top_k, 0, 200, 1)
        self._add_spinbox_row(body, "Top-p:", self.adv_top_p, 0.0, 1.0, 0.05)
        self._add_spinbox_row(
            body, "Repetition penalty:", self.adv_repetition_penalty, 1.0, 2.0, 0.05
        )
        self._add_spinbox_row(
            body, "Max new tokens:", self.adv_max_new_tokens, 256, 8192, 256
        )

        ttk.Separator(body, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Label(
            body, text="Sub-talker (12Hz tokenizer):", font=("Arial", 9, "bold")
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            body,
            text="Sub-talker sampling enabled",
            variable=self.adv_subtalker_dosample,
        ).pack(anchor=tk.W, pady=2)
        self._add_spinbox_row(
            body, "Sub-talker top-k:", self.adv_subtalker_top_k, 0, 200, 1
        )
        self._add_spinbox_row(
            body, "Sub-talker top-p:", self.adv_subtalker_top_p, 0.0, 1.0, 0.05
        )
        self._add_spinbox_row(
            body,
            "Sub-talker temperature:",
            self.adv_subtalker_temperature,
            0.1,
            2.0,
            0.05,
        )

    def _add_spinbox_row(self, parent, label, var, from_, to, increment):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=22, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Spinbox(
            row, textvariable=var, from_=from_, to=to, increment=increment, width=10
        ).pack(side=tk.LEFT)

    def advanced_generation_kwargs(self):
        """Current Advanced-section values, forwarded as **kwargs to
        generate_custom_voice/generate_voice_clone (both pass unrecognized kwargs
        through to the model's own generate())."""
        return {
            "temperature": self.adv_temperature.get(),
            "top_k": self.adv_top_k.get(),
            "top_p": self.adv_top_p.get(),
            "repetition_penalty": self.adv_repetition_penalty.get(),
            "max_new_tokens": self.adv_max_new_tokens.get(),
            "subtalker_dosample": self.adv_subtalker_dosample.get(),
            "subtalker_top_k": self.adv_subtalker_top_k.get(),
            "subtalker_top_p": self.adv_subtalker_top_p.get(),
            "subtalker_temperature": self.adv_subtalker_temperature.get(),
        }

    def build_settings_dialog(self):
        """Builds the Settings dialog once, hidden until `open_settings_dialog`
        deiconifies it — opened from a gear icon rather than a tab. Settings you'd
        set once for this machine, rather than per-run choices: a Global section
        (just Device — a fact about the hardware, not a per-flow tradeoff), then
        Train and Generate sections for settings where the two flows might reasonably
        want different tradeoffs (a bigger/slower model for a voice you're keeping
        forever, a smaller/faster one for quick generation previews)."""
        self.settings_window = tk.Toplevel(self.root)
        self.settings_window.title("Settings")
        self.settings_window.geometry("480x560")
        self.settings_window.withdraw()
        self.settings_window.transient(self.root)
        self.settings_window.protocol("WM_DELETE_WINDOW", self.close_settings_dialog)

        self.configure_frame = ttk.Frame(self.settings_window)
        self.configure_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(
            self.configure_frame, text="Configuration", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        global_frame = ttk.LabelFrame(self.configure_frame, text="Global", padding=10)
        global_frame.pack(fill=tk.X, padx=20, pady=10)

        device_frame = ttk.Frame(global_frame)
        device_frame.pack(fill=tk.X)
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT, padx=(0, 10))
        cuda_radio = ttk.Radiobutton(
            device_frame, text="CUDA (GPU)", variable=self.device_type, value="cuda"
        )
        cuda_radio.pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(
            device_frame, text="CPU", variable=self.device_type, value="cpu"
        ).pack(side=tk.LEFT, padx=10)

        if not torch.cuda.is_available():
            # Disabled rather than left selectable-but-warned: picking CUDA when
            # it's not there just re-triggers the same "falling back to CPU"
            # dialog on every model load.
            cuda_radio.config(state=tk.DISABLED)
            self.device_type.set("cpu")
            ttk.Label(
                device_frame,
                text="(CUDA not available on this machine)",
                foreground="gray",
            ).pack(side=tk.LEFT, padx=10)

        # (dtype_frame, dtype_var) pairs — CPU always forces float32 in
        # get_model_config regardless of the dtype setting, so when Device is CPU,
        # the dtype choice is disabled (not just documented) rather than left
        # selectable-but-inert.
        self._dtype_controls = []
        self._setup_model_settings_section(
            self.configure_frame, "Train", self.train_model_size, self.train_dtype
        )
        self._setup_model_settings_section(
            self.configure_frame,
            "Generate",
            self.generate_model_size,
            self.generate_dtype,
        )
        self._update_dtype_availability()
        self.device_type.trace_add(
            "write", lambda *_args: self._update_dtype_availability()
        )

        ttk.Button(
            self.configure_frame, text="Done", command=self.close_settings_dialog
        ).pack(pady=20)

    def open_settings_dialog(self):
        self.settings_window.deiconify()
        self.settings_window.lift()
        self.settings_window.focus_set()
        self.settings_window.grab_set()

    def close_settings_dialog(self):
        self.settings_window.grab_release()
        self.settings_window.withdraw()

    def _setup_model_settings_section(self, parent, title, model_size_var, dtype_var):
        """Model size + dtype radios, shared layout for the Train/Generate sections."""
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill=tk.X, padx=20, pady=10)

        size_frame = ttk.Frame(frame)
        size_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(size_frame, text="Model size:").pack(side=tk.LEFT, padx=(0, 10))
        for size in MODEL_SIZES:
            ttk.Radiobutton(
                size_frame, text=size, variable=model_size_var, value=size
            ).pack(side=tk.LEFT, padx=10)

        dtype_frame = ttk.Frame(frame)
        dtype_frame.pack(fill=tk.X)
        ttk.Label(dtype_frame, text="dtype:").pack(side=tk.LEFT, padx=(0, 10))
        for dtype_name in DTYPE_OPTIONS:
            ttk.Radiobutton(
                dtype_frame, text=dtype_name, variable=dtype_var, value=dtype_name
            ).pack(side=tk.LEFT, padx=10)
        ttk.Label(
            frame,
            text="dtype only applies when running on CUDA — CPU always uses float32.",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(5, 0))
        self._dtype_controls.append((dtype_frame, dtype_var))

    def _update_dtype_availability(self):
        """Enable dtype radios only on CUDA; force float32 on CPU so the displayed
        value always matches what get_model_config will actually use."""
        cuda_selected = self.device_type.get() == "cuda"
        for dtype_frame, dtype_var in self._dtype_controls:
            self._set_frame_enabled(dtype_frame, cuda_selected)
            if not cuda_selected:
                dtype_var.set("float32")

    # --- Top bar: wordmark, script import, settings. ---

    def build_top_bar(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar, text="STUDIO", font=("Arial", 14, "bold")).pack(
            side=tk.LEFT, padx=(15, 20), pady=10
        )
        ttk.Button(bar, text="⚙ Settings", command=self.open_settings_dialog).pack(
            side=tk.RIGHT, padx=(0, 15), pady=10
        )
        ttk.Button(bar, text="Import Script", command=self.open_import_dialog).pack(
            side=tk.RIGHT, padx=(0, 8), pady=10
        )

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X)

    # --- Cast sidebar: every available actor (preset, cloned, or designed voice),
    # with a play-to-preview button and a way to cast a new one. ---

    def build_cast_sidebar(self, parent):
        sidebar = ttk.Frame(parent, width=230)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        self.cast_header_label = ttk.Label(
            sidebar, text="CAST", font=("Arial", 10, "bold")
        )
        self.cast_header_label.pack(anchor=tk.W, padx=14, pady=(14, 8))

        ttk.Button(
            sidebar,
            text="+ New Actor",
            style="Accent.TButton",
            command=lambda: self.open_new_actor_dialog(),
        ).pack(fill=tk.X, padx=12, pady=(0, 10))

        self.cast_scroll = ScrollableFrame(sidebar)
        self.cast_scroll.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 10))

    def _tick_preview_spinner(self):
        """Advance every pending Cast sidebar button to the next spinner frame,
        forever — a no-op whenever nothing is pending. Simpler than starting/
        stopping the animation loop to match, and cheap when idle."""
        self._preview_spinner_frame = (self._preview_spinner_frame + 1) % len(
            self.icon_spinner_frames
        )
        frame = self.icon_spinner_frames[self._preview_spinner_frame]
        for button in self._preview_spinner_buttons:
            if button.winfo_exists():
                button.config(image=frame)
        self.root.after(150, self._tick_preview_spinner)

    def refresh_cast_sidebar(self):
        """Rebuild the Cast list from `build_voice_lookup()` — called on startup and
        whenever a voice is (re-)trained or deleted."""
        for child in self.cast_scroll.body.winfo_children():
            child.destroy()
        self._preview_spinner_buttons = []

        lookup = build_voice_lookup()
        actors = sorted({v for v in lookup.values()}, key=lambda kv: kv[1].lower())
        self.cast_header_label.config(text=f"CAST · {len(actors)}")

        for kind, key in actors:
            color = color_for_actor(key)
            row = ttk.Frame(self.cast_scroll.body)
            row.pack(fill=tk.X, pady=3, padx=4)

            ttk.Label(row, text="●", foreground=color).pack(side=tk.LEFT, padx=(4, 8))

            info = ttk.Frame(row)
            info.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(info, text=key, font=("Arial", 10, "bold")).pack(anchor=tk.W)
            badge = "PRESET" if kind == "preset" else "CLONED"
            ttk.Label(info, text=badge, foreground=color, font=("Arial", 8)).pack(
                anchor=tk.W
            )

            # Play (or its spinner) is packed first so it always claims the outer-
            # right slot — the same column on every row, preset or custom — with
            # edit/delete (custom actors only) stacking to its left, since side=
            # RIGHT packs each new widget to the left of the ones already there.
            # Every icon is the same fixed-size bitmap (see ICON_SIZE), so unlike
            # font glyphs, cycling the spinner can't change the button's footprint.
            ready = os.path.exists(preview_path_for(key))
            play_button = ttk.Button(
                row,
                image=self.icon_play if ready else self.icon_spinner_frames[0],
                style="Icon.TButton",
                command=lambda k=key: self.play_preview(k),
            )
            play_button.pack(side=tk.RIGHT, padx=(2, 4))
            if not ready:
                play_button.config(state=tk.DISABLED)
                self._preview_spinner_buttons.append(play_button)

            # Presets aren't files on disk — nothing to edit or delete.
            if kind == "custom":
                ttk.Button(
                    row,
                    image=self.icon_delete,
                    style="Icon.TButton",
                    command=lambda k=key: self.delete_voice(k),
                ).pack(side=tk.RIGHT, padx=2)
                ttk.Button(
                    row,
                    image=self.icon_edit,
                    style="Icon.TButton",
                    command=lambda k=key: self.edit_voice(k),
                ).pack(side=tk.RIGHT, padx=2)

            self.cast_scroll.bind_scroll(row)

    def play_preview(self, key):
        """Play `key`'s cached preview clip. The button is disabled (showing a
        spinner instead of ▶) until the preview exists, so this is only ever
        reachable once it does — bar a rare race with a just-deleted voice."""
        preview_path = preview_path_for(key)
        if not os.path.exists(preview_path):
            return
        try:
            data, sr = sf.read(preview_path, dtype="float32")
            sd.play(data, sr)
        except Exception as e:
            messagebox.showerror("Playback Error", f"Couldn't play preview: {e}")

    def edit_voice(self, key):
        """Open the New Actor dialog in "edit" mode for an existing cloned/designed
        actor: name pre-filled, retraining overwrites `key` in place. Changing the
        name before retraining renames it instead — see `train_voice`."""
        self.open_new_actor_dialog(prefill_name=key)

    def _delete_voice_files(self, key):
        """Remove a custom voice's saved file, cached preview, and any recordings
        filed under its name — the on-disk side of both `delete_voice` and a
        rename in `edit_voice` (train under the new name, then drop the old)."""
        voice_file = os.path.join(OUTPUT_DIR, f"{key}.pt")
        if os.path.exists(voice_file):
            os.remove(voice_file)
        preview_file = preview_path_for(key)
        if os.path.exists(preview_file):
            os.remove(preview_file)
        recordings_dir = os.path.join(OUTPUT_DIR, key)
        if os.path.isdir(recordings_dir):
            shutil.rmtree(recordings_dir)

    def delete_voice(self, key):
        """Permanently remove a cloned/designed actor: its saved voice file, cached
        preview, and any recordings filed under its name."""
        if not messagebox.askyesno(
            "Delete Actor?",
            f'Delete "{key}"? This permanently removes its saved voice and '
            "preview — this can't be undone.",
        ):
            return
        self._delete_voice_files(key)
        self.refresh_voice_lists()

    # --- Script canvas: the canvas *is* the script — a stack of dialogue blocks
    # (colored dot + actor name + optional tone + the line itself), built and edited
    # directly, rather than a text box with bracket-marker syntax. ---

    def build_script_canvas(self, parent):
        self.script_scroll = ScrollableFrame(parent)
        self.script_scroll.pack(fill=tk.BOTH, expand=True, padx=20, pady=(15, 10))
        self.render_script_lines()

    def render_script_lines(self):
        """(Re)build every dialogue block from `self.script_lines`, plus the
        trailing "+ Add Line" control. Simplest correct approach given how rarely
        this needs to run (an edit, an add, a remove, or an import) — no incremental
        diffing."""
        for child in self.script_scroll.body.winfo_children():
            child.destroy()

        for index, line in enumerate(self.script_lines):
            self._build_script_block(self.script_scroll.body, index, line)

        self.add_line_button = ttk.Button(
            self.script_scroll.body,
            text="+ Add Line",
            command=lambda: self.open_actor_picker(
                self.add_line_button, self._append_script_line
            ),
        )
        self.add_line_button.pack(fill=tk.X, pady=8)
        self.script_scroll.bind_scroll(self.add_line_button)

    def _build_script_block(self, parent, index, line):
        color = color_for_actor(line["key"])
        block = ttk.Frame(parent, style="Block.TFrame", padding=(12, 10))
        block.pack(fill=tk.X, pady=(0, 16), anchor=tk.W)

        header = ttk.Frame(block, style="Block.TFrame")
        header.pack(fill=tk.X, anchor=tk.W)

        ttk.Label(header, text="●", foreground=color, style="Block.TLabel").pack(
            side=tk.LEFT, padx=(0, 6)
        )
        name_label = ttk.Label(
            header,
            text=line["key"].upper(),
            foreground=color,
            font=(SCRIPT_FONT_FAMILY, 10, "bold"),
            cursor="hand2",
            style="Block.TLabel",
        )
        name_label.pack(side=tk.LEFT)
        name_label.bind(
            "<ButtonRelease-1>",
            lambda e, i=index, w=name_label: self.open_actor_picker(
                w, lambda kind, key, i=i: self._set_line_actor(i, kind, key)
            ),
        )

        # Tone has no meaning for a voice-clone line (generate_voice_clone has no
        # instruct concept) — offered only for preset lines, rather than shown then
        # silently ignored.
        if line["kind"] == "preset":
            tone_entry = ttk.Entry(
                header,
                width=22,
                font=(SCRIPT_FONT_FAMILY, 9, "italic"),
                style="Block.TEntry",
            )
            tone_entry.pack(side=tk.LEFT, padx=(10, 0))
            if line.get("tone"):
                tone_entry.insert(0, line["tone"])
            else:
                self.add_placeholder_entry(tone_entry, "add a tone…")
            tone_entry.bind(
                "<KeyRelease>",
                lambda e, i=index, w=tone_entry: self._update_line_tone(i, w),
                add="+",
            )

        ttk.Button(
            header,
            text="✕",
            width=2,
            command=lambda i=index: self._remove_script_line(i),
        ).pack(side=tk.RIGHT)

        text_widget = tk.Text(
            block,
            height=1,
            wrap=tk.WORD,
            background=SCRIPT_BLOCK_BG,
            foreground=THEME_FG,
            insertbackground=THEME_FG,
            borderwidth=1,
            relief=tk.SOLID,
            highlightthickness=1,
            highlightbackground=SCRIPT_BLOCK_BORDER,
            highlightcolor=SCRIPT_BLOCK_FOCUS_BORDER,
            font=(SCRIPT_FONT_FAMILY, 11),
            padx=6,
            pady=4,
        )
        text_widget.insert("1.0", line["text"])
        text_widget.pack(fill=tk.X, padx=(15, 0), pady=(6, 0))
        text_widget.bind(
            "<KeyRelease>",
            lambda e, i=index, w=text_widget: self._on_line_text_changed(i, w),
            add="+",
        )
        self._autosize_text(text_widget)
        self.script_scroll.bind_scroll(block)

    def _autosize_text(self, widget):
        """Grow/shrink a dialogue Text widget to fit its content — Tkinter Text
        boxes don't do this on their own."""
        widget.update_idletasks()
        try:
            num_lines = int(widget.count("1.0", "end", "displaylines")[0])
        except (TypeError, IndexError):
            num_lines = 1
        widget.config(height=max(1, num_lines))

    def _on_line_text_changed(self, index, widget):
        if 0 <= index < len(self.script_lines):
            self.script_lines[index]["text"] = widget.get("1.0", "end-1c")
        self._autosize_text(widget)

    def _update_line_tone(self, index, widget):
        if 0 <= index < len(self.script_lines):
            self.script_lines[index]["tone"] = self.get_entry_value(widget)

    def _set_line_actor(self, index, kind, key):
        if 0 <= index < len(self.script_lines):
            self.script_lines[index]["kind"] = kind
            self.script_lines[index]["key"] = key
            if kind == "custom":
                self.script_lines[index]["tone"] = ""
            self.render_script_lines()

    def _append_script_line(self, kind, key):
        self.script_lines.append({"kind": kind, "key": key, "tone": "", "text": ""})
        self.render_script_lines()

    def _remove_script_line(self, index):
        if 0 <= index < len(self.script_lines):
            del self.script_lines[index]
            self.render_script_lines()

    # --- Script import: paste a plain-text script (screenplay format — an LLM
    # writes scripts in this shape naturally) and drop it straight into
    # `self.script_lines`. ---

    def build_import_dialog(self):
        self.import_window = tk.Toplevel(self.root)
        self.import_window.title("Import a Script")
        self.import_window.geometry("640x600")
        self.import_window.withdraw()
        self.import_window.transient(self.root)
        self.import_window.protocol("WM_DELETE_WINDOW", self.close_import_dialog)

        frame = ttk.Frame(self.import_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        ttk.Label(frame, text="Import a Script", font=("Arial", 16, "bold")).pack(
            anchor=tk.W
        )
        ttk.Label(
            frame,
            text="Paste plain text — written by hand, or generated by an LLM. "
            "Format: NAME on its own line, optional (tone) right after it, "
            "dialogue below, blank line between actors.",
            foreground="gray",
            wraplength=580,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 10))

        self.import_text = scrolledtext.ScrolledText(frame, height=16, wrap=tk.WORD)
        self.import_text.pack(fill=tk.BOTH, expand=True)

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(button_row, text="Cancel", command=self.close_import_dialog).pack(
            side=tk.RIGHT
        )
        ttk.Button(
            button_row,
            text="Import Script",
            style="Accent.TButton",
            command=self._commit_import,
        ).pack(side=tk.RIGHT, padx=(0, 10))

    def open_import_dialog(self):
        self.import_text.delete("1.0", tk.END)
        self.import_window.deiconify()
        self.import_window.lift()
        self.import_window.focus_set()
        self.import_window.grab_set()

    def close_import_dialog(self):
        self.import_window.grab_release()
        self.import_window.withdraw()

    def _commit_import(self):
        text = self.import_text.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showerror("Error", "Paste a script first.")
            return

        lines, errors = parse_screenplay_text(text, build_voice_lookup())
        if errors:
            messagebox.showerror(
                "Unrecognized Actors",
                "Cast these actors first, or fix the spelling, then import again:\n\n"
                + "\n".join(errors),
            )
            return
        if not lines:
            messagebox.showerror("Error", "Couldn't find any dialogue to import.")
            return

        self.script_lines = lines
        self.render_script_lines()
        self.close_import_dialog()

    # --- Proactive voice previews: generated right after a voice is cast, and via a
    # startup sweep that backfills anything missing (including the 9 presets, on
    # first-ever run) — so the Cast sidebar's play button is always instant rather
    # than triggering a model load. Cached as .wav, not .m4a: soundfile/libsndfile
    # (used for playback) can't decode AAC/M4A, only encode_audio's PyAV path can. ---

    def generate_voice_preview(
        self, kind, key, model=None, device_type=None, dtype=None
    ):
        """Synthesize and cache a short preview clip for one (kind, key) voice.

        Args:
            kind: "preset" or "custom".
            key: Voice/speaker name.
            model, device_type, dtype: Reuse an already-loaded model (e.g. right
                after casting a voice) instead of loading a fresh one.
        """
        device_type = device_type or self.device_type.get()
        dtype = dtype or DTYPE_MAP[self.generate_dtype.get()]
        model_size = self.generate_model_size.get()

        if model is None:
            model_repo = (
                MODEL_REPOS_CUSTOM_VOICE[model_size]
                if kind == "preset"
                else MODEL_REPOS_BASE[model_size]
            )
            model, device_type = self._load_model(
                model_repo, device_type, dtype, show_warning=False
            )

        if kind == "preset":
            wavs, sr = model.generate_custom_voice(
                text=PREVIEW_TEXT, speaker=key, language="Auto"
            )
        else:
            prompt_item = self._load_voice_clone_prompt(key, device_type)
            wavs, sr = model.generate_voice_clone(
                text=PREVIEW_TEXT, language="Auto", voice_clone_prompt=[prompt_item]
            )

        sf.write(preview_path_for(key), wavs[0], sr)

    def sweep_missing_previews(self):
        """Backfill any voice missing a cached preview — run once, on a background
        thread, right after the window is constructed. Batches the missing voices by
        kind so each of Base/CustomVoice loads at most once, same as generation."""
        try:
            lookup = build_voice_lookup()
            actors = {v for v in lookup.values()}
            existing = {
                f.removesuffix("_preview.wav")
                for f in os.listdir(PREVIEW_DIR)
                if f.endswith("_preview.wav")
            }
            missing = [(kind, key) for kind, key in actors if key not in existing]
            if not missing:
                return

            device_type = self.device_type.get()
            dtype = DTYPE_MAP[self.generate_dtype.get()]
            model_size = self.generate_model_size.get()
            groups = {}
            for kind, key in missing:
                groups.setdefault(kind, []).append(key)

            for kind, keys in groups.items():
                model_repo = (
                    MODEL_REPOS_CUSTOM_VOICE[model_size]
                    if kind == "preset"
                    else MODEL_REPOS_BASE[model_size]
                )
                model, device_type = self._load_model(
                    model_repo, device_type, dtype, show_warning=False
                )
                if kind == "preset":
                    wavs, sr = model.generate_custom_voice(
                        text=[PREVIEW_TEXT] * len(keys), speaker=keys, language="Auto"
                    )
                else:
                    prompts = [
                        self._load_voice_clone_prompt(key, device_type) for key in keys
                    ]
                    wavs, sr = model.generate_voice_clone(
                        text=[PREVIEW_TEXT] * len(keys),
                        language="Auto",
                        voice_clone_prompt=prompts,
                    )
                for key, wav in zip(keys, wavs):
                    sf.write(preview_path_for(key), wav, sr)
                # Refresh after each kind group completes (rather than only once at
                # the very end) so preset and cloned buttons flip from spinner to ▶
                # as soon as their own group is actually done, not the slower of
                # the two.
                self.root.after(0, self.refresh_cast_sidebar)
        except Exception:
            # Best-effort backfill — a failure here just means some preview buttons
            # stay in the "not ready yet" state until the next launch retries.
            pass

    def browse_audio_file(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if filename:
            self.audio_file_entry.delete(0, tk.END)
            self.audio_file_entry.insert(0, filename)

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.voice_name_entry.get().strip():
            messagebox.showerror("Error", "Please enter a voice name before recording")
            return

        self.recording = True
        self.audio_chunks = []
        self.record_status_label.config(
            text="Recording... Press Stop to finish", foreground="red"
        )
        self.record_button.config(text="Stop Recording")

        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_chunks.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback
        )
        self.stream.start()

    def stop_recording(self):
        self.recording = False
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()

        if self.audio_chunks:
            voice_name = self.voice_name_entry.get().strip()
            voice_dir = os.path.join(OUTPUT_DIR, voice_name)
            os.makedirs(voice_dir, exist_ok=True)

            audio_data = np.concatenate(self.audio_chunks, axis=0)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            recording_path = os.path.join(voice_dir, f"mic_{timestamp}.wav")
            sf.write(recording_path, audio_data, SAMPLE_RATE)
            self.refresh_recording_list()  # new take always becomes the selection
            self.record_status_label.config(
                text=f"Recording saved to {recording_path}", foreground="green"
            )
        else:
            self.record_status_label.config(
                text="No audio recorded", foreground="orange"
            )

        self.record_button.config(text="Start Recording")

    def get_model_config(self, device_type, dtype, show_warning=True):
        """Build `Qwen3TTSModel.from_pretrained` kwargs for the given device.

        Falls back to CPU if `device_type` is "cuda" but CUDA isn't actually
        available. `dtype` is only honored on CUDA — CPU always forces float32,
        since float16/bfloat16 support on CPU-only torch builds is inconsistent.

        Args:
            device_type: "cuda" or "cpu".
            dtype: A torch dtype (from Configure's dtype setting), used when
                `device_type` resolves to "cuda".
            show_warning: Whether to show a dialog when falling back from cuda to cpu.

        Returns:
            A dict of device_map/dtype/attn_implementation kwargs.
        """
        if device_type == "cuda" and not torch.cuda.is_available():
            cuda_info = []
            cuda_info.append(f"PyTorch version: {torch.__version__}")
            cuda_info.append(f"CUDA available: {torch.cuda.is_available()}")
            if hasattr(torch.version, "cuda"):
                cuda_info.append(f"PyTorch CUDA version: {torch.version.cuda}")
            if hasattr(sys, "_MEIPASS"):
                cuda_info.append("Running as bundled executable")
                torch_lib = os.path.join(sys._MEIPASS, "torch", "lib")
                cuda_info.append(f"Torch lib path exists: {os.path.exists(torch_lib)}")
            else:
                cuda_info.append("Running as Python script")

            if show_warning:
                info_msg = (
                    "CUDA is not available. Falling back to CPU mode.\n\n"
                    + "\n".join(cuda_info)
                )
                self.root.after(
                    0, lambda: messagebox.showwarning("CUDA Not Available", info_msg)
                )
            device_type = "cpu"

        if device_type == "cuda" and torch.cuda.is_available():
            # flash_attn is an optional dependency; sdpa works everywhere torch does.
            # A full import (not just importlib.util.find_spec) so a present-but-broken
            # install (e.g. compiled against a mismatched CUDA/torch) falls back too.
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"

            return {
                "device_map": "auto",
                "dtype": dtype,
                "attn_implementation": attn_impl,
            }
        return {
            "device_map": "cpu",
            "dtype": torch.float32,
            "attn_implementation": "eager",
        }

    def _load_model(
        self, model_repo, device_type, dtype, show_warning=True, status_label=None
    ):
        """Load `model_repo`, falling back to CPU once on a CUDA compat error.

        Args:
            model_repo: HuggingFace repo id to load.
            device_type: "cuda" or "cpu".
            dtype: A torch dtype, passed through to `get_model_config`.
            show_warning: Passed through to `get_model_config`.
            status_label: If given, updated with a fallback message on retry.

        Returns:
            A (model, effective_device_type) tuple — `effective_device_type` is
            "cpu" if a CUDA compatibility error forced a fallback.
        """
        config = self.get_model_config(device_type, dtype, show_warning=show_warning)
        try:
            return Qwen3TTSModel.from_pretrained(model_repo, **config), device_type
        except (RuntimeError, torch.cuda.CudaError) as cuda_error:
            if not _is_cuda_compat_error(cuda_error):
                raise
            # Python clears the `as` binding once this except block exits, so capture
            # the message now — the messagebox lambda below runs later, via root.after.
            error_text = str(cuda_error)
            if status_label is not None:
                self.root.after(
                    0,
                    lambda: status_label.config(
                        text="CUDA error detected, falling back to CPU...",
                        foreground="orange",
                    ),
                )
            self.root.after(
                0,
                lambda: messagebox.showwarning(
                    "CUDA Compatibility Issue",
                    f"CUDA error detected: {error_text}\n\nFalling back to CPU mode. This may be slower.",
                ),
            )
            config = self.get_model_config("cpu", dtype, show_warning=False)
            return Qwen3TTSModel.from_pretrained(model_repo, **config), "cpu"

    def train_voice(self):
        """Validate the Train Voice form and kick off `_train_voice_thread`.

        In "new actor" mode (`self._editing_voice_key` is None), any name already
        in use — preset or custom — is rejected outright; there's no "New Actor"
        reason to collide with one. In "edit" mode, retraining under the actor's
        own name needs no extra check (that's the point of Edit); typing a
        *different*, free name renames it (passed through as `renaming_from` for
        `_train_voice_thread` to drop the old files under once the new ones save).
        """
        voice_name = self.voice_name_entry.get().strip()
        if not voice_name:
            messagebox.showerror("Error", "Please enter a voice name")
            return

        editing_key = self._editing_voice_key
        renaming_from = None
        is_same_voice = (
            editing_key is not None and voice_name.lower() == editing_key.lower()
        )
        if not is_same_voice:
            existing = build_voice_lookup().get(voice_name.lower())
            if existing is not None:
                _, existing_key = existing
                messagebox.showerror(
                    "Name Already Used",
                    f"A voice named '{existing_key}' already exists — pick a "
                    "different name, or use that actor's Edit button instead.",
                )
                return
            renaming_from = editing_key

        method = self.train_method.get()
        device_type = self.device_type.get()
        x_vector_only = self.x_vector_only_var.get()

        recording_path = None
        script_text = None
        design_instruct = None
        if method == "record":
            recording_path = self.recording_map.get(self.recording_combo.get())
            script_text = TRAINING_SCRIPTS[self.script_combo.get()]
        elif method == "design":
            design_instruct = self.get_entry_value(self.voice_design_instruct_entry)
            if not design_instruct:
                messagebox.showerror("Error", "Please enter a voice description")
                return
            script_text = VOICE_DESIGN_REFERENCE_TEXT

        self.set_train_busy(True)

        thread = threading.Thread(
            target=self._train_voice_thread,
            args=(
                voice_name,
                method,
                device_type,
                recording_path,
                script_text,
                x_vector_only,
                design_instruct,
                renaming_from,
            ),
        )
        thread.daemon = True
        thread.start()

    def _train_voice_thread(
        self,
        voice_name,
        method,
        device_type,
        recording_path=None,
        script_text=None,
        x_vector_only=False,
        design_instruct=None,
        renaming_from=None,
    ):
        """Build and save a VoiceClonePromptItem for `voice_name` (background thread).

        Args:
            voice_name: Name to save the trained voice under (OUTPUT_DIR/{voice_name}.pt).
            method: "file" (audio_file_entry + transcript_entry), "record"
                (recording_path + script_text), or "design" (design_instruct +
                script_text — synthesizes a reference clip with VoiceDesign first,
                then clone-prompts it like any other reference recording).
            device_type: "cuda" or "cpu".
            recording_path: Recorded take to use, when method == "record".
            script_text: The training script read aloud (method == "record") or
                synthesized by VoiceDesign (method == "design") — either way, the
                exact text the resulting reference audio actually says.
            x_vector_only: "Quick clone" — use only the reference audio's speaker
                embedding, skipping in-context conditioning on its transcript.
                Faster and needs no matching transcript, but lower fidelity.
            design_instruct: Natural-language voice description, when
                method == "design" (e.g. "warm elderly British female voice").
            renaming_from: Set when editing an actor under a new name — once
                `voice_name` saves successfully, this old voice's files are
                removed so the rename doesn't leave a duplicate behind.
        """
        panel = self.train_progress_panel
        # "design" has an extra leading step (see set_train_busy) — everything below
        # is indexed off this offset so step numbers stay correct either way.
        step_offset = 1 if method == "design" else 0
        step = 0
        try:
            if method == "design":
                self.root.after(0, lambda: panel.start_step(0))
                self.root.after(
                    0,
                    lambda: self.train_status_label.config(
                        text="Designing reference voice...", foreground="blue"
                    ),
                )
                design_dtype = DTYPE_MAP[self.train_dtype.get()]
                design_model, _ = self._load_model(
                    MODEL_REPO_VOICE_DESIGN, device_type, design_dtype
                )

                def do_design(m):
                    return m.generate_voice_design(
                        text=script_text, instruct=design_instruct, language="Auto"
                    )

                try:
                    design_wavs, design_sr = do_design(design_model)
                except (RuntimeError, torch.cuda.CudaError) as cuda_error:
                    if not _is_cuda_compat_error(cuda_error):
                        raise
                    design_model, _ = self._load_model(
                        MODEL_REPO_VOICE_DESIGN, "cpu", design_dtype
                    )
                    design_wavs, design_sr = do_design(design_model)

                ref_audio = (design_wavs[0], design_sr)
                ref_text = script_text
                self.root.after(0, lambda: panel.complete_step(0))

            step = step_offset
            self.root.after(0, lambda i=step: panel.start_step(i))
            self.root.after(
                0,
                lambda: self.train_status_label.config(
                    text="Loading model...", foreground="blue"
                ),
            )
            model_repo = MODEL_REPOS_BASE[self.train_model_size.get()]
            dtype = DTYPE_MAP[self.train_dtype.get()]
            model, device_type = self._load_model(
                model_repo,
                device_type,
                dtype,
                # Already warned once above if method == "design" loaded first.
                show_warning=(step_offset == 0),
                status_label=self.train_status_label,
            )
            self.root.after(0, lambda i=step: panel.complete_step(i))

            if method == "file":
                audio_file = self.audio_file_entry.get().strip()
                ref_text = self.get_text_value(self.transcript_entry)

                if not audio_file:
                    self.root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Error", "Please select an audio file"
                        ),
                    )
                    return

                if not x_vector_only and not ref_text:
                    self.root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Error",
                            'Please enter a transcript (or check "Quick clone" '
                            "to skip it)",
                        ),
                    )
                    return

                if not os.path.exists(audio_file):
                    self.root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Error", f"Audio file not found: {audio_file}"
                        ),
                    )
                    return

                ref_audio = audio_file

            elif method == "record":
                if not recording_path or not os.path.exists(recording_path):
                    self.root.after(
                        0,
                        lambda: messagebox.showerror(
                            "Error",
                            "Please record audio first using the recording section",
                        ),
                    )
                    return

                ref_audio = recording_path
                ref_text = script_text

            # else: method == "design" — ref_audio/ref_text already set above, from
            # the synthesized reference clip.

            step = step_offset + 1
            self.root.after(0, lambda i=step: panel.start_step(i))
            self.root.after(
                0,
                lambda: self.train_status_label.config(
                    text="Creating voice clone prompt...", foreground="blue"
                ),
            )

            def do_create_prompt(m):
                return m.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only,
                )

            try:
                prompt_items = do_create_prompt(model)
            except (RuntimeError, torch.cuda.CudaError) as cuda_error:
                if not _is_cuda_compat_error(cuda_error):
                    raise
                error_text = str(cuda_error)
                self.root.after(
                    0,
                    lambda: self.train_status_label.config(
                        text="CUDA error during processing, retrying on CPU...",
                        foreground="orange",
                    ),
                )
                self.root.after(
                    0,
                    lambda: messagebox.showwarning(
                        "CUDA Compatibility Issue",
                        f"CUDA error during processing: {error_text}\n\n"
                        "Reloading model on CPU and retrying. Training may be slower.",
                    ),
                )
                model, device_type = self._load_model(
                    model_repo, "cpu", dtype, show_warning=False
                )
                prompt_items = do_create_prompt(model)
            self.root.after(0, lambda i=step: panel.complete_step(i))

            step = step_offset + 2
            self.root.after(0, lambda i=step: panel.start_step(i))
            # Voices always live in the local cache (OUTPUT_DIR) — that's what makes
            # the voice picker's cache scan work.
            output_file = os.path.join(OUTPUT_DIR, f"{voice_name}.pt")
            torch.save(prompt_items, output_file)
            if renaming_from is not None:
                self._delete_voice_files(renaming_from)
            self.root.after(0, lambda i=step: panel.complete_step(i))

            self.root.after(
                0,
                lambda: self.train_status_label.config(
                    text=f"Voice '{voice_name}' trained successfully! Saved to {output_file}",
                    foreground="green",
                ),
            )
            self.root.after(
                0,
                lambda: self.show_success_with_link(
                    "Success",
                    f"Voice '{voice_name}' has been trained and saved to:",
                    output_file,
                ),
            )
            self.root.after(0, self.refresh_voice_lists)

            try:
                # Reuse the model already loaded above — cheap, and means the Cast
                # sidebar's preview is ready without waiting for the next sweep.
                self.generate_voice_preview(
                    "custom",
                    voice_name,
                    model=model,
                    device_type=device_type,
                    dtype=dtype,
                )
            except Exception:
                pass  # best-effort — the next startup sweep will retry
            # Flips the new actor's Cast sidebar button from spinner to ▶ (the
            # refresh above ran too early to catch this — it's scheduled before
            # this preview even starts generating).
            self.root.after(0, self.refresh_cast_sidebar)

            on_created = self._new_actor_on_created
            self.root.after(
                0, lambda: self._finish_new_actor_dialog(voice_name, on_created)
            )

        except Exception as e:
            error_msg = f"Error during training: {e}"
            self.root.after(0, lambda i=step: panel.fail_step(i))
            self.root.after(
                0,
                lambda: self.train_status_label.config(
                    text=error_msg, foreground="red"
                ),
            )
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            self.root.after(0, lambda: self.set_train_busy(False))

    def generate_speech(self):
        """Build segments from `self.script_lines`, resolve a save path, and kick
        off `_generate_speech_thread`. No parsing needed — every line already has a
        real actor, since it only ever came from the actor picker or a validated
        import."""
        output_format = self.output_format.get()
        language = self.use_language_combo.get()

        if not self.script_lines:
            messagebox.showerror("Error", "Add at least one line to the script first")
            return

        segments = [
            (line["kind"], line["key"], line["text"], line["tone"])
            for line in self.script_lines
            if line["text"].strip()
        ]
        if not segments:
            messagebox.showerror("Error", "Every line in the script is empty")
            return

        output_file = self._ask_save_path(segments, output_format)
        if not output_file:
            return

        self.use_progress_panel.set_steps(self._generation_steps(segments))
        self.set_use_busy(True)

        thread = threading.Thread(
            target=self._generate_speech_thread,
            args=(segments, output_format, language, output_file),
        )
        thread.daemon = True
        thread.start()

    def _ask_save_path(self, segments, output_format):
        """Native Save As dialog, defaulting to a dated filename and the last folder
        used this session. Returns the chosen path, or "" if the user cancelled."""
        distinct_voices = {(kind, key) for kind, key, _, _ in segments}
        if len(distinct_voices) == 1:
            _, key = next(iter(distinct_voices))
            filename_stub = f"{key}_output"
        else:
            filename_stub = "multivoice_output"
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        default_filename = f"{filename_stub}_{timestamp}.{output_format}"

        output_file = filedialog.asksaveasfilename(
            title="Save Generated Speech",
            initialdir=self.last_save_dir,
            initialfile=default_filename,
            defaultextension=f".{output_format}",
            filetypes=[(output_format.upper(), f"*.{output_format}")],
        )
        if output_file:
            self.last_save_dir = os.path.dirname(output_file)
        return output_file

    def _generation_steps(self, segments):
        """Step descriptions for the progress panel, matching exactly what
        `_generate_speech_thread` will actually do, in the same order — one step per
        voice-kind group, then Stitch audio (only if there's more than one segment to
        stitch), then Save file."""
        groups, kinds_needed = group_segments_by_kind(segments)
        steps = []
        for kind in kinds_needed:
            names = ", ".join(sorted({key for _, key, _, _ in groups[kind]}))
            kind_label = "voice-clone" if kind == "custom" else "preset"
            steps.append(f"Generate ({kind_label}): {names}")
        if len(segments) > 1:
            steps.append("Stitch audio")
        steps.append("Save file")
        return steps

    def _load_voice_clone_prompt(self, key, device_type):
        """Load a trained custom voice's VoiceClonePromptItem.

        Args:
            key: Voice name; read from OUTPUT_DIR/{key}.pt.
            device_type: "cuda" or "cpu" — where the tensors should end up.

        Returns:
            The VoiceClonePromptItem (train_voice always saves a single-item list,
            since ref_audio there is always a single reference recording).

        Raises:
            FileNotFoundError: If OUTPUT_DIR/{key}.pt doesn't exist.
            Exception: If the file exists but can't be unpickled.
        """
        voice_file = os.path.join(OUTPUT_DIR, f"{key}.pt")
        if not os.path.exists(voice_file):
            raise FileNotFoundError(f"Voice file not found: {voice_file}")

        # torch.load with weights_only (the safe default since PyTorch 2.6) needs
        # VoiceClonePromptItem allowlisted, since it's not a builtin type.
        torch.serialization.add_safe_globals([VoiceClonePromptItem])

        map_location = (
            "cpu" if device_type == "cpu" or not torch.cuda.is_available() else None
        )

        try:
            try:
                if map_location:
                    prompt_items = torch.load(
                        voice_file, map_location=map_location, weights_only=False
                    )
                else:
                    prompt_items = torch.load(voice_file, weights_only=False)
            except TypeError:
                # Older PyTorch versions don't accept weights_only at all.
                if map_location:
                    prompt_items = torch.load(voice_file, map_location=map_location)
                else:
                    prompt_items = torch.load(voice_file)
        except Exception:
            try:
                prompt_items = torch.load(voice_file, map_location="cpu")
            except Exception as e:
                error_details = str(e)
                if (
                    "pickle" in error_details.lower()
                    or "unpickling" in error_details.lower()
                ):
                    raise Exception(
                        f"Failed to load voice file. The file may be corrupted or incompatible. Error: {error_details}"
                    )
                else:
                    raise Exception(
                        f"Failed to load voice file '{voice_file}': {error_details}"
                    )

        return prompt_items[0]

    def _cancelled(self):
        """Whether cancellation was requested. Only meaningful *between* steps — see
        `cancel_generation`'s docstring for why a step already running can't stop
        mid-flight. Also reports it to the status label when true."""
        if not self.generate_cancel_event.is_set():
            return False
        self.root.after(
            0,
            lambda: self.use_status_label.config(
                text="Generation cancelled.", foreground="orange"
            ),
        )
        return True

    def _generate_speech_thread(
        self, segments, output_format="wav", language="Auto", output_file=""
    ):
        """Generate and stitch together speech for parsed [VoiceName] segments.

        Groups segments by model kind ("custom" -> Base, "preset" -> CustomVoice),
        loading each model at most once and generating its group's segments in a
        single batched call, then reassembles the results in original text order.

        Args:
            segments: Ordered (kind, key, text, instruct) tuples built from
                `self.script_lines`. `instruct` is only meaningful for "preset"
                segments — generate_voice_clone has no instruct concept.
            output_format: "wav", "mp3", or "m4a".
            language: One of SUPPORTED_LANGUAGES, applied to every segment.
            output_file: Full path chosen via the Save As dialog in `generate_speech`.
        """
        panel = self.use_progress_panel
        step = 0
        try:
            device_type = self.device_type.get()
            advanced_kwargs = self.advanced_generation_kwargs()

            groups, kinds_needed = group_segments_by_kind(segments)

            wavs_by_index = {}
            sr = None
            loaded_prompts = {}  # custom voice key -> VoiceClonePromptItem

            total_groups = len(kinds_needed)
            model_size = self.generate_model_size.get()
            dtype = DTYPE_MAP[self.generate_dtype.get()]

            for group_index, kind in enumerate(kinds_needed):
                if self._cancelled():
                    return
                step = group_index
                self.root.after(0, lambda i=step: panel.start_step(i))
                group = groups[kind]
                model_repo = (
                    MODEL_REPOS_CUSTOM_VOICE[model_size]
                    if kind == "preset"
                    else MODEL_REPOS_BASE[model_size]
                )

                # Only worth naming the group/voices once there's more than one group —
                # for the common single-voice case this stays exactly as it always was.
                if total_groups > 1:
                    distinct_keys = ", ".join(sorted({k for _, k, _, _ in group}))
                    kind_label = "custom" if kind == "custom" else "preset"
                    phase = f" ({group_index + 1}/{total_groups}: {kind_label} — {distinct_keys})"
                else:
                    phase = ""

                self.root.after(
                    0,
                    lambda p=phase: self.use_status_label.config(
                        text=f"Loading model{p}...", foreground="blue"
                    ),
                )
                model, device_type = self._load_model(
                    model_repo,
                    device_type,
                    dtype,
                    show_warning=(group_index == 0),
                    status_label=self.use_status_label,
                )

                texts = [t for _, _, t, _ in group]

                if kind == "preset":
                    speakers = [k for _, k, _, _ in group]
                    instructs = [instr for _, _, _, instr in group]

                    def do_generate(
                        m,
                        texts=texts,
                        speakers=speakers,
                        instructs=instructs,
                        adv=advanced_kwargs,
                    ):
                        return m.generate_custom_voice(
                            text=texts,
                            speaker=speakers,
                            language=language,
                            instruct=instructs,
                            **adv,
                        )
                else:
                    self.root.after(
                        0,
                        lambda p=phase: self.use_status_label.config(
                            text=f"Loading voice(s){p}...", foreground="blue"
                        ),
                    )
                    for _, key, _, _ in group:
                        if key not in loaded_prompts:
                            loaded_prompts[key] = self._load_voice_clone_prompt(
                                key, device_type
                            )
                    prompts = [loaded_prompts[k] for _, k, _, _ in group]

                    def do_generate(
                        m, texts=texts, prompts=prompts, adv=advanced_kwargs
                    ):
                        return m.generate_voice_clone(
                            text=texts,
                            language=language,
                            voice_clone_prompt=prompts,
                            **adv,
                        )

                self.root.after(
                    0,
                    lambda p=phase: self.use_status_label.config(
                        text=f"Generating speech{p}...", foreground="blue"
                    ),
                )

                try:
                    group_wavs, group_sr = do_generate(model)
                except (RuntimeError, torch.cuda.CudaError) as cuda_error:
                    if not _is_cuda_compat_error(cuda_error):
                        raise
                    error_text = str(cuda_error)
                    self.root.after(
                        0,
                        lambda: self.use_status_label.config(
                            text="CUDA error during generation, retrying on CPU...",
                            foreground="orange",
                        ),
                    )
                    self.root.after(
                        0,
                        lambda error_text=error_text: messagebox.showwarning(
                            "CUDA Compatibility Issue",
                            f"CUDA error during generation: {error_text}\n\n"
                            "Reloading model on CPU and retrying. Generation may be slower.",
                        ),
                    )
                    model, device_type = self._load_model(
                        model_repo, "cpu", dtype, show_warning=False
                    )
                    if kind == "custom":
                        # Prompts loaded above may be pinned to the failed device — reload on
                        # CPU and pass explicitly, since do_generate's default was bound to
                        # the now-stale list.
                        for _, key, _, _ in group:
                            loaded_prompts[key] = self._load_voice_clone_prompt(
                                key, "cpu"
                            )
                        prompts = [loaded_prompts[k] for _, k, _, _ in group]
                        group_wavs, group_sr = do_generate(model, prompts=prompts)
                    else:
                        group_wavs, group_sr = do_generate(model)

                if sr is None:
                    sr = group_sr
                elif sr != group_sr:
                    raise RuntimeError(
                        f"Sample rate mismatch between voice groups ({sr} vs {group_sr}) — "
                        "can't stitch this generation together."
                    )

                for (i, _, _, _), wav in zip(group, group_wavs):
                    wavs_by_index[i] = wav
                self.root.after(0, lambda i=group_index: panel.complete_step(i))

            if self._cancelled():
                return

            if len(segments) > 1:
                step = total_groups
                self.root.after(0, lambda i=step: panel.start_step(i))
                self.root.after(
                    0,
                    lambda: self.use_status_label.config(
                        text="Stitching audio...", foreground="blue"
                    ),
                )

            # Reassemble in original text order, with a short silence gap at each
            # voice change so the splice doesn't sound abrupt.
            silence_gap = np.zeros(int(sr * 0.15), dtype=wavs_by_index[0].dtype)
            pieces = []
            for i, (kind, key, _, _) in enumerate(segments):
                if i > 0 and (kind, key) != (segments[i - 1][0], segments[i - 1][1]):
                    pieces.append(silence_gap)
                pieces.append(wavs_by_index[i])
            final_wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

            if len(segments) > 1:
                self.root.after(0, lambda i=step: panel.complete_step(i))

            if self._cancelled():
                return

            step = total_groups + (1 if len(segments) > 1 else 0)
            self.root.after(0, lambda i=step: panel.start_step(i))

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            if output_format == "wav":
                sf.write(output_file, final_wav, sr)
            else:
                encode_audio(final_wav, sr, output_file, output_format)
            self.root.after(0, lambda i=step: panel.complete_step(i))

            self.root.after(
                0,
                lambda: self.use_status_label.config(
                    text=f"Speech generated successfully! Saved to {output_file}",
                    foreground="green",
                ),
            )
            self.root.after(
                0,
                lambda: self.show_success_with_link(
                    "Success", "Speech has been generated and saved to:", output_file
                ),
            )

        except Exception as e:
            error_msg = f"Error during generation: {e}"
            self.root.after(0, lambda i=step: panel.fail_step(i))
            self.root.after(
                0,
                lambda: self.use_status_label.config(text=error_msg, foreground="red"),
            )
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            self.root.after(0, lambda: self.set_use_busy(False))


def main():
    root = tk.Tk()
    sv_ttk.set_theme("light")
    QwenTTSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
