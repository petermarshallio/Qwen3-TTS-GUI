import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import av
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
import re
import threading
import os
import sys
import subprocess
from datetime import datetime

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # PyInstaller's bundled CUDA DLLs live under _MEIPASS, which isn't on PATH by
    # default — without this, torch.cuda.is_available() reports False in a frozen
    # build even though the DLLs are right there.
    torch_lib_path = os.path.join(sys._MEIPASS, 'torch', 'lib')
    current_path = os.environ.get('PATH', '')
    if os.path.exists(torch_lib_path):
        current_path = torch_lib_path + os.pathsep + current_path
    if sys._MEIPASS not in current_path:
        current_path = sys._MEIPASS + os.pathsep + current_path
    os.environ['PATH'] = current_path

    try:
        import torch._C
        _ = torch.cuda.is_available()  # re-check now that PATH includes the DLLs
    except Exception:
        pass

SAMPLE_RATE = 44100
CHANNELS = 1

# Default output location for trained voices, generated speech, and mic
# recordings — gitignored, so nothing written by the app lands in the repo.
if getattr(sys, 'frozen', False):
    _base_dir = os.path.dirname(sys.executable)
else:
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(_base_dir, "local")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_REPO_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_REPO_CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Fixed speaker list for MODEL_REPO_CUSTOM_VOICE (from the model card). Not fetched
# dynamically via AutoConfig, which would need network/HF-cache access just to
# populate a dropdown before the user has done anything.
PRESET_VOICES = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]

SUPPORTED_LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German",
                        "French", "Russian", "Portuguese", "Spanish", "Italian"]

NEW_VOICE_LABEL = "New..."


def list_custom_voices():
    """Voice names available in the local cache (OUTPUT_DIR/*.pt)."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".pt")
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
        os.path.join(voice_dir, f) for f in os.listdir(voice_dir) if f.lower().endswith(".wav")
    )


VOICE_MARKER_RE = re.compile(r"\[([^\[\]]+)\]")


def build_voice_lookup():
    """Bare voice name (lowercased) -> (kind, key), for resolving [VoiceName] markers.
    Presets are added first so a same-named custom voice takes precedence."""
    lookup = {}
    for speaker in PRESET_VOICES:
        lookup[speaker.lower()] = ("preset", speaker)
    for name in list_custom_voices():
        lookup[name.lower()] = ("custom", name)
    return lookup


def parse_voice_segments(text, default_voice, lookup):
    """Split `text` on [VoiceName] markers into ordered (kind, key, segment_text)
    tuples. Text before the first marker uses `default_voice`; each marker sets the
    active voice for everything after it until the next marker. Empty segments are
    dropped. Returns (segments, unresolved) where `unresolved` lists marker names (as
    typed) that didn't match a known voice in `lookup`."""
    segments = []
    unresolved = []
    voice = default_voice
    pos = 0
    for match in VOICE_MARKER_RE.finditer(text):
        chunk = text[pos:match.start()].strip()
        if chunk:
            segments.append((voice[0], voice[1], chunk))

        name = match.group(1)
        resolved = lookup.get(name.strip().lower())
        if resolved is None:
            unresolved.append(name)
        else:
            voice = resolved
        pos = match.end()

    chunk = text[pos:].strip()
    if chunk:
        segments.append((voice[0], voice[1], chunk))

    return segments, unresolved


def _is_cuda_compat_error(exc):
    """Whether `exc` looks like a CUDA kernel/device compatibility error (as opposed to
    some other RuntimeError) — the case worth silently falling back to CPU for."""
    error_str = str(exc).lower()
    return "cuda" in error_str and ("kernel" in error_str or "no kernel image" in error_str or "device" in error_str)


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

    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(data.astype(np.float32)), format="fltp", layout=layout)
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


class QwenTTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3 TTS Voice Training & Generation")
        self.root.geometry("800x1000")
        
        self.device_type = tk.StringVar(value="cuda")
        self.recording = False
        self.audio_chunks = []
        self.recording_map = {}  # filename -> full path, for the current voice's takes

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Train Voice")
        self.setup_train_tab()

        self.use_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.use_frame, text="Use Voice")
        self.setup_use_tab()

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
                widget.config(foreground="black")
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
                widget.config(foreground="black")
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

    # --- [VoiceName] autocomplete: pop up a filtered voice list when '[' is typed in
    # a text widget, so marker names don't have to be memorized/typed by hand. ---

    def setup_voice_autocomplete(self, text_widget):
        """Bind `text_widget` so typing '[' opens a filtered voice-name popup.
        Enter/Tab/click on a match inserts 'Name]' and closes it. Escape, clicking
        elsewhere, losing focus, or confirming with no match selected are all
        *abandoned* closes — they erase the dangling '[' + partial name so it
        doesn't linger as noise. Typing ']' by hand is a deliberate, complete marker
        typed without the popup's help, so that text is left exactly as typed."""
        state = {"popup": None, "listbox": None, "names": []}
        MARK = "voice_ac_start"

        def close_popup(abandoned=False):
            if state["popup"] is None:
                return
            if abandoned and MARK in text_widget.mark_names() and \
                    text_widget.compare("insert", ">=", MARK):
                text_widget.delete(MARK, "insert")
            state["popup"].destroy()
            state["popup"] = None
            state["listbox"] = None
            if MARK in text_widget.mark_names():
                text_widget.mark_unset(MARK)

        def current_filter():
            return text_widget.get(f"{MARK}+1c", "insert")

        def refresh_listbox():
            filter_text = current_filter().strip().lower()
            matches = [n for n in state["names"] if n.lower().startswith(filter_text)] or \
                [n for n in state["names"] if filter_text in n.lower()]
            listbox = state["listbox"]
            listbox.delete(0, tk.END)
            for name in matches:
                listbox.insert(tk.END, name)
            if matches:
                listbox.selection_set(0)

        def confirm_selection():
            listbox = state["listbox"]
            if not listbox or not listbox.curselection():
                close_popup(abandoned=True)
                return
            name = listbox.get(listbox.curselection()[0])
            text_widget.delete(f"{MARK}+1c", "insert")
            text_widget.insert("insert", f"{name}]")
            close_popup()

        def move_selection(delta):
            listbox = state["listbox"]
            if not listbox or listbox.size() == 0:
                return
            current = listbox.curselection()
            idx = current[0] if current else 0
            idx = max(0, min(listbox.size() - 1, idx + delta))
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(idx)
            listbox.see(idx)

        def open_popup():
            text_widget.mark_set(MARK, "insert - 1c")
            text_widget.mark_gravity(MARK, tk.LEFT)

            names = sorted({name for _, name in build_voice_lookup().values()})
            state["names"] = names
            width = max((len(n) for n in names), default=10) + 2

            popup = tk.Toplevel(text_widget)
            popup.wm_overrideredirect(True)
            popup.wm_attributes("-topmost", True)
            # Padding keeps text clear of the native floating panel's rounded
            # corners, which otherwise visually clip the first/last row.
            frame = tk.Frame(popup, borderwidth=1, relief=tk.SOLID)
            frame.pack(padx=1, pady=1)
            listbox = tk.Listbox(frame, height=6, width=width, exportselection=False,
                                  highlightthickness=0, borderwidth=0)
            listbox.pack(padx=4, pady=4)
            listbox.bind("<ButtonRelease-1>", lambda e: confirm_selection())
            state["popup"] = popup
            state["listbox"] = listbox

            bbox = text_widget.bbox("insert")
            if bbox:
                x, y, _, h = bbox
                popup.wm_geometry(
                    f"+{text_widget.winfo_rootx() + x}+{text_widget.winfo_rooty() + y + h}")
            refresh_listbox()

        def on_key_release(event):
            if state["popup"] is None:
                if event.char == "[":
                    open_popup()
                return
            if event.keysym == "Escape":
                close_popup(abandoned=True)
                return
            if event.keysym in ("Down", "Up", "Return", "Tab"):
                # Already fully handled in on_key_press (with "break"). Falling through
                # to refresh_listbox() below would re-select index 0 on every arrow
                # press, undoing the move_selection() that just ran.
                return
            filter_text = current_filter()
            if "]" in filter_text:
                # Typed a complete marker by hand — leave it as-is.
                close_popup()
                return
            if "\n" in filter_text or text_widget.compare("insert", "<=", MARK):
                # Newline (e.g. pasted text) or the cursor moved back onto/before the
                # marker (e.g. backspaced past it) — nothing coherent to keep filtering.
                close_popup(abandoned=True)
                return
            refresh_listbox()

        def on_key_press(event):
            if state["popup"] is None:
                return None
            if event.keysym == "Down":
                move_selection(1)
                return "break"
            if event.keysym == "Up":
                move_selection(-1)
                return "break"
            if event.keysym in ("Return", "Tab"):
                confirm_selection()
                return "break"
            return None

        text_widget.bind("<KeyRelease>", on_key_release, add="+")
        text_widget.bind("<KeyPress>", on_key_press, add="+")
        text_widget.bind("<Button-1>", lambda e: close_popup(abandoned=True), add="+")
        text_widget.bind("<FocusOut>", lambda e: close_popup(abandoned=True), add="+")

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
            self.train_progress.pack(fill=tk.X, padx=20, pady=(0, 10), before=self.train_status_label)
            self.train_progress.start(10)
        else:
            self.train_progress.stop()
            self.train_progress.pack_forget()

        self._set_frame_enabled(self.train_frame, not busy)
        self.train_button.config(text="Training..." if busy else "Train Voice")
        if not busy and self.train_voice_combo.get() != NEW_VOICE_LABEL:
            # Voice Name must stay disabled (auto-filled) when updating an existing
            # voice — re-set just the state, without touching its content.
            self.voice_name_entry.config(state=tk.DISABLED)

    def set_use_busy(self, busy):
        if busy:
            self.use_progress.pack(fill=tk.X, padx=20, pady=(0, 10), before=self.use_status_label)
            self.use_progress.start(10)
        else:
            self.use_progress.stop()
            self.use_progress.pack_forget()

        self._set_frame_enabled(self.use_frame, not busy)
        self.generate_button.config(text="Generating..." if busy else "Generate Speech")
        if not busy:
            # Instruction box's visibility depends on whether a preset is selected.
            self.on_use_voice_selected()

    def show_success_with_link(self, title, message, file_path):
        """Success dialog with a clickable link that reveals the generated file."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        ttk.Label(dialog, text=message, wraplength=400, justify=tk.LEFT).pack(padx=20, pady=(20, 10))

        link = ttk.Label(dialog, text=file_path, foreground="blue", cursor="hand2",
                          font=("TkDefaultFont", 10, "underline"), wraplength=400, justify=tk.LEFT)
        link.pack(padx=20, pady=(0, 10))
        link.bind("<Button-1>", lambda e: reveal_in_file_manager(file_path))

        ttk.Button(dialog, text="OK", command=dialog.destroy).pack(pady=(0, 20))
        dialog.grab_set()

    def setup_train_tab(self):
        # Title
        title_label = ttk.Label(self.train_frame, text="Train a New Voice", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Voice picker: "New..." to create a voice, or an existing custom voice to
        # re-train/overwrite it. Presets aren't shown here — they aren't trainable.
        voice_frame = ttk.Frame(self.train_frame)
        voice_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(voice_frame, text="Voice:").pack(side=tk.LEFT, padx=5)
        self.train_voice_combo = ttk.Combobox(voice_frame, width=27, state="readonly")
        self.train_voice_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.train_voice_combo.bind("<<ComboboxSelected>>", lambda e: self.on_train_voice_selected())

        name_frame = ttk.Frame(self.train_frame)
        name_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(name_frame, text="Voice Name:").pack(side=tk.LEFT, padx=5)
        self.voice_name_entry = ttk.Entry(name_frame, width=30)
        self.voice_name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        # Recordings are filed under this name (local/{name}/mic_*.wav), so the
        # recording picker below needs to follow it as the user types.
        self.voice_name_entry.bind("<KeyRelease>", lambda e: self.refresh_recording_list(), add="+")

        # Device selection
        device_frame = ttk.LabelFrame(self.train_frame, text="Device Selection", padding=10)
        device_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Radiobutton(device_frame, text="CUDA (GPU)", variable=self.device_type, 
                       value="cuda").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_type, 
                       value="cpu").pack(side=tk.LEFT, padx=10)
        
        # Training method selection
        method_frame = ttk.LabelFrame(self.train_frame, text="Training Method", padding=10)
        method_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.train_method = tk.StringVar(value="file")
        ttk.Radiobutton(method_frame, text="From Audio File + Transcript",
                       variable=self.train_method, value="file",
                       command=self.update_train_method_visibility).pack(anchor=tk.W, pady=5)
        ttk.Radiobutton(method_frame, text="Record Audio (Pre-made Script)",
                       variable=self.train_method, value="record",
                       command=self.update_train_method_visibility).pack(anchor=tk.W, pady=5)

        # File input section (shown only for the "file" method)
        self.file_frame = ttk.LabelFrame(self.train_frame, text="File Input", padding=10)

        audio_file_frame = ttk.Frame(self.file_frame)
        audio_file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(audio_file_frame, text="Audio File:").pack(side=tk.LEFT, padx=5)
        self.audio_file_entry = ttk.Entry(audio_file_frame, width=40)
        self.audio_file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(audio_file_frame, text="Browse", command=self.browse_audio_file).pack(side=tk.LEFT, padx=5)
        
        transcript_frame = ttk.Frame(self.file_frame)
        transcript_frame.pack(fill=tk.X, pady=5)
        ttk.Label(transcript_frame, text="Transcript:").pack(anchor=tk.W, padx=5)
        self.transcript_entry = scrolledtext.ScrolledText(transcript_frame, height=4, width=50, wrap=tk.WORD)
        self.transcript_entry.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.add_placeholder_text(self.transcript_entry, "Enter the exact transcript of the reference audio...")
        self.transcript_counter_label = ttk.Label(transcript_frame, text="", foreground="gray")
        self.transcript_counter_label.pack(anchor=tk.E, padx=5)
        self.bind_text_counter(self.transcript_entry, self.transcript_counter_label)

        # Recording section (shown only for the "record" method)
        self.record_frame = ttk.LabelFrame(self.train_frame, text="Recording", padding=10)

        self.script_text = "On a warm Saturday morning, the quick brown fox jumped over several lazy dogs while distant musicians played jazzy tunes near the quiet park. People checked their phones, argued about numbers, dates, and prices, and casually mentioned names like Alex, Jordan, and Taylor. A cyclist shouted warnings, a train horn echoed, and someone asked, ‘Why does this even matter?’ as rain began falling lightly at exactly 9:47 a.m., changing plans, moods, and expectations all at once."
        ttk.Label(self.record_frame, text="Script to read:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        script_display = scrolledtext.ScrolledText(self.record_frame, height=7, width=50, wrap=tk.WORD)
        script_display.insert("1.0", self.script_text)
        script_display.config(state=tk.DISABLED)
        script_display.pack(fill=tk.X, pady=5)

        self.record_status_label = ttk.Label(self.record_frame, text="Ready to record", foreground="green")
        self.record_status_label.pack(pady=5)

        self.record_button = ttk.Button(self.record_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=5)

        # Every take is kept (local/{voice}/mic_{timestamp}.wav) rather than
        # overwriting a single fixed file — this picks which one Train Voice uses.
        take_frame = ttk.Frame(self.record_frame)
        take_frame.pack(fill=tk.X, pady=5)
        ttk.Label(take_frame, text="Take to use:").pack(side=tk.LEFT, padx=5)
        self.recording_combo = ttk.Combobox(take_frame, width=30, state="disabled")
        self.recording_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Status and progress
        self.train_status_label = ttk.Label(self.train_frame, text="", foreground="blue")
        self.train_status_label.pack(pady=10)

        self.train_progress = ttk.Progressbar(self.train_frame, mode="indeterminate")
        # Not packed here — shown only while training is in progress (set_train_busy).

        # Train button
        self.train_button = ttk.Button(self.train_frame, text="Train Voice", command=self.train_voice,
                                 style="Accent.TButton")
        self.train_button.pack(pady=20)

        self.update_train_method_visibility()
        self.refresh_train_voice_list()

    def update_train_method_visibility(self):
        """Show only the File Input or Recording section, matching the selected training method."""
        if self.train_method.get() == "file":
            self.record_frame.pack_forget()
            self.file_frame.pack(fill=tk.X, padx=20, pady=10, before=self.train_status_label)
        else:
            self.file_frame.pack_forget()
            self.record_frame.pack(fill=tk.X, padx=20, pady=10, before=self.train_status_label)

    def refresh_train_voice_list(self):
        """Repopulate the Train tab's voice picker from the local voice cache."""
        current = self.train_voice_combo.get()
        values = [NEW_VOICE_LABEL] + list_custom_voices()
        self.train_voice_combo["values"] = values
        if current in values:
            self.train_voice_combo.set(current)
        else:
            self.train_voice_combo.set(NEW_VOICE_LABEL)
        self.on_train_voice_selected()

    def on_train_voice_selected(self):
        """Editable+empty for a new voice; disabled+filled for an existing one (update flow)."""
        selected = self.train_voice_combo.get()
        self.voice_name_entry.config(state=tk.NORMAL)
        self.voice_name_entry.delete(0, tk.END)
        if selected != NEW_VOICE_LABEL:
            self.voice_name_entry.insert(0, selected)
            self.voice_name_entry.config(state=tk.DISABLED)
        self.refresh_recording_list()

    def refresh_recording_list(self):
        """Repopulate the "Take to use" picker for the current Voice Name. Called
        whenever that name changes, and after every new recording (which always
        becomes the selection)."""
        voice_name = self.voice_name_entry.get().strip()
        self.recording_map = {os.path.basename(p): p for p in list_voice_recordings(voice_name)}
        values = list(self.recording_map.keys())
        self.recording_combo["values"] = values
        if values:
            self.recording_combo.set(values[-1])
            self.recording_combo.config(state="readonly")
        else:
            self.recording_combo.set("")
            self.recording_combo.config(state="disabled")

    def refresh_voice_lists(self):
        """Repopulate both tabs' voice pickers — called after a voice is (re-)trained
        so it shows up immediately without restarting the app."""
        self.refresh_train_voice_list()
        self.refresh_use_voice_list()

    def setup_use_tab(self):
        # Title
        title_label = ttk.Label(self.use_frame, text="Generate Speech from a pretrained Voice", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Voice selection: custom voices from the local cache + read-only presets
        voice_frame = ttk.LabelFrame(self.use_frame, text="Voice Selection", padding=10)
        voice_frame.pack(fill=tk.X, padx=20, pady=10)

        voice_select_frame = ttk.Frame(voice_frame)
        voice_select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(voice_select_frame, text="Voice:").pack(side=tk.LEFT, padx=5)
        self.use_voice_combo = ttk.Combobox(voice_select_frame, width=37, state="readonly")
        self.use_voice_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.use_voice_combo.bind("<<ComboboxSelected>>", lambda e: self.on_use_voice_selected())

        language_frame = ttk.Frame(voice_frame)
        language_frame.pack(fill=tk.X, pady=5)
        ttk.Label(language_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        self.use_language_combo = ttk.Combobox(language_frame, width=37, state="readonly",
                                                values=SUPPORTED_LANGUAGES)
        self.use_language_combo.set("Auto")
        self.use_language_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Instruction — only meaningful for preset voices (generate_voice_clone has no
        # instruct parameter at all). Stays visible for both, greyed out and explained
        # rather than hidden for custom voices — see on_use_voice_selected.
        self.instruct_frame = ttk.Frame(voice_frame)
        self.instruct_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.instruct_frame, text="Instruction:").pack(side=tk.LEFT, padx=5)
        self.instruct_entry = ttk.Entry(self.instruct_frame, width=37)
        self.instruct_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.add_placeholder_entry(self.instruct_entry, "e.g. say it in an angry tone")
        # add_placeholder_entry's own FocusOut always restores the placeholder text it
        # was given above; re-run afterwards so the hint matches the current voice kind.
        self.instruct_entry.bind("<FocusOut>", lambda e: self.on_use_voice_selected(), add="+")

        # Device selection
        device_frame = ttk.LabelFrame(self.use_frame, text="Device Selection", padding=10)
        device_frame.pack(fill=tk.X, padx=20, pady=10)

        self.use_device_type = tk.StringVar(value="cuda")
        ttk.Radiobutton(device_frame, text="CUDA (GPU)", variable=self.use_device_type,
                       value="cuda").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.use_device_type,
                       value="cpu").pack(side=tk.LEFT, padx=10)

        # Text input
        text_frame = ttk.LabelFrame(self.use_frame, text="Text to Generate", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        ttk.Label(text_frame, text="Tip: type [VoiceName] to switch voices from that "
                  "point on — try typing '['.", foreground="gray").pack(anchor=tk.W, pady=(0, 5))

        self.text_entry = scrolledtext.ScrolledText(text_frame, height=8, width=50, wrap=tk.WORD)
        self.text_entry.pack(fill=tk.BOTH, expand=True)
        self.add_placeholder_text(self.text_entry, "Enter text to synthesize...")
        self.text_counter_label = ttk.Label(text_frame, text="", foreground="gray")
        self.text_counter_label.pack(anchor=tk.E)
        self.bind_text_counter(self.text_entry, self.text_counter_label)
        self.setup_voice_autocomplete(self.text_entry)

        # Output format selection
        format_frame = ttk.LabelFrame(self.use_frame, text="Output Format", padding=10)
        format_frame.pack(fill=tk.X, padx=20, pady=10)

        self.output_format = tk.StringVar(value="wav")
        ttk.Radiobutton(format_frame, text="WAV", variable=self.output_format,
                       value="wav").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(format_frame, text="MP3", variable=self.output_format,
                       value="mp3").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(format_frame, text="M4A", variable=self.output_format,
                       value="m4a").pack(side=tk.LEFT, padx=10)

        # Save location for output
        save_frame = ttk.Frame(self.use_frame)
        save_frame.pack(fill=tk.X, padx=20, pady=10)
        ttk.Label(save_frame, text="Save Location:").pack(side=tk.LEFT, padx=5)
        self.use_save_entry = ttk.Entry(save_frame, width=40)
        self.use_save_entry.insert(0, OUTPUT_DIR)
        self.use_save_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(save_frame, text="Browse", command=self.browse_save_location_use).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.use_status_label = ttk.Label(self.use_frame, text="", foreground="blue")
        self.use_status_label.pack(pady=10)

        self.use_progress = ttk.Progressbar(self.use_frame, mode="indeterminate")
        # Not packed here — shown only while generation is in progress (set_use_busy).

        # Generate button
        self.generate_button = ttk.Button(self.use_frame, text="Generate Speech", command=self.generate_speech,
                                    style="Accent.TButton")
        self.generate_button.pack(pady=20)

        self.use_voice_map = {}
        self.refresh_use_voice_list()

    def refresh_use_voice_list(self):
        """Repopulate the Use tab's voice picker with local custom voices + presets."""
        current = self.use_voice_combo.get()
        self.use_voice_map = {}
        values = []
        for name in list_custom_voices():
            self.use_voice_map[name] = ("custom", name)
            values.append(name)
        for speaker in PRESET_VOICES:
            label = f"{speaker} (Preset)"
            self.use_voice_map[label] = ("preset", speaker)
            values.append(label)

        self.use_voice_combo["values"] = values
        if current in self.use_voice_map:
            self.use_voice_combo.set(current)
        elif values:
            self.use_voice_combo.set(values[0])
        else:
            self.use_voice_combo.set("")
        self.on_use_voice_selected()

    def on_use_voice_selected(self):
        """Instruction only does anything for preset voices (generate_voice_clone has no
        instruct parameter at all). Stays editable even when the default voice is a
        custom voice, since a [PresetName] marker later in the text may still need one
        — only the placeholder hint text changes, and only while empty."""
        kind, _ = self.use_voice_map.get(self.use_voice_combo.get(), (None, None))
        entry = self.instruct_entry
        entry.config(state=tk.NORMAL)

        hint = "e.g. say it in an angry tone" if kind == "preset" else \
            "Only applies to preset voices, incl. any [VoiceName] segments."
        if getattr(entry, "showing_placeholder", False):
            entry.delete(0, tk.END)
            entry.insert(0, hint)
            entry.config(foreground="grey")

    def browse_audio_file(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            self.audio_file_entry.delete(0, tk.END)
            self.audio_file_entry.insert(0, filename)
    
    def browse_save_location_use(self):
        """Browse for save folder when generating speech"""
        folder = filedialog.askdirectory(
            title="Select Folder to Save Generated Speech",
            initialdir=OUTPUT_DIR
        )
        if folder:
            self.use_save_entry.delete(0, tk.END)
            self.use_save_entry.insert(0, folder)
    
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
        self.record_status_label.config(text="Recording... Press Stop to finish", foreground="red")
        self.record_button.config(text="Stop Recording")
        
        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_chunks.append(indata.copy())
        
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback)
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'stream'):
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
            self.record_status_label.config(text=f"Recording saved to {recording_path}", foreground="green")
        else:
            self.record_status_label.config(text="No audio recorded", foreground="orange")
        
        self.record_button.config(text="Start Recording")
    
    def get_model_config(self, device_type, show_warning=True):
        """Get model configuration based on device type"""
        # Check if CUDA is requested but not available
        if device_type == "cuda" and not torch.cuda.is_available():
            # Provide detailed diagnostic information
            cuda_info = []
            cuda_info.append(f"PyTorch version: {torch.__version__}")
            cuda_info.append(f"CUDA available: {torch.cuda.is_available()}")
            if hasattr(torch.version, 'cuda'):
                cuda_info.append(f"PyTorch CUDA version: {torch.version.cuda}")
            if hasattr(sys, '_MEIPASS'):
                cuda_info.append(f"Running as bundled executable")
                torch_lib = os.path.join(sys._MEIPASS, 'torch', 'lib')
                cuda_info.append(f"Torch lib path exists: {os.path.exists(torch_lib)}")
            else:
                cuda_info.append(f"Running as Python script")
            
            if show_warning:
                info_msg = "CUDA is not available. Falling back to CPU mode.\n\n" + "\n".join(cuda_info)
                self.root.after(0, lambda: messagebox.showwarning(
                    "CUDA Not Available", info_msg))
            device_type = "cpu"
        
        # Return CUDA config if CUDA is available and requested
        if device_type == "cuda" and torch.cuda.is_available():
            # Try flash_attention_2, fallback to sdpa if not available
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
            
            return {
                "device_map": "auto",
                "dtype": torch.bfloat16,
                "attn_implementation": attn_impl,
            }
        else:
            # CPU configuration
            return {
                "device_map": "cpu",
                "dtype": torch.float32,
                "attn_implementation": "eager",
            }

    def _load_model(self, model_repo, device_type, show_warning=True, status_label=None):
        """Load `model_repo`, falling back to CPU and retrying once on a CUDA
        kernel/device compatibility error. Returns (model, effective_device_type)."""
        config = self.get_model_config(device_type, show_warning=show_warning)
        try:
            return Qwen3TTSModel.from_pretrained(model_repo, **config), device_type
        except (RuntimeError, torch.cuda.CudaError) as cuda_error:
            if not _is_cuda_compat_error(cuda_error):
                raise
            if status_label is not None:
                self.root.after(0, lambda: status_label.config(
                    text="CUDA error detected, falling back to CPU...", foreground="orange"))
            self.root.after(0, lambda: messagebox.showwarning(
                "CUDA Compatibility Issue",
                f"CUDA error detected: {cuda_error}\n\nFalling back to CPU mode. This may be slower."))
            config = self.get_model_config("cpu", show_warning=False)
            return Qwen3TTSModel.from_pretrained(model_repo, **config), "cpu"

    def train_voice(self):
        voice_name = self.voice_name_entry.get().strip()
        if not voice_name:
            messagebox.showerror("Error", "Please enter a voice name")
            return

        if self.train_voice_combo.get() == NEW_VOICE_LABEL and voice_name in list_custom_voices():
            if not messagebox.askyesno(
                "Overwrite Voice?",
                f"A voice named '{voice_name}' already exists. Overwrite it?"
            ):
                return

        method = self.train_method.get()
        device_type = self.device_type.get()

        recording_path = None
        if method == "record":
            recording_path = self.recording_map.get(self.recording_combo.get())

        self.set_train_busy(True)

        # Run training in a separate thread to avoid blocking UI
        thread = threading.Thread(
            target=self._train_voice_thread,
            args=(voice_name, method, device_type, recording_path))
        thread.daemon = True
        thread.start()

    def _train_voice_thread(self, voice_name, method, device_type, recording_path=None):
        try:
            self.root.after(0, lambda: self.train_status_label.config(
                text="Loading model...", foreground="blue"))

            # Load model
            config = self.get_model_config(device_type, show_warning=True)
            try:
                model = Qwen3TTSModel.from_pretrained(
                    MODEL_REPO_BASE,
                    **config
                )
            except (RuntimeError, torch.cuda.CudaError) as cuda_error:
                # Check if it's a CUDA compatibility error
                error_str = str(cuda_error).lower()
                if "cuda" in error_str and ("kernel" in error_str or "no kernel image" in error_str or "device" in error_str):
                    # CUDA error - fall back to CPU
                    device_type = "cpu"  # Update device_type for fallback
                    self.root.after(0, lambda: self.train_status_label.config(
                        text="CUDA error detected, falling back to CPU...", foreground="orange"))
                    self.root.after(0, lambda: messagebox.showwarning(
                        "CUDA Compatibility Issue",
                        f"CUDA error detected: {str(cuda_error)}\n\nFalling back to CPU mode. Training may be slower."))
                    # Retry with CPU config
                    config = self.get_model_config("cpu", show_warning=False)
                    model = Qwen3TTSModel.from_pretrained(
                        MODEL_REPO_BASE,
                        **config
                    )
                else:
                    # Re-raise if it's not a CUDA compatibility error
                    raise

            ref_audio = None
            ref_text = None
            
            if method == "file":
                audio_file = self.audio_file_entry.get().strip()
                ref_text = self.get_text_value(self.transcript_entry)
                
                if not audio_file:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Please select an audio file"))
                    return
                
                if not ref_text:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Please enter a transcript"))
                    return
                
                if not os.path.exists(audio_file):
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Audio file not found: {audio_file}"))
                    return
                
                ref_audio = audio_file
                
            else:  # recording
                if not recording_path or not os.path.exists(recording_path):
                    self.root.after(0, lambda: messagebox.showerror("Error",
                        "Please record audio first using the recording section"))
                    return

                ref_audio = recording_path
                ref_text = self.script_text
            
            self.root.after(0, lambda: self.train_status_label.config(
                text="Creating voice clone prompt...", foreground="blue"))
            
            # Generate voice clone prompt
            try:
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
            except (RuntimeError, torch.cuda.CudaError) as cuda_error:
                # Check if it's a CUDA compatibility error during operation
                error_str = str(cuda_error).lower()
                if "cuda" in error_str and ("kernel" in error_str or "no kernel image" in error_str or "device" in error_str):
                    # CUDA error during operation - reload model on CPU and retry
                    device_type = "cpu"  # Update device_type for fallback
                    self.root.after(0, lambda: self.train_status_label.config(
                        text="CUDA error during processing, retrying on CPU...", foreground="orange"))
                    self.root.after(0, lambda: messagebox.showwarning(
                        "CUDA Compatibility Issue",
                        f"CUDA error during processing: {str(cuda_error)}\n\nReloading model on CPU and retrying. Training may be slower."))
                    # Reload model with CPU config
                    config = self.get_model_config("cpu", show_warning=False)
                    model = Qwen3TTSModel.from_pretrained(
                        MODEL_REPO_BASE,
                        **config
                    )
                    # Retry the operation
                    prompt_items = model.create_voice_clone_prompt(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        x_vector_only_mode=False,
                    )
                else:
                    # Re-raise if it's not a CUDA compatibility error
                    raise
            
            # Voices always live in the local cache (OUTPUT_DIR) — that's what makes
            # the voice picker's cache scan work.
            output_file = os.path.join(OUTPUT_DIR, f"{voice_name}.pt")

            # Save the prompt
            torch.save(prompt_items, output_file)

            self.root.after(0, lambda: self.train_status_label.config(
                text=f"Voice '{voice_name}' trained successfully! Saved to {output_file}",
                foreground="green"))
            self.root.after(0, lambda: self.show_success_with_link(
                "Success", f"Voice '{voice_name}' has been trained and saved to:", output_file))
            self.root.after(0, self.refresh_voice_lists)

        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            self.root.after(0, lambda: self.train_status_label.config(
                text=error_msg, foreground="red"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            self.root.after(0, lambda: self.set_train_busy(False))

    def generate_speech(self):
        text = self.get_text_value(self.text_entry)
        output_format = self.output_format.get()
        language = self.use_language_combo.get()
        instruct = self.get_entry_value(self.instruct_entry)

        selected = self.use_voice_combo.get()
        default_voice = self.use_voice_map.get(selected)
        if not default_voice:
            messagebox.showerror("Error", "Please select a voice")
            return

        if not text:
            messagebox.showerror("Error", "Please enter text to generate")
            return

        segments, unresolved = parse_voice_segments(text, default_voice, build_voice_lookup())
        if unresolved:
            names = ", ".join(f"[{name}]" for name in unresolved)
            messagebox.showerror(
                "Unknown Voice",
                f"Unknown voice(s) in the text: {names}\n\n"
                "Check spelling against the Voice dropdown / your trained voices.")
            return

        if not segments:
            messagebox.showerror("Error", "Please enter text to generate")
            return

        self.set_use_busy(True)

        # Run generation in a separate thread
        thread = threading.Thread(
            target=self._generate_speech_thread,
            args=(segments, output_format, language, instruct)
        )
        thread.daemon = True
        thread.start()

    def _load_voice_clone_prompt(self, key, device_type):
        """Load a custom voice's saved VoiceClonePromptItem from OUTPUT_DIR/{key}.pt
        (train_voice always saves a single-item list, since ref_audio there is always
        a single reference recording)."""
        voice_file = os.path.join(OUTPUT_DIR, f"{key}.pt")
        if not os.path.exists(voice_file):
            raise FileNotFoundError(f"Voice file not found: {voice_file}")

        # Allowlist the class
        torch.serialization.add_safe_globals([VoiceClonePromptItem])

        # Use map_location to handle CPU/GPU properly
        map_location = "cpu" if device_type == "cpu" or not torch.cuda.is_available() else None

        try:
            # Try loading with weights_only parameter (PyTorch 2.0+)
            try:
                if map_location:
                    prompt_items = torch.load(voice_file, map_location=map_location, weights_only=False)
                else:
                    prompt_items = torch.load(voice_file, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions that don't support weights_only
                if map_location:
                    prompt_items = torch.load(voice_file, map_location=map_location)
                else:
                    prompt_items = torch.load(voice_file)
        except Exception:
            # Try loading on CPU as final fallback
            try:
                prompt_items = torch.load(voice_file, map_location="cpu")
            except Exception as e:
                error_details = str(e)
                if "pickle" in error_details.lower() or "unpickling" in error_details.lower():
                    raise Exception(f"Failed to load voice file. The file may be corrupted or incompatible. Error: {error_details}")
                else:
                    raise Exception(f"Failed to load voice file '{voice_file}': {error_details}")

        return prompt_items[0]

    def _generate_speech_thread(self, segments, output_format="wav", language="Auto", instruct=""):
        try:
            device_type = self.use_device_type.get()

            groups = {}  # kind -> list of (segment_index, key, text)
            for i, (kind, key, seg_text) in enumerate(segments):
                groups.setdefault(kind, []).append((i, key, seg_text))
            kinds_needed = [k for k in ("custom", "preset") if groups.get(k)]

            wavs_by_index = {}
            sr = None
            loaded_prompts = {}  # custom voice key -> VoiceClonePromptItem

            total_groups = len(kinds_needed)

            for group_index, kind in enumerate(kinds_needed):
                group = groups[kind]
                model_repo = MODEL_REPO_CUSTOM_VOICE if kind == "preset" else MODEL_REPO_BASE

                # Only worth naming the group/voices once there's more than one group —
                # for the common single-voice case this stays exactly as it always was.
                if total_groups > 1:
                    distinct_keys = ", ".join(sorted({k for _, k, _ in group}))
                    kind_label = "custom" if kind == "custom" else "preset"
                    phase = f" ({group_index + 1}/{total_groups}: {kind_label} — {distinct_keys})"
                else:
                    phase = ""

                self.root.after(0, lambda p=phase: self.use_status_label.config(
                    text=f"Loading model{p}...", foreground="blue"))
                model, device_type = self._load_model(
                    model_repo, device_type, show_warning=(group_index == 0),
                    status_label=self.use_status_label)

                texts = [t for _, _, t in group]

                if kind == "preset":
                    speakers = [k for _, k, _ in group]

                    def do_generate(m):
                        return m.generate_custom_voice(
                            text=texts,
                            speaker=speakers,
                            language=language,
                            instruct=instruct or None,
                        )
                else:
                    self.root.after(0, lambda p=phase: self.use_status_label.config(
                        text=f"Loading voice(s){p}...", foreground="blue"))
                    for _, key, _ in group:
                        if key not in loaded_prompts:
                            loaded_prompts[key] = self._load_voice_clone_prompt(key, device_type)
                    prompts = [loaded_prompts[k] for _, k, _ in group]

                    def do_generate(m):
                        return m.generate_voice_clone(
                            text=texts,
                            language=language,
                            voice_clone_prompt=prompts,
                        )

                self.root.after(0, lambda p=phase: self.use_status_label.config(
                    text=f"Generating speech{p}...", foreground="blue"))

                try:
                    group_wavs, group_sr = do_generate(model)
                except (RuntimeError, torch.cuda.CudaError) as cuda_error:
                    if not _is_cuda_compat_error(cuda_error):
                        raise
                    self.root.after(0, lambda: self.use_status_label.config(
                        text="CUDA error during generation, retrying on CPU...", foreground="orange"))
                    self.root.after(0, lambda: messagebox.showwarning(
                        "CUDA Compatibility Issue",
                        f"CUDA error during generation: {cuda_error}\n\n"
                        "Reloading model on CPU and retrying. Generation may be slower."))
                    model, device_type = self._load_model(model_repo, "cpu", show_warning=False)
                    if kind == "custom":
                        # Prompts loaded above may be pinned to the failed device — reload on CPU.
                        for _, key, _ in group:
                            loaded_prompts[key] = self._load_voice_clone_prompt(key, "cpu")
                        prompts = [loaded_prompts[k] for _, k, _ in group]
                    group_wavs, group_sr = do_generate(model)

                if sr is None:
                    sr = group_sr
                elif sr != group_sr:
                    raise RuntimeError(
                        f"Sample rate mismatch between voice groups ({sr} vs {group_sr}) — "
                        "can't stitch this generation together.")

                for (i, _, _), wav in zip(group, group_wavs):
                    wavs_by_index[i] = wav

            if len(segments) > 1:
                self.root.after(0, lambda: self.use_status_label.config(
                    text="Stitching audio...", foreground="blue"))

            # Reassemble in original text order, with a short silence gap at each
            # voice change so the splice doesn't sound abrupt.
            silence_gap = np.zeros(int(sr * 0.15), dtype=wavs_by_index[0].dtype)
            pieces = []
            for i, (kind, key, _) in enumerate(segments):
                if i > 0 and (kind, key) != (segments[i - 1][0], segments[i - 1][1]):
                    pieces.append(silence_gap)
                pieces.append(wavs_by_index[i])
            final_wav = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

            # Determine save location
            save_folder = self.use_save_entry.get().strip()
            output_dir = save_folder if save_folder else OUTPUT_DIR
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            distinct_voices = {(kind, key) for kind, key, _ in segments}
            if len(distinct_voices) == 1:
                _, key = next(iter(distinct_voices))
                filename_stub = f"{key}_output"
            else:
                filename_stub = "multivoice_output"

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_file = os.path.join(output_dir, f"{filename_stub}_{timestamp}.{output_format}")

            if output_format == "wav":
                sf.write(output_file, final_wav, sr)
            else:
                encode_audio(final_wav, sr, output_file, output_format)

            self.root.after(0, lambda: self.use_status_label.config(
                text=f"Speech generated successfully! Saved to {output_file}",
                foreground="green"))
            self.root.after(0, lambda: self.show_success_with_link(
                "Success", "Speech has been generated and saved to:", output_file))

        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            self.root.after(0, lambda: self.use_status_label.config(
                text=error_msg, foreground="red"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        finally:
            self.root.after(0, lambda: self.set_use_busy(False))

def main():
    root = tk.Tk()
    app = QwenTTSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
