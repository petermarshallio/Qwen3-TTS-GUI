import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import torch
import sounddevice as sd
import soundfile as sf
import numpy as np
import av
from qwen_tts import Qwen3TTSModel
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
import threading
import os
import sys

# Fix CUDA detection in PyInstaller bundles
# PyInstaller sometimes doesn't properly detect CUDA libraries
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    # Add PyInstaller's temporary directory to PATH so CUDA DLLs can be found
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # Add torch/lib subdirectory to PATH for CUDA DLLs (highest priority)
        torch_lib_path = os.path.join(sys._MEIPASS, 'torch', 'lib')
        if os.path.exists(torch_lib_path):
            # Add to front of PATH so it's checked first
            current_path = os.environ.get('PATH', '')
            os.environ['PATH'] = torch_lib_path + os.pathsep + current_path
        
        # Also add the main _MEIPASS directory
        current_path = os.environ.get('PATH', '')
        if sys._MEIPASS not in current_path:
            os.environ['PATH'] = sys._MEIPASS + os.pathsep + current_path
        
        # Force PyTorch to re-check CUDA availability after PATH update
        # This helps if CUDA DLLs weren't found initially
        try:
            import torch._C
            # Trigger CUDA initialization by checking availability
            _ = torch.cuda.is_available()
        except Exception:
            pass  # CUDA might not be available, that's okay

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
MIC_RECORDING_PATH = os.path.join(OUTPUT_DIR, "mic.wav")

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


class QwenTTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3 TTS Voice Training & Generation")
        self.root.geometry("800x1000")
        
        # Variables
        self.device_type = tk.StringVar(value="cuda")
        self.recording = False
        self.audio_chunks = []
        self.model = None
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Train Voice Tab
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Train Voice")
        self.setup_train_tab()
        
        # Use Voice Tab
        self.use_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.use_frame, text="Use Voice")
        self.setup_use_tab()
        
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
        self.transcript_entry = scrolledtext.ScrolledText(transcript_frame, height=4, width=50)
        self.transcript_entry.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
        
        # Status and progress
        self.train_status_label = ttk.Label(self.train_frame, text="", foreground="blue")
        self.train_status_label.pack(pady=10)
        
        # Train button
        train_button = ttk.Button(self.train_frame, text="Train Voice", command=self.train_voice,
                                 style="Accent.TButton")
        train_button.pack(pady=20)

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

        # Instruction (shown only for preset voices, e.g. "say it in an angry tone")
        self.instruct_frame = ttk.Frame(voice_frame)
        ttk.Label(self.instruct_frame, text="Instruction:").pack(side=tk.LEFT, padx=5)
        self.instruct_entry = ttk.Entry(self.instruct_frame, width=37)
        self.instruct_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

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
        
        self.text_entry = scrolledtext.ScrolledText(text_frame, height=8, width=50)
        self.text_entry.pack(fill=tk.BOTH, expand=True)

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
        
        # Generate button
        generate_button = ttk.Button(self.use_frame, text="Generate Speech", command=self.generate_speech,
                                    style="Accent.TButton")
        generate_button.pack(pady=20)

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
        """Show the Instruction box only for preset voices (generate_custom_voice's instruct)."""
        kind, _ = self.use_voice_map.get(self.use_voice_combo.get(), (None, None))
        if kind == "preset":
            self.instruct_frame.pack(fill=tk.X, pady=5)
        else:
            self.instruct_frame.pack_forget()

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
            audio_data = np.concatenate(self.audio_chunks, axis=0)
            sf.write(MIC_RECORDING_PATH, audio_data, SAMPLE_RATE)
            self.record_status_label.config(text=f"Recording saved to {MIC_RECORDING_PATH}", foreground="green")
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
        
        # Run training in a separate thread to avoid blocking UI
        thread = threading.Thread(target=self._train_voice_thread, args=(voice_name, method, device_type))
        thread.daemon = True
        thread.start()
    
    def _train_voice_thread(self, voice_name, method, device_type):
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
                ref_text = self.transcript_entry.get("1.0", tk.END).strip()
                
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
                if not os.path.exists(MIC_RECORDING_PATH):
                    self.root.after(0, lambda: messagebox.showerror("Error",
                        "Please record audio first using the recording section"))
                    return

                ref_audio = MIC_RECORDING_PATH
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
            self.root.after(0, lambda: messagebox.showinfo("Success",
                f"Voice '{voice_name}' has been trained and saved to {output_file}"))
            self.root.after(0, self.refresh_voice_lists)
            
        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            self.root.after(0, lambda: self.train_status_label.config(
                text=error_msg, foreground="red"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
    
    def generate_speech(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        output_format = self.output_format.get()
        language = self.use_language_combo.get()
        instruct = self.instruct_entry.get().strip()

        selected = self.use_voice_combo.get()
        voice = self.use_voice_map.get(selected)
        if not voice:
            messagebox.showerror("Error", "Please select a voice")
            return
        kind, key = voice

        if not text:
            messagebox.showerror("Error", "Please enter text to generate")
            return

        # Run generation in a separate thread
        thread = threading.Thread(
            target=self._generate_speech_thread,
            args=(kind, key, text, output_format, language, instruct)
        )
        thread.daemon = True
        thread.start()
    
    def _generate_speech_thread(self, kind, key, text, output_format="wav", language="Auto", instruct=""):
        try:
            self.root.after(0, lambda: self.use_status_label.config(
                text="Loading model...", foreground="blue"))

            device_type = self.use_device_type.get()
            config = self.get_model_config(device_type, show_warning=True)
            model_repo = MODEL_REPO_CUSTOM_VOICE if kind == "preset" else MODEL_REPO_BASE

            # Load model
            try:
                model = Qwen3TTSModel.from_pretrained(
                    model_repo,
                    **config
                )
            except (RuntimeError, torch.cuda.CudaError) as cuda_error:
                # Check if it's a CUDA compatibility error
                error_str = str(cuda_error).lower()
                if "cuda" in error_str and ("kernel" in error_str or "no kernel image" in error_str or "device" in error_str):
                    # CUDA error - fall back to CPU
                    device_type = "cpu"  # Update device_type for fallback
                    self.root.after(0, lambda: self.use_status_label.config(
                        text="CUDA error detected, falling back to CPU...", foreground="orange"))
                    self.root.after(0, lambda: messagebox.showwarning(
                        "CUDA Compatibility Issue",
                        f"CUDA error detected: {str(cuda_error)}\n\nFalling back to CPU mode. Generation may be slower."))
                    # Retry with CPU config
                    config = self.get_model_config("cpu", show_warning=False)
                    model = Qwen3TTSModel.from_pretrained(
                        model_repo,
                        **config
                    )
                else:
                    # Re-raise if it's not a CUDA compatibility error
                    raise

            # Custom (cloned) voices need their saved prompt loaded; presets don't —
            # generate_custom_voice() just takes the speaker name directly.
            prompt_items = None
            if kind == "custom":
                self.root.after(0, lambda: self.use_status_label.config(
                    text="Loading voice...", foreground="blue"))

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

            self.root.after(0, lambda: self.use_status_label.config(
                text="Generating speech...", foreground="blue"))

            def do_generate(m):
                if kind == "preset":
                    return m.generate_custom_voice(
                        text=text,
                        speaker=key,
                        language=language,
                        instruct=instruct or None,
                    )
                return m.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt_items,
                )

            # Generate speech
            try:
                wavs, sr = do_generate(model)
            except (RuntimeError, torch.cuda.CudaError) as cuda_error:
                # Check if it's a CUDA compatibility error during generation
                error_str = str(cuda_error).lower()
                if "cuda" in error_str and ("kernel" in error_str or "no kernel image" in error_str or "device" in error_str):
                    # CUDA error during generation - reload model on CPU and retry
                    device_type = "cpu"  # Update device_type for fallback
                    self.root.after(0, lambda: self.use_status_label.config(
                        text="CUDA error during generation, retrying on CPU...", foreground="orange"))
                    self.root.after(0, lambda: messagebox.showwarning(
                        "CUDA Compatibility Issue",
                        f"CUDA error during generation: {str(cuda_error)}\n\nReloading model on CPU and retrying. Generation may be slower."))
                    # Reload model with CPU config
                    config = self.get_model_config("cpu", show_warning=False)
                    model = Qwen3TTSModel.from_pretrained(
                        model_repo,
                        **config
                    )
                    # Retry the generation
                    wavs, sr = do_generate(model)
                else:
                    # Re-raise if it's not a CUDA compatibility error
                    raise

            # Determine save location
            save_folder = self.use_save_entry.get().strip()
            output_dir = save_folder if save_folder else OUTPUT_DIR
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"{key}_output.{output_format}")

            if output_format == "wav":
                sf.write(output_file, wavs[0], sr)
            else:
                encode_audio(wavs[0], sr, output_file, output_format)

            self.root.after(0, lambda: self.use_status_label.config(
                text=f"Speech generated successfully! Saved to {output_file}",
                foreground="green"))
            self.root.after(0, lambda: messagebox.showinfo("Success",
                f"Speech has been generated and saved to {output_file}"))

        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            self.root.after(0, lambda: self.use_status_label.config(
                text=error_msg, foreground="red"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

def main():
    root = tk.Tk()
    app = QwenTTSGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
