# TODO: qwen-tts feature gaps

A "Configure" tab now exists for settings you'd set once rather than per run (currently
just device/CUDA selection — disabled when CUDA isn't available, so you can't pick it
and get repeated "falling back to CPU" warnings). Model size and dtype below belong
there when implemented.

## Multi-voice generation (`[VoiceName]` markers)

- [x] `[VoiceName]` placeholder switches voice from that point forward; text is split
      into per-voice segments, generated, and stitched into one output file with a
      short silence gap at each voice change
- [x] Mixing preset and custom-cloned voices in the same generation is allowed — each
      kind's segments are batched into a single `generate_custom_voice` /
      `generate_voice_clone` call, loading each of Base/CustomVoice at most once
- [x] Unresolved `[Name]` markers block generate entirely, listing the bad names
- [x] Typing `[` in the text box pops up a filtered, keyboard-navigable voice picker
      (Enter/Tab/click inserts `Name]`, cursor landing just before the `]` so
      `:instruction` can follow; Escape/click-away/losing focus erases the abandoned
      `[partial` text; typing `]` or `:` by hand leaves a manually-typed marker as-is)
- [x] `[VoiceName:instruction]` sets a per-segment style instruction inline (superseded
      the standalone Instruction text box — see below); an instruction on a voice-clone
      segment (no instruct concept) warns rather than blocking
- [x] There's no Voice/Instruction dropdown or text box left on the Use Voice tab at
      all — every generation is entirely described by the text box's markers. Language
      stays a separate, single dropdown (applies to every segment; nothing so far has
      asked for per-segment language)

## CustomVoice model (preset voices)

Landed together with a broader voice-picker redesign, since superseded by the
`[VoiceName]` marker system above: Train Voice still has a "Voice" dropdown
("New..." plus local custom voices, no presets — they aren't trainable) for
training, but Use Voice has no dropdown of its own — voices are chosen via markers,
resolved against the local custom-voice cache plus the 9 presets. A `[PresetName]`
marker generates via the CustomVoice checkpoint instead of Base.

- [x] Add a way to select the `CustomVoice` checkpoint (via a `[PresetName]` marker,
      resolved to `MODEL_REPO_CUSTOM_VOICE`)
- [x] Add a "Speaker" dropdown populated from `model.get_supported_speakers()` — actually
      hardcoded (`PRESET_VOICES`) rather than fetched at runtime, see decision below
      (9 presets: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)
- [x] Add optional style instruction (e.g. "say it in an angry tone") — now the
      `[PresetName:instruction]` marker syntax rather than a standalone text box
- [x] Wire up to `model.generate_custom_voice(text, speaker, language, instruct=...)`

Note: the speaker list is hardcoded, not fetched via `model.get_supported_speakers()` at
runtime — that would require loading the CustomVoice checkpoint (or at least its config
via `AutoConfig.from_pretrained`, network/HF-cache dependent) just to populate a dropdown
before the user has done anything. The 9 names are fixed for this released checkpoint.

## VoiceDesign model

- [ ] Add a way to select the `VoiceDesign` checkpoint
- [ ] Add a "Voice Design Instruction" text box (natural-language voice description,
      e.g. "warm elderly British female voice")
- [ ] Wire up to `model.generate_voice_design(text, instruct, language)`

## Language selection

- [x] Replace hardcoded `language="English"` with a dropdown (Use Voice tab, applies to
      both custom-voice cloning and preset generation)
- [x] Populate from the known supported languages (`SUPPORTED_LANGUAGES` — hardcoded for
      the same reason as the preset list above, not fetched via
      `model.get_supported_languages()`) — Auto, Chinese, English, Japanese, Korean,
      German, French, Russian, Portuguese, Spanish, Italian

## Voice-clone mode (x_vector_only_mode)

- [ ] Expose a checkbox for `x_vector_only_mode` in the Train Voice tab (currently
      hardcoded to `False` at [qwen_tts_gui.py:971](src/qwen_tts_gui.py#L971))
- [ ] When checked: skip requiring a reference transcript, using speaker-embedding-only
      cloning (faster, lower fidelity)

## Generation / sampling controls

- [ ] Expose (likely as an "Advanced" collapsible section, defaults untouched):
  - [ ] `temperature`
  - [ ] `top_k` / `top_p`
  - [ ] `repetition_penalty`
  - [ ] `max_new_tokens`
  - [ ] `subtalker_*` variants (tokenizer-v2-specific — check whether relevant model
        checkpoints actually use tokenizer v2 before bothering)

## Model size (Configure tab)

- [ ] Offer `0.6B` variants alongside `1.7B` for each model type (faster/lighter,
      same three model types: Base / CustomVoice / VoiceDesign)

## dtype (Configure tab)

- [ ] Add `float16` as a third option alongside the current `bfloat16` (CUDA) /
      `float32` (CPU)

## Batch generation

- [ ] Consider a "generate multiple lines at once" mode — the underlying API already
      accepts lists for `text`/`ref_audio`/`language`/etc. on all three generate methods

## Reference audio input flexibility (low priority)

- [ ] `ref_audio` also accepts a URL or base64 string, not just a local file path —
      probably not worth UI space, but note in case a use case comes up

## Not a gap — don't implement

- `non_streaming_mode`: despite the name, the library's own docstring says this only
  simulates streaming text input internally; it is not real-time audio streaming.
  No user-facing capability is being missed here.
