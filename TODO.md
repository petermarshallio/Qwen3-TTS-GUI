# TODO: qwen-tts feature gaps

Gap analysis between what `qwen_tts` (0.1.1) exposes and what `src/qwen_tts_gui.py`
currently uses. Local tracking only, not committed. Ordered roughly by bang-for-buck.

## CustomVoice model (preset voices)

Landed together with a broader voice-picker redesign: Train Voice now has a "Voice"
dropdown ("New..." + local custom voices, no presets — they aren't trainable), and Use
Voice's dropdown lists custom voices + the 9 presets (suffixed `(Preset)`). Selecting a
preset loads the CustomVoice checkpoint instead of Base.

- [x] Add a way to select the `CustomVoice` checkpoint (via the Use Voice dropdown,
      resolved to `MODEL_REPO_CUSTOM_VOICE` when a preset is selected)
- [x] Add a "Speaker" dropdown populated from `model.get_supported_speakers()` — actually
      hardcoded (`PRESET_VOICES`) rather than fetched at runtime, see decision below
      (9 presets: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee)
- [x] Add optional style "Instruction" text box (e.g. "say it in an angry tone") — shown
      only when a preset is selected
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
      hardcoded to `False` at [qwen_tts_gui.py:458,481](src/qwen_tts_gui.py#L458))
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

## Model size

- [ ] Offer `0.6B` variants alongside `1.7B` for each model type (faster/lighter,
      same three model types: Base / CustomVoice / VoiceDesign)

## dtype

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
