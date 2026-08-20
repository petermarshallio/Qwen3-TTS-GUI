# DONE: qwen-tts feature gaps

## 2026-08-20 — qwen-tts feature gaps closed out

Multi-voice `[VoiceName]` / `[VoiceName:instruction]` markers replaced the old
Voice/Instruction dropdown and text box entirely: per-segment style instructions,
mixed preset + voice-clone generation batched by model kind, keyboard-navigable
autocomplete on `[`, and blocking validation for unknown voices. VoiceDesign landed
as a third Train Voice method (design a voice from a natural-language description,
then clone-prompt the synthesized reference exactly like a real recording — no
separate "designed voice" concept downstream). `x_vector_only_mode` became a "Quick
clone" checkbox on Train Voice. A collapsed Advanced section on Use Voice exposes
temperature/top_k/top_p/repetition_penalty/max_new_tokens/subtalker_* (confirmed
relevant — every model here uses the 12Hz "tokenizer v2"). A three-section Configure
tab (Global device; Train and Generate each with independent model size and dtype,
dtype disabled on CPU) replaced the old scattered per-tab device pickers. Declined:
batch generation (multi-voice generation already batches under the hood) and
`ref_audio` URL/base64 support (no use case, not worth UI space).
