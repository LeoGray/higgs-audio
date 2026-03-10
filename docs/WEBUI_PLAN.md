# Higgs Audio Web UI Plan

## Goal

Provide a browser-based interface for local Higgs Audio generation so users do not need to work directly with the CLI.

The first version should optimize for:

- local single-user usage
- low implementation risk
- reuse of existing inference code
- support for the main flows already covered by `examples/generation.py`

## Recommended Direction

Use a Python-first stack for the first Web UI:

- UI: Gradio Blocks
- Service layer: a new reusable Python module extracted from `examples/generation.py`
- Inference layer: existing model loading and generation code
- Optional future API layer: FastAPI if remote access or multi-client support becomes necessary

Why this direction:

- the repository is already fully Python-based
- Gradio handles text inputs, file uploads, audio playback, and progress UI well
- it avoids adding a Node/Vite/React toolchain before the product surface is stable
- it can later be replaced or wrapped without losing backend logic

## Current Reusable Pieces

The repository already has most of the hard backend pieces:

- `boson_multimodal/serve/serve_engine.py`
  - reusable inference engine
  - supports synchronous generation
  - has async delta streaming primitives for future upgrades
- `examples/generation.py`
  - current user-facing generation workflow
  - transcript normalization
  - chunking logic
  - voice preset handling
  - scene prompt and multi-speaker preparation
- `examples/voice_prompts/`
  - built-in voice presets
- `examples/scene_prompts/`
  - built-in scene presets
- `examples/transcript/`
  - built-in input examples
- `examples/vllm/`
  - future option if the UI should talk to a remote OpenAI-compatible backend instead of local model code

## Product Scope

### MVP

The first Web UI should cover four tasks:

- smart voice generation from text
- voice cloning with a preset voice or uploaded reference audio
- multi-speaker generation from tagged transcript
- playback and download of the generated wav file

### Nice-to-have After MVP

- generation history
- preset transcript browser
- streaming status updates
- cancel running generation
- switch between local backend and vLLM backend

## Suggested User Experience

### Main Page

Single-page layout with three zones:

- left: input form
- center: generation progress and logs
- right: result audio player and recent outputs

### Input Form

Core fields:

- transcript text area
- transcript file upload
- task mode selector
  - smart voice
  - voice clone
  - multi-speaker
- scene prompt selector
  - preset
  - custom text
  - empty
- voice source
  - built-in preset
  - multiple presets
  - uploaded reference audio
- advanced settings accordion
  - temperature
  - top k
  - top p
  - max new tokens
  - seed
  - chunk method
  - chunk sizes
  - device
  - static kv cache toggle

### Output Area

- generation status
- generation time
- normalized transcript preview
- audio player
- download button
- generated text output if returned
- save path or output id

### Presets Area

- built-in example transcripts
- built-in scene prompts
- built-in voice presets
- one-click "load example" actions

## Backend Shape

### Step 1: Extract a Service Layer

Do not let the Gradio app call `examples/generation.py` directly.

Instead, move the reusable parts into a new module, for example:

- `boson_multimodal/webui/generation_service.py`

This module should own:

- model initialization and singleton caching
- transcript normalization
- transcript chunking
- scene prompt loading
- voice preset loading
- generation request validation
- output file creation

This is the most important refactor because the current CLI logic is concentrated in an example script rather than a reusable package module.

### Suggested Internal API

Define a typed request object and a typed response object.

Request fields:

- transcript_text
- transcript_file_path or uploaded file
- scene_prompt_text
- scene_prompt_preset
- ref_audio_presets
- ref_audio_uploads
- ref_audio_in_system_message
- chunk_method
- chunk_max_word_num
- chunk_max_num_turns
- temperature
- top_k
- top_p
- max_new_tokens
- ras_win_len
- ras_win_max_num_repeat
- seed
- device
- use_static_kv_cache

Response fields:

- output_audio_path
- sampling_rate
- generated_text
- normalized_transcript
- elapsed_seconds
- warnings

### Output Storage

Store outputs in a dedicated folder, for example:

- `outputs/webui/<timestamp>_<short_id>/`

Artifacts:

- `output.wav`
- `request.json`
- `result.json`

This keeps runs reproducible and makes a later history page trivial.

## UI-to-Backend Mapping

### MVP Without HTTP API

The Gradio app can call the service layer in-process.

This is the fastest path and the best fit for local usage.

### Future HTTP API

If the project later needs browser clients separate from the Python process, add FastAPI routes:

- `GET /api/presets/voices`
- `GET /api/presets/scenes`
- `GET /api/presets/transcripts`
- `POST /api/generate`
- `GET /api/jobs/{id}`
- `GET /api/jobs/{id}/audio`
- `POST /api/jobs/{id}/cancel`

For long generations, job-based APIs are safer than blocking one huge request.

## Recommended File Layout

One possible layout:

- `boson_multimodal/webui/generation_service.py`
- `boson_multimodal/webui/presets.py`
- `boson_multimodal/webui/schemas.py`
- `apps/gradio_app.py`
- `outputs/webui/`

If FastAPI is added later:

- `apps/api_app.py`

## Implementation Phases

### Phase 1: Refactor for Reuse

- extract helper functions from `examples/generation.py`
- wrap model loading in a reusable singleton or cached factory
- expose one clean `generate()` service method
- preserve parity with current CLI behavior

Exit criteria:

- current CLI and new service layer produce equivalent outputs for the same input

### Phase 2: Build the MVP Web UI

- add Gradio app with one main generation page
- support text input, preset selection, file upload, and audio playback
- save outputs under `outputs/webui`
- surface errors clearly in the UI

Exit criteria:

- a user can generate speech and download the wav without touching CLI flags

### Phase 3: Improve Interaction

- add preset browser
- add recent output history
- add progress messages
- add simple input validation before generation starts

Exit criteria:

- common mistakes are caught before the model runs

### Phase 4: Optional Serving Layer

- add FastAPI endpoints
- support queued jobs and cancellation
- optionally point the UI at local inference or vLLM

Exit criteria:

- UI and backend can run as separate processes

## Key Risks

### Model Load Time

Initial startup will be slow because model and tokenizer loading is heavy.

Mitigation:

- load once and reuse
- show a clear "model loading" state in the UI

### GPU Memory and Concurrency

This project is not ready for many concurrent generations in one process.

Mitigation:

- start with one active job at a time
- disable concurrent generations in MVP

### Long-Running Requests

Generation can take noticeable time for long transcripts.

Mitigation:

- show progress states
- save outputs incrementally
- add cancel support later

### Feature Drift Between CLI and UI

If logic stays split between the CLI script and UI code, the two flows will diverge.

Mitigation:

- move generation preparation into a shared module first

## What I Would Build First

If the goal is to get a usable Web UI quickly, the order should be:

1. Extract shared generation logic from `examples/generation.py`.
2. Add a Gradio app with smart voice, voice clone, and multi-speaker tabs.
3. Save every run to `outputs/webui`.
4. Only add FastAPI or a JS frontend if local Gradio proves too limiting.

## Non-Goal for First Version

Do not optimize for:

- multi-user auth
- cloud deployment
- distributed job scheduling
- React-based visual polish

Those are reasonable later, but they would slow down the first useful release.
