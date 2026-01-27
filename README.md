# Qwen3-TTS Local Setup

## Project Location
```
/Users/chad.kunsman/Documents/PythonProject/qwen3_tts/
```

## Environment
- Python 3.11 via UV
- Virtual env: `.venv/`
- Activate: `source .venv/bin/activate`

## Models Downloaded (cached in `~/.cache/huggingface/hub/`)

| Model | Size | Use Case |
|-------|------|----------|
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` | ~1.5 GB | Voice cloning (has silence bug) |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | ~4.5 GB | Voice cloning (has silence bug) |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | ~4.5 GB | Preset voices (has silence bug) |

> **Warning:** All models have a probabilistic silence bug in the mlx-audio port. See Known Issues below.

## What Works

### Voice Cloning (0.6B-Base) - BEST OPTION
```bash
source .venv/bin/activate && python -m mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 \
  --text "Your text here" \
  --ref_audio /path/to/reference.m4a \
  --ref_text "Transcript of reference audio" \
  --output_path ./output \
  --file_prefix my_clip
```

**Key findings:**
- Reference audio energy level affects output energy
- Use energetic reference for energetic output
- **Note:** 0.6B also has the silence bug (see Known Issues)

### Preset Voices with Instructions (1.7B-CustomVoice)
```bash
source .venv/bin/activate && python -m mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
  --text "Your text here" \
  --voice Ryan \
  --instruct "Instructional, like teaching a process" \
  --temperature 0.7 \
  --output_path ./output \
  --file_prefix my_clip
```

> **Warning:** Has 25-50% chance of producing silent output. Retry if needed. (See Known Issues)

**Available voices:** Ryan, Aiden, Vivian, Serena, Dylan, Eric, Uncle_Fu, Ono_Anna, Sohee
- Ryan/Aiden: Best for English
- Dylan/Eric: Chinese only (Beijing/Sichuan dialects)

## Known Issues

### Silence Gaps (All Models) - mlx-audio Port Bug

All Qwen3-TTS models in mlx-audio sometimes produce 96-second silent output files. This is a **probabilistic bug in the MLX port** - not present in the official PyTorch implementation.

- **Affects:** 0.6B-Base, 1.7B-Base, 1.7B-CustomVoice
- **Cause:** Token sampling gets "stuck" generating silence tokens
- **Status:** Open issue, maintainer investigating (Issue #395, reported Jan 23, 2026)

#### Consistency Test Results (8 trials per temperature)

| Temperature | Failure Rate | Notes |
|-------------|--------------|-------|
| 0.5 | 25% (2/8) | |
| 0.7 | 50% (4/8) | |
| 0.85 | 25% (2/8) | |
| 0.9 | 50% (4/8) | Default temp |

**Key Finding:** Temperature does NOT reliably prevent silence issues. All temperatures tested show 25-50% failure rates.

#### Quick Detection
- If output file is ~4.6 MB or duration is 96 seconds → **BROKEN**
- Working files are typically 0.7-1.1 MB for ~15-22 sec speech

#### Workaround: Retry on Failure
```bash
# Generate and check if broken (duration = 96 sec means failure)
# If broken, regenerate - each attempt has 50-75% success rate
```

#### Recommended Approach
1. **Implement retry logic** - each attempt has 50-75% success rate
2. **Check file size/duration** after generation to detect failures
3. **Wait for fix** - watch Issue #395 for updates
4. **Alternative:** Use official PyTorch implementation (requires NVIDIA GPU)

#### GitHub Status
- **Issue #395** - Open, maintainer investigating (same problem)
- **PR #68** fixed similar issue in Orpheus by adjusting sampling parameters
- See: https://github.com/Blaizzy/mlx-audio/issues/395

See `SILENCE_INVESTIGATION.md` for full test data.

### Flags That DON'T Work
- `--speed` - Not implemented for Qwen3-TTS in mlx-audio
- `--instruct` with `--ref_audio` - Cannot combine (mutually exclusive)
- `--pitch`, `--exaggeration` - Not implemented

## Reference Audio Files

| File | Description |
|------|-------------|
| `ref_voice.m4a` | Original calm reference (6.6 sec) |
| `ref_voice_energetic.m4a` | Energetic reference (10.7 sec) - RECOMMENDED |

**Transcripts:**
- Calm: "This is chad. this is my voice. I am talking to you right now. Thanks for listening"
- Energetic: "Okay, we're about to try something new. I'm really excited to try this out, and I hope it works out well for all of us here. Thank you so much for trying this out, I appreciate it so much."

## Voice Memos Location (macOS)
```
~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings/
```

## Transcription with Whisper
```bash
source .venv/bin/activate && python -m mlx_audio.stt.generate \
  --model mlx-community/whisper-large-v3-turbo \
  --audio /path/to/audio.m4a \
  --output-path ./transcript.txt \
  --format txt \
  --language en
```

## Web App (Gradio)

A full-featured Gradio web interface for TTS generation.

### Run the App
```bash
uv run python app.py
# Opens at http://127.0.0.1:7860
```

### Features
- **Two-column layout**: Controls on left, output waveforms on right
- **Preset Voices tab**: 9 built-in voices with descriptions and style presets
- **Voice Cloning tab**: Clone voices from reference audio (supports M4A, MP3, WAV)
- **Generation history**: Up to 5 recent outputs with waveform playback
- **Inline management**: Edit filenames directly, delete with confirmation
- **Metadata display**: Voice, temperature, and instruction shown in audio label
- **Auto-save**: All generations saved to `outputs/` folder

### Available Voices (Preset Tab)
| Voice | Description | Best For |
|-------|-------------|----------|
| Ryan | Dynamic, strong rhythm | English |
| Aiden | Sunny, clear midrange | English |
| Vivian | Bright, slightly edgy | Chinese |
| Serena | Warm, gentle | Chinese |
| Dylan | Youthful Beijing | Chinese (Beijing) |
| Eric | Lively, husky | Chinese (Sichuan) |
| Uncle_Fu | Seasoned, low mellow | Chinese |
| Ono_Anna | Playful | Japanese |
| Sohee | Warm | Korean |

### Style Instruction Tips
- Use specific delivery instructions: `speak with excitement and enthusiasm`
- Control pace: `slow deliberate pace with dramatic pauses`
- Articulation: `steady speed, clear articulation`
- Special effects: `whispered, secretive tone`
- Prevent unwanted sounds: `happy but without laughing`

### Voice Cloning
- **1.7B-Base** (default): Better quality, larger model
- **0.6B-Base**: Faster, smaller model
- Supports M4A, MP3, WAV audio files via librosa
- Save voices for reuse - stored in `saved_voices/`
- Create new voices via the "Create New Voice" accordion

### Output Management
- Edit filename textbox directly to rename files on disk
- Click ⏮ to rewind audio to beginning
- Click 🗑️ to delete (confirms deletion from disk)
- Files saved as `{voice}_{timestamp}.wav` or `clone_{timestamp}.wav`

## Project Structure
```
qwen3_tts/
├── .venv/                      # Virtual environment
├── app.py                      # Gradio web app
├── outputs/                    # Generated audio files (auto-created)
├── saved_voices/               # Saved cloned voices (auto-created)
├── reference_voices/           # Reference audio for cloning
├── analyze_audio.py            # Audio file analyzer (detect broken files)
├── test_silence.py             # Temperature sweep test runner
├── test_tts.py                 # TTS generation tests
├── SILENCE_INVESTIGATION.md    # Full investigation results
├── README.md                   # This file
└── main.py                     # (unused placeholder)
```

## Quick Commands

### Clone your voice (energetic)
```bash
source .venv/bin/activate && python -m mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16 \
  --text "Your text here" \
  --ref_audio ./ref_voice_energetic.m4a \
  --ref_text "Okay, we're about to try something new. I'm really excited to try this out, and I hope it works out well for all of us here. Thank you so much for trying this out, I appreciate it so much." \
  --output_path ./output \
  --file_prefix clone
```

### Use Ryan voice with instructions
```bash
source .venv/bin/activate && python -m mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
  --text "Your text here" \
  --voice Ryan \
  --instruct "Your style instruction here" \
  --temperature 0.7 \
  --output_path ./output \
  --file_prefix ryan
```

### Play generated audio
```bash
afplay ./output/clone_000.wav
```

## Supported Languages
Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

Use `--lang_code` flag: `en`, `zh`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`
