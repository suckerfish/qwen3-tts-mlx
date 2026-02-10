# Qwen3-TTS MLX

A Gradio web app for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech on Apple Silicon, powered by [mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Features

- **Preset voices** - 9 built-in voices with style instruction control
- **Voice design** - Create custom voices from text descriptions
- **Voice cloning** - Clone voices from short reference audio clips (M4A, MP3, WAV)
- **Model management** - Download and delete models from the UI
- **Generation history** - 5 recent outputs with waveform playback, inline rename, and delete
- **REST API** - FastAPI server for programmatic access

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.11+
- [UV](https://docs.astral.sh/uv/) for environment management

## Setup

```bash
git clone https://github.com/chadkunsman/qwen3-tts-mlx.git
cd qwen3-tts-mlx
uv sync
```

## Usage

### Gradio UI

```bash
uv run python app.py
# Opens at http://127.0.0.1:7860
```

### API Server

```bash
uv run python server.py --port 8000
# Docs at http://localhost:8000/docs
```

Options:
- `--host` - Bind address (default: `0.0.0.0`)
- `--port` - Bind port (default: `8000`)

On first use, go to the **Models** tab in the Gradio UI and download at least one model. The 1.7B-CustomVoice model is recommended for preset voices.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/voices` | List preset and saved voices |
| `GET` | `/v1/models` | List models and download status |
| `POST` | `/v1/tts/generate` | Generate with preset voice |
| `POST` | `/v1/tts/design` | Generate with voice design |
| `POST` | `/v1/tts/clone` | Generate with saved voice |

### Examples

```bash
# Health check
curl http://localhost:8000/v1/health

# List voices
curl http://localhost:8000/v1/voices

# Generate with preset voice
curl -X POST http://localhost:8000/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output hello.wav

# Generate with style instruction
curl -X POST http://localhost:8000/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "Serena (warm, gentle)", "instruct": "speak with excitement", "temperature": 0.8}' \
  --output hello_excited.wav

# Generate with voice design
curl -X POST http://localhost:8000/v1/tts/design \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "instruct": "calm professional news anchor"}' \
  --output designed.wav

# Generate with saved voice
curl -X POST http://localhost:8000/v1/tts/clone \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "my_saved_voice"}' \
  --output cloned.wav
```

## Auto-start with launchd (macOS)

Create `~/Library/LaunchAgents/com.qwen3tts.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.qwen3tts.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/.local/bin/uv</string>
        <string>run</string>
        <string>python</string>
        <string>server.py</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/YOUR_USERNAME/path/to/qwen3-tts-mlx</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/qwen3tts.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/qwen3tts.stderr.log</string>
</dict>
</plist>
```

Replace `YOUR_USERNAME` and the `WorkingDirectory` path, then:

```bash
# Load (start on boot)
launchctl load ~/Library/LaunchAgents/com.qwen3tts.server.plist

# Unload (stop auto-start)
launchctl unload ~/Library/LaunchAgents/com.qwen3tts.server.plist

# Check status
launchctl list | grep qwen3tts
```

## Models

| Model | Size | Use Case |
|-------|------|----------|
| 1.7B-CustomVoice | ~4.5 GB | Preset voices with style instructions |
| 1.7B-VoiceDesign | ~4.5 GB | Custom voice creation from descriptions |
| 1.7B-Base | ~4.5 GB | Voice cloning (higher quality) |
| 0.6B-Base | ~1.5 GB | Voice cloning (faster, smaller) |

Models are downloaded to `~/.cache/huggingface/hub/` and can be managed from the Models tab.

## Available Voices

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

## Voice Cloning

1. Go to the **Clone Voice** tab and upload a reference audio clip with its transcript
2. Save the voice with a name
3. Switch to the **Voice Cloning** tab to generate speech using the saved voice

Saved voices are stored in `saved_voices/`.

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
