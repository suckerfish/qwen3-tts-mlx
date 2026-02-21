# Qwen3-TTS MLX

A Gradio web app for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech on Apple Silicon, powered by [mlx-audio](https://github.com/Blaizzy/mlx-audio).

## Features

- **Preset voices** - 9 built-in voices with style instruction control
- **Voice design** - Create custom voices from text descriptions
- **Voice cloning** - Clone voices from short reference audio clips (M4A, MP3, WAV)
- **Model management** - Download and delete models from the UI
- **Generation history** - 5 recent outputs with waveform playback, inline rename, and delete
- **REST API** - FastAPI server for programmatic access
- **MCP server** - Docker-based [MCP](https://modelcontextprotocol.io/) server for Claude Desktop and other MCP clients

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

## MCP Server

The MCP server runs in Docker and proxies requests to the FastAPI backend, letting MCP clients (Claude Desktop, mcporter, etc.) generate speech via tool calls. Generated audio is served over HTTP — tools return a download URL rather than inline audio.

### Architecture

```
MCP Client (Claude Desktop, mcporter, etc.)
    ↓ streamable-http
Docker Container (MCP Server :8080)
    ↓ HTTP → TTS_API_URL
macOS Host (FastAPI server.py :8000)
    ↓ MLX inference on Metal GPU
WAV audio returned as download URL
```

### Quick Start

The MCP server requires the FastAPI backend to be running on the host.

```bash
# 1. Start the TTS backend
uv run python server.py

# 2. Start the MCP server
docker compose up -d

# 3. Verify
curl http://localhost:8080/health
```

### Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_API_URL` | `http://host.docker.internal:8000` | URL of the FastAPI TTS backend |
| `PUBLIC_BASE_URL` | `http://localhost:8080` | Base URL for audio download links |
| `MCP_PORT` | `8080` | Host port mapped to the container |
| `LOG_LEVEL` | `INFO` | Logging level |

On Linux Docker (no `host.docker.internal`), set `TTS_API_URL` to the host's IP (e.g. `http://172.17.0.1:8000` or a Tailscale IP).

### MCP Tools

| Tool | Description | Returns |
|------|-------------|---------|
| `health_check` | Check if TTS backend is reachable | Status dict |
| `list_voices` | List preset and saved voices | `{preset: [...], saved: [...]}` |
| `list_models` | List models and download status | `{models: [...]}` |
| `generate_speech` | Generate with a preset voice | `{url, filename, voice, ...}` |
| `design_voice_speech` | Generate with an AI-designed voice | `{url, filename, instruct, ...}` |
| `clone_voice_speech` | Generate with a saved/cloned voice | `{url, filename, voice, ...}` |

Audio files are saved to `./outputs/` (bind-mounted) and served at `{PUBLIC_BASE_URL}/files/{filename}`.

### Building Locally

```bash
docker build -t ghcr.io/suckerfish/qwen3-tts-mcp:latest ./mcp
```

Pre-built multi-arch images (amd64 + arm64) are published to GHCR on every push to `master`.

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

## Known Issues

- **"model of type qwen3_tts to instantiate a model of type" / "incorrect regex pattern"** — Harmless warnings from `transformers` about the tokenizer bundled with the mlx-community model weights. These come from upstream and don't affect generation.
- **"PySoundFile failed. Trying audioread instead."** — Librosa falls back to audioread when loading non-WAV reference audio (M4A, MP3). Generation still works; the warning is cosmetic.

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
