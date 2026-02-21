"""FastMCP server exposing Qwen3-TTS generation tools."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .tts_client import TTSClient, TTSClientError

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qwen3-tts-mcp")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

mcp = FastMCP(
    "Qwen3-TTS",
    instructions=(
        "Text-to-speech server powered by Qwen3-TTS. "
        "Use health_check to verify the backend is running, "
        "list_voices / list_models to discover options, "
        "then generate_speech, design_voice_speech, or clone_voice_speech "
        "to produce audio. Audio tools return a URL where the generated "
        "WAV file can be downloaded."
    ),
)

client = TTSClient()


def _save_and_url(wav_bytes: bytes, filename: str) -> str:
    """Save WAV to output dir and return the public URL."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(wav_bytes)
    logger.info("Saved %s (%d bytes)", out_path, len(wav_bytes))
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL}/files/{filename}"
    return f"/files/{filename}"


# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------


@mcp.tool()
async def health_check() -> dict:
    """Check whether the TTS backend server is reachable and healthy."""
    try:
        return await client.health_check()
    except TTSClientError as exc:
        return {"status": "error", "detail": str(exc)}


@mcp.tool()
async def list_voices() -> dict:
    """List available preset and saved voices.

    Returns a dict with 'preset' (built-in voice names) and 'saved'
    (user-recorded voice names for cloning).
    """
    try:
        return await client.list_voices()
    except TTSClientError as exc:
        raise ToolError(str(exc))


@mcp.tool()
async def list_models() -> dict:
    """List available TTS models and their download status.

    Returns a dict with 'models' containing name, loaded state, and size
    for each model.
    """
    try:
        return await client.list_models()
    except TTSClientError as exc:
        raise ToolError(str(exc))


@mcp.tool()
async def generate_speech(
    text: str,
    voice: str = "Ryan",
    instruct: str = "",
    temperature: float = 1.0,
    model: str | None = None,
) -> dict:
    """Generate speech audio from text using a preset voice.

    Args:
        text: The text to speak.
        voice: Preset voice name (use list_voices to see options).
        instruct: Optional instruction to control speaking style,
                  e.g. "speak slowly and softly".
        temperature: Sampling temperature (0.0-2.0). Higher = more varied.
        model: Model name (use list_models to see options). Defaults to first available.

    Returns a dict with 'url' pointing to the generated WAV file.
    """
    try:
        wav_bytes, filename = await client.generate_speech(
            text=text,
            voice=voice,
            instruct=instruct,
            temperature=temperature,
            model=model,
        )
    except TTSClientError as exc:
        raise ToolError(str(exc))

    try:
        url = _save_and_url(wav_bytes, filename)
    except OSError as exc:
        raise ToolError(f"Failed to save audio: {exc}")

    return {
        "url": url,
        "filename": filename,
        "voice": voice,
        "temperature": temperature,
        "model": model,
    }


@mcp.tool()
async def design_voice_speech(
    text: str,
    instruct: str,
    language: str = "Auto",
    temperature: float = 0.9,
) -> dict:
    """Generate speech with a designed (AI-created) voice.

    Instead of picking a preset voice, describe the voice characteristics
    in the instruct field and the model will create a matching voice.

    Args:
        text: The text to speak.
        instruct: Description of the desired voice, e.g.
                  "a deep male voice with a British accent".
        language: Language code or "Auto" for automatic detection.
        temperature: Sampling temperature (0.0-2.0).

    Returns a dict with 'url' pointing to the generated WAV file.
    """
    try:
        wav_bytes, filename = await client.design_voice_speech(
            text=text,
            instruct=instruct,
            language=language,
            temperature=temperature,
        )
    except TTSClientError as exc:
        raise ToolError(str(exc))

    try:
        url = _save_and_url(wav_bytes, filename)
    except OSError as exc:
        raise ToolError(f"Failed to save audio: {exc}")

    return {
        "url": url,
        "filename": filename,
        "instruct": instruct,
        "language": language,
        "temperature": temperature,
    }


@mcp.tool()
async def clone_voice_speech(
    text: str,
    voice: str,
    temperature: float = 1.0,
    model: str | None = None,
) -> dict:
    """Generate speech using a previously saved (cloned) voice.

    The voice must be one of the saved voices -- use list_voices to see
    what's available under the 'saved' key.

    Args:
        text: The text to speak.
        voice: Name of a saved voice (from list_voices 'saved' list).
        temperature: Sampling temperature (0.0-2.0).
        model: Model name (use list_models to see options). Defaults to first available.

    Returns a dict with 'url' pointing to the generated WAV file.
    """
    try:
        wav_bytes, filename = await client.clone_voice_speech(
            text=text,
            voice=voice,
            temperature=temperature,
            model=model,
        )
    except TTSClientError as exc:
        raise ToolError(str(exc))

    try:
        url = _save_and_url(wav_bytes, filename)
    except OSError as exc:
        raise ToolError(f"Failed to save audio: {exc}")

    return {
        "url": url,
        "filename": filename,
        "voice": voice,
        "temperature": temperature,
        "model": model,
    }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
        help="MCP transport (default: streamable-http)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    args = parser.parse_args()

    # Ensure output dir exists at startup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.transport in ("streamable-http", "sse"):
        import uvicorn
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route
        from starlette.staticfiles import StaticFiles

        # Get the MCP ASGI app
        mcp_app = mcp.http_app(transport=args.transport)

        async def health(request):
            return JSONResponse({"status": "ok"})

        # Combine MCP + static file serving + health endpoint
        # lifespan must be passed through for FastMCP's session manager
        app = Starlette(
            routes=[
                Route("/health", health),
                Mount("/files", app=StaticFiles(directory=str(OUTPUT_DIR)), name="files"),
                Mount("/", app=mcp_app),
            ],
            lifespan=mcp_app.lifespan,
        )

        uvicorn.run(app, host=args.host, port=args.port)
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
