"""Async HTTP client for the Qwen3-TTS FastAPI backend."""

import os
import re

import aiohttp


class TTSClientError(Exception):
    """Raised when a TTS API request fails."""


class TTSClient:
    """Thin async wrapper around the TTS REST API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url
            or os.environ.get("TTS_API_URL", "http://host.docker.internal:8000")
        ).rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request_json(self, method: str, path: str, **kwargs) -> dict:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status == 422:
                    body = await resp.json()
                    detail = body.get("detail", body)
                    raise TTSClientError(f"Validation error: {detail}")
                if resp.status == 503:
                    body = await resp.json()
                    raise TTSClientError(
                        f"Service unavailable: {body.get('detail', body)}"
                    )
                if resp.status >= 400:
                    text = await resp.text()
                    raise TTSClientError(
                        f"API error {resp.status}: {text}"
                    )
                return await resp.json()
        except aiohttp.ClientConnectorError:
            raise TTSClientError(
                f"Cannot connect to TTS server at {self.base_url}. "
                "Is the FastAPI server running?"
            )

    async def _request_binary(
        self, method: str, path: str, **kwargs
    ) -> tuple[bytes, str]:
        """Return (wav_bytes, filename) from a binary response."""
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        try:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status == 422:
                    body = await resp.json()
                    detail = body.get("detail", body)
                    raise TTSClientError(f"Validation error: {detail}")
                if resp.status == 503:
                    body = await resp.json()
                    raise TTSClientError(
                        f"Service unavailable: {body.get('detail', body)}"
                    )
                if resp.status >= 400:
                    text = await resp.text()
                    raise TTSClientError(
                        f"API error {resp.status}: {text}"
                    )

                wav_bytes = await resp.read()

                # Parse filename from Content-Disposition header
                filename = "output.wav"
                cd = resp.headers.get("Content-Disposition", "")
                match = re.search(r'filename="?([^";\s]+)"?', cd)
                if match:
                    filename = match.group(1)

                return wav_bytes, filename
        except aiohttp.ClientConnectorError:
            raise TTSClientError(
                f"Cannot connect to TTS server at {self.base_url}. "
                "Is the FastAPI server running?"
            )

    # ------------------------------------------------------------------
    # JSON endpoints
    # ------------------------------------------------------------------

    async def health_check(self) -> dict:
        return await self._request_json("GET", "/v1/health")

    async def list_voices(self) -> dict:
        return await self._request_json("GET", "/v1/voices")

    async def list_models(self) -> dict:
        return await self._request_json("GET", "/v1/models")

    # ------------------------------------------------------------------
    # Binary (WAV) endpoints
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        voice: str = "Ryan",
        instruct: str = "",
        temperature: float = 1.0,
        model: str | None = None,
    ) -> tuple[bytes, str]:
        payload: dict = {
            "text": text,
            "voice": voice,
            "instruct": instruct,
            "temperature": temperature,
        }
        if model is not None:
            payload["model"] = model
        return await self._request_binary("POST", "/v1/tts/generate", json=payload)

    async def design_voice_speech(
        self,
        text: str,
        instruct: str,
        language: str = "Auto",
        temperature: float = 0.9,
    ) -> tuple[bytes, str]:
        payload = {
            "text": text,
            "instruct": instruct,
            "language": language,
            "temperature": temperature,
        }
        return await self._request_binary("POST", "/v1/tts/design", json=payload)

    async def clone_voice_speech(
        self,
        text: str,
        voice: str,
        temperature: float = 1.0,
        model: str | None = None,
    ) -> tuple[bytes, str]:
        payload: dict = {
            "text": text,
            "voice": voice,
            "temperature": temperature,
        }
        if model is not None:
            payload["model"] = model
        return await self._request_binary("POST", "/v1/tts/clone", json=payload)
