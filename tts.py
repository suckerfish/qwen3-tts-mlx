"""Shared TTS generation logic for Qwen3-TTS."""

import threading
from datetime import datetime
from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf
from huggingface_hub import scan_cache_dir
from mlx_audio.tts.utils import load_model

VOICES = [
    "Ryan (dynamic, strong rhythm)",
    "Aiden (sunny, clear midrange)",
    "Vivian (bright, slightly edgy)",
    "Serena (warm, gentle)",
    "Dylan (youthful Beijing)",
    "Eric (lively Chengdu, husky)",
    "Uncle_Fu (seasoned, low mellow)",
    "Ono_Anna (playful Japanese)",
    "Sohee (warm Korean)",
]

ALL_MODELS = {
    "1.7B-CustomVoice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "1.7B-VoiceDesign": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "1.7B-Base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "0.6B-Base": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
}

PRESET_MODELS = {"1.7B-CustomVoice": ALL_MODELS["1.7B-CustomVoice"]}
DESIGN_MODELS = {"1.7B-VoiceDesign": ALL_MODELS["1.7B-VoiceDesign"]}
CLONE_MODELS = {k: v for k, v in ALL_MODELS.items() if "Base" in k}

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

OUTPUTS_DIR = Path("outputs")
SAVED_VOICES_DIR = Path("saved_voices")
OUTPUTS_DIR.mkdir(exist_ok=True)
SAVED_VOICES_DIR.mkdir(exist_ok=True)

_models: dict = {}
_model_lock = threading.Lock()


def get_model(model_path: str):
    """Lazy load models with thread safety."""
    with _model_lock:
        if model_path not in _models:
            _models[model_path] = load_model(model_path)
        return _models[model_path]


def get_cached_models() -> dict[str, tuple[bool, int]]:
    """Return dict of {repo_id: (is_cached, size_bytes)}."""
    cache_info = scan_cache_dir()
    cached = {}
    for repo in cache_info.repos:
        cached[repo.repo_id] = (True, repo.size_on_disk)
    return cached


def is_model_downloaded(repo_id: str) -> bool:
    """Check if a specific model is in cache."""
    cached = get_cached_models()
    return repo_id in cached


def get_model_status() -> list[dict]:
    """Get status of all models."""
    cached = get_cached_models()
    statuses = []
    for name, repo_id in ALL_MODELS.items():
        is_cached = repo_id in cached
        size = cached.get(repo_id, (False, 0))[1] if is_cached else 0
        statuses.append({
            "name": name,
            "repo_id": repo_id,
            "downloaded": is_cached,
            "size": size,
        })
    return statuses


def load_saved_voices() -> dict:
    """Scan saved_voices/ and return dict of saved voices."""
    voices = {}
    for voice_dir in SAVED_VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            transcript_path = voice_dir / "transcript.txt"
            metadata_path = voice_dir / "metadata.json"
            audio_files = list(voice_dir.glob("audio.*"))
            if audio_files and transcript_path.exists():
                import json
                voices[voice_dir.name] = {
                    "audio": str(audio_files[0]),
                    "transcript": transcript_path.read_text().strip(),
                    "metadata": json.loads(metadata_path.read_text()) if metadata_path.exists() else {},
                }
    return voices


def validate_temperature(temperature: float) -> float:
    """Validate temperature and return float."""
    if not isinstance(temperature, (int, float)):
        raise ValueError("Temperature must be a number")
    if temperature < 0.0 or temperature > 2.0:
        raise ValueError("Temperature must be between 0.0 and 2.0")
    return float(temperature)


def save_generation(audio: np.ndarray, voice: str, temp: float, instruct: str, is_clone: bool = False) -> dict:
    """Save generated audio to outputs/ and return metadata."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "clone" if is_clone else voice
    filename = f"{prefix}_{timestamp}.wav"
    filepath = OUTPUTS_DIR / filename

    sf.write(filepath, audio, 24000)

    return {
        "path": str(filepath),
        "filename": filename,
        "voice": voice,
        "temperature": temp,
        "instruct": instruct,
        "timestamp": timestamp,
        "is_clone": is_clone,
    }


def generate_preset_audio(
    text: str,
    voice: str,
    instruct: str,
    temperature: float,
    model_name: str = "1.7B-CustomVoice",
) -> tuple[np.ndarray, dict]:
    """Generate audio using a preset voice.

    Returns (audio_array, metadata_dict).
    Raises ValueError for bad input, RuntimeError for generation failures.
    """
    if not text.strip():
        raise ValueError("Please enter text to synthesize")

    model_path = PRESET_MODELS.get(model_name)
    if not model_path:
        raise ValueError(f"Unknown preset model: {model_name}")
    if not is_model_downloaded(model_path):
        raise RuntimeError(f"Model not downloaded: '{model_name}'")

    temp = validate_temperature(temperature)
    model = get_model(model_path)

    voice_name = voice.split(" (")[0]

    results = list(model.generate(
        text=text,
        voice=voice_name,
        instruct=instruct.strip() or None,
        temperature=temp,
    ))

    if not results:
        raise RuntimeError("Generation failed - no audio produced")

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, voice_name, temp, instruct.strip())
    return audio, metadata


def generate_design_audio(
    text: str,
    instruct: str,
    language: str = "Auto",
    temperature: float = 0.9,
) -> tuple[np.ndarray, dict]:
    """Generate audio using the VoiceDesign model.

    Returns (audio_array, metadata_dict).
    Raises ValueError for bad input, RuntimeError for generation failures.
    """
    if not text.strip():
        raise ValueError("Please enter text to synthesize")
    if not instruct.strip():
        raise ValueError("Please enter a voice description")

    model_path = DESIGN_MODELS["1.7B-VoiceDesign"]
    if not is_model_downloaded(model_path):
        raise RuntimeError("Model not downloaded: '1.7B-VoiceDesign'")

    temp = validate_temperature(temperature)
    model = get_model(model_path)

    results = list(model.generate(
        text=text,
        instruct=instruct.strip(),
        lang_code=language.lower(),
        temperature=temp,
    ))

    if not results:
        raise RuntimeError("Generation failed - no audio produced")

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, "designed", temp, instruct.strip())
    return audio, metadata


def generate_clone_audio(
    text: str,
    saved_voice: str,
    temperature: float = 1.0,
    model_name: str = "1.7B-Base",
) -> tuple[np.ndarray, dict]:
    """Generate audio using voice cloning with a saved voice.

    Returns (audio_array, metadata_dict).
    Raises ValueError for bad input, RuntimeError for generation failures.
    """
    if not text.strip():
        raise ValueError("Please enter text to synthesize")
    if not saved_voice:
        raise ValueError("Please select a voice")

    model_path = CLONE_MODELS.get(model_name)
    if not model_path:
        raise ValueError(f"Unknown clone model: {model_name}")
    if not is_model_downloaded(model_path):
        raise RuntimeError(f"Model not downloaded: '{model_name}'")

    saved_voices = load_saved_voices()
    if saved_voice not in saved_voices:
        raise ValueError(f"Voice '{saved_voice}' not found. Create a voice first.")

    voice_data = saved_voices[saved_voice]
    actual_audio = voice_data["audio"]
    actual_text = voice_data["transcript"]

    temp = validate_temperature(temperature)
    model = get_model(model_path)

    audio_np, _ = librosa.load(actual_audio, sr=24000, mono=True)
    audio_mx = mx.array(audio_np)

    results = list(model.generate(
        text=text,
        ref_audio=audio_mx,
        ref_text=actual_text,
        temperature=temp,
    ))

    if not results:
        raise RuntimeError("Generation failed - no audio produced")

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, saved_voice, temp, "", is_clone=True)
    return audio, metadata
