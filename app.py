"""Gradio web app for Qwen3-TTS on Apple Silicon."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
from huggingface_hub import scan_cache_dir, snapshot_download
import librosa
import mlx.core as mx
import numpy as np
import soundfile as sf
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
INSTRUCT_PRESETS = [
    "speak with excitement and enthusiasm",
    "slow deliberate pace with dramatic pauses",
    "steady speed, clear articulation",
    "whispered, secretive tone",
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

VOICE_DESIGN_PRESETS = [
    "wise elderly mentor, warm and reassuring, measured pace",
    "energetic young narrator, bright and enthusiastic",
    "calm professional news anchor, clear and authoritative",
    "friendly storyteller, expressive with gentle warmth",
]

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

OUTPUTS_DIR = Path("outputs")
SAVED_VOICES_DIR = Path("saved_voices")
OUTPUTS_DIR.mkdir(exist_ok=True)
SAVED_VOICES_DIR.mkdir(exist_ok=True)

models = {}


def get_model(model_path: str):
    """Lazy load models to save memory."""
    if model_path not in models:
        models[model_path] = load_model(model_path)
    return models[model_path]


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
    """Get status of all models for UI display."""
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


def download_model(repo_id: str, progress=gr.Progress()) -> str:
    """Download model with Gradio progress tracking."""
    progress(0, desc="Starting download...")
    snapshot_download(repo_id, allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.tiktoken"])
    return f"Downloaded {repo_id}"


def delete_model(repo_id: str) -> str:
    """Delete model from cache."""
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            revision_hashes = [rev.commit_hash for rev in repo.revisions]
            strategy = cache_info.delete_revisions(*revision_hashes)
            strategy.execute()
            return f"Deleted {repo_id}"
    return f"Model {repo_id} not found in cache"


def load_saved_voices() -> dict:
    """Scan saved_voices/ on startup and return dict of saved voices."""
    voices = {}
    for voice_dir in SAVED_VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            transcript_path = voice_dir / "transcript.txt"
            metadata_path = voice_dir / "metadata.json"
            # Find audio file with any extension
            audio_files = list(voice_dir.glob("audio.*"))
            if audio_files and transcript_path.exists():
                voices[voice_dir.name] = {
                    "audio": str(audio_files[0]),
                    "transcript": transcript_path.read_text().strip(),
                    "metadata": json.loads(metadata_path.read_text()) if metadata_path.exists() else {},
                }
    return voices


def save_cloned_voice(audio_path: str, transcript: str, name: str) -> str:
    """Persist a cloned voice to saved_voices/."""
    if not name.strip():
        raise gr.Error("Please enter a name for the voice")
    if not audio_path:
        raise gr.Error("Please upload reference audio first")
    if not transcript.strip():
        raise gr.Error("Please enter the transcript first")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    voice_dir = SAVED_VOICES_DIR / safe_name

    if voice_dir.exists():
        raise gr.Error(f"Voice '{safe_name}' already exists")

    voice_dir.mkdir(parents=True)

    # Copy audio file preserving original format
    original_ext = Path(audio_path).suffix or ".wav"
    shutil.copy(audio_path, voice_dir / f"audio{original_ext}")

    (voice_dir / "transcript.txt").write_text(transcript.strip())
    (voice_dir / "metadata.json").write_text(json.dumps({
        "name": name.strip(),
        "created": datetime.now().isoformat(),
    }))

    return safe_name


def save_designed_voice(audio_path: str, transcript: str, instruct: str, name: str) -> str:
    """Save a designed voice for reuse in voice cloning."""
    if not name.strip():
        raise gr.Error("Please enter a name for the voice")
    if not audio_path:
        raise gr.Error("Please generate audio first")
    if not transcript.strip():
        raise gr.Error("No transcript available")

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    voice_dir = SAVED_VOICES_DIR / safe_name

    if voice_dir.exists():
        raise gr.Error(f"Voice '{safe_name}' already exists")

    voice_dir.mkdir(parents=True)

    # Copy audio file
    shutil.copy(audio_path, voice_dir / "audio.wav")

    (voice_dir / "transcript.txt").write_text(transcript.strip())
    (voice_dir / "metadata.json").write_text(json.dumps({
        "name": name.strip(),
        "created": datetime.now().isoformat(),
        "voice_description": instruct.strip() if instruct else "",
        "type": "designed",
    }))

    return safe_name


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


def rename_generation(history: list, index: int, new_name: str) -> list:
    """Rename a file on disk and update history."""
    if index < 0 or index >= len(history):
        return history

    entry = history[index]
    old_path = Path(entry["path"])

    # Strip any path components to prevent path traversal
    new_name = Path(new_name).name

    if not new_name.endswith(".wav"):
        new_name = new_name + ".wav"

    # Reject invalid names
    if not new_name or new_name == ".wav":
        raise gr.Error("Invalid filename")

    new_path = old_path.parent / new_name

    if new_path.exists() and new_path != old_path:
        raise gr.Error(f"File '{new_name}' already exists")

    if old_path.exists():
        old_path.rename(new_path)

    history[index]["path"] = str(new_path)
    history[index]["filename"] = new_name

    return history


def delete_generation(history: list, index: int, delete_file: bool) -> list:
    """Remove entry from history and optionally delete file from disk."""
    if index < 0 or index >= len(history):
        return history

    entry = history[index]
    if delete_file:
        filepath = Path(entry["path"])
        if filepath.exists():
            filepath.unlink()

    return history[:index] + history[index + 1:]


def validate_temperature(temp_str: str) -> float:
    """Validate temperature input and return float."""
    try:
        temp = float(temp_str)
    except ValueError:
        raise gr.Error("Temperature must be a number")

    if temp < 0.0 or temp > 2.0:
        raise gr.Error("Temperature must be between 0.0 and 2.0")

    return temp


def generate_preset(
    text: str, voice: str, instruct: str, temp_str: str, model_name: str, history: list
) -> tuple:
    """Generate audio using preset voice."""
    if not text.strip():
        raise gr.Error("Please enter text to synthesize")

    model_path = PRESET_MODELS[model_name]
    if not is_model_downloaded(model_path):
        raise gr.Error(f"Model not downloaded. Go to Models tab to download '{model_name}'")

    temp = validate_temperature(temp_str)
    model = get_model(model_path)

    # Extract voice name without description
    voice_name = voice.split(" (")[0]

    results = list(model.generate(
        text=text,
        voice=voice_name,
        instruct=instruct.strip() or None,
        temperature=temp,
    ))

    if not results:
        raise gr.Error("Generation failed - no audio produced")

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, voice_name, temp, instruct.strip())

    new_history = [metadata] + history
    return metadata["path"], new_history


def generate_design(
    text: str, instruct: str, language: str, temp_str: str, history: list
) -> tuple:
    """Generate audio using VoiceDesign model."""
    if not text.strip():
        raise gr.Error("Please enter text to synthesize")
    if not instruct.strip():
        raise gr.Error("Please enter a voice description")

    model_path = DESIGN_MODELS["1.7B-VoiceDesign"]
    if not is_model_downloaded(model_path):
        raise gr.Error("Model not downloaded. Go to Models tab to download '1.7B-VoiceDesign'")

    temp = validate_temperature(temp_str)
    model = get_model(model_path)

    results = list(model.generate(
        text=text,
        instruct=instruct.strip(),
        lang_code=language.lower(),
        temperature=temp,
    ))

    if not results:
        raise gr.Error("Generation failed - no audio produced")

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, "designed", temp, instruct.strip())

    new_history = [metadata] + history
    return metadata["path"], new_history


def generate_clone(
    text: str,
    saved_voice: str,
    ref_audio: str,
    ref_text: str,
    temp_str: str,
    model_name: str,
    history: list,
) -> tuple:
    """Generate audio using voice cloning."""
    if not text.strip():
        raise gr.Error("Please enter text to synthesize")

    model_path = CLONE_MODELS[model_name]
    if not is_model_downloaded(model_path):
        raise gr.Error(f"Model not downloaded. Go to Models tab to download '{model_name}'")

    saved_voices = load_saved_voices()

    if saved_voice and saved_voice != "-- Upload new --":
        if saved_voice not in saved_voices:
            raise gr.Error(f"Saved voice '{saved_voice}' not found")
        voice_data = saved_voices[saved_voice]
        actual_audio = voice_data["audio"]
        actual_text = voice_data["transcript"]
        voice_name = saved_voice
    else:
        if not ref_audio:
            raise gr.Error("Please upload a reference audio file")
        if not ref_text.strip():
            raise gr.Error("Please enter the reference audio transcript")
        actual_audio = ref_audio
        actual_text = ref_text.strip()
        voice_name = "custom"

    temp = validate_temperature(temp_str)
    model = get_model(model_path)

    # Load audio with librosa (supports M4A, MP3, etc.) and convert to mlx array
    audio_np, _ = librosa.load(actual_audio, sr=24000, mono=True)
    audio_mx = mx.array(audio_np)

    results = list(model.generate(
        text=text,
        ref_audio=audio_mx,
        ref_text=actual_text,
        temperature=temp,
    ))

    if not results:
        raise gr.Error("Generation failed - no audio produced")

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, voice_name, temp, "", is_clone=True)

    new_history = [metadata] + history
    return metadata["path"], new_history


def get_saved_voice_choices():
    """Get list of saved voice names for dropdown."""
    voices = load_saved_voices()
    choices = ["-- Upload new --"] + list(voices.keys())
    return choices


def build_metadata_str(entry: dict) -> str:
    """Build metadata string for display."""
    voice_label = f"Clone: {entry['voice']}" if entry.get("is_clone") else entry["voice"]
    parts = [voice_label, f"T={entry['temperature']}"]
    if entry["instruct"]:
        instruct_short = entry["instruct"][:30] + "..." if len(entry["instruct"]) > 30 else entry["instruct"]
        parts.append(instruct_short)
    return " | ".join(parts)


def refresh_all_slots(history: list):
    """Refresh all 5 output slots based on history state.

    Returns updates for: [slot0_container, slot0_filename, slot0_audio, ...]
    """
    updates = []
    for i in range(5):
        if i < len(history):
            entry = history[i]
            updates.append(gr.update(visible=True))  # container
            updates.append(gr.update(value=entry["filename"]))  # filename
            updates.append(gr.update(value=entry["path"], label=build_metadata_str(entry)))  # audio with metadata label
        else:
            # Slot 0 stays visible but empty, others hide
            updates.append(gr.update(visible=(i == 0)))  # container
            updates.append(gr.update(value=""))  # filename
            updates.append(gr.update(value=None, label=""))  # audio
    return updates


def refresh_slots_for_shift(history: list):
    """Prepare slots for shift before generation - shows current history in shifted positions."""
    updates = []
    # Slot 0: will show "Generating..." state
    updates.append(gr.update(visible=True))  # container
    updates.append(gr.update(value=""))  # filename cleared
    updates.append(gr.update(value=None, label="Generating..."))  # audio with status label

    # Slots 1-4: show current history items (which will become previous)
    for i in range(4):
        if i < len(history):
            entry = history[i]
            updates.append(gr.update(visible=True))  # container
            updates.append(gr.update(value=entry["filename"]))  # filename
            updates.append(gr.update(value=entry["path"], label=build_metadata_str(entry)))  # audio with metadata label
        else:
            updates.append(gr.update(visible=False))  # container
            updates.append(gr.update(value=""))  # filename
            updates.append(gr.update(value=None, label=""))  # audio
    return updates


CSS = """
.compact-input textarea { font-size: 14px !important; }
.compact-input input { font-size: 14px !important; }
.preset-btn { min-width: 0 !important; padding: 4px 8px !important; font-size: 12px !important; }
.history-section { border-left: 2px solid #444; padding-left: 16px; }
.generate-btn { margin-top: 8px !important; }
.filename-row .progress-bar, .filename-row .progress-text, .filename-row .eta-bar,
.filename-row .wrap, .filename-row .generating { display: none !important; }
.icon-btn { min-width: 36px !important; max-width: 36px !important; min-height: 42px !important; padding: 4px !important; font-size: 20px !important; }
.icon-btn-divider { border-right: 1px solid #666 !important; }
.filename-row { align-items: stretch !important; }
.filename-row > div { display: flex !important; align-items: stretch !important; }
"""

with gr.Blocks(title="Qwen3-TTS") as app:
    history_state = gr.State([])

    gr.Markdown("# Qwen3-TTS")

    with gr.Row():
        # LEFT COLUMN - Controls
        with gr.Column(scale=1):
            with gr.Tabs():
                # PRESET VOICES TAB
                with gr.Tab("Preset Voices"):
                    preset_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to synthesize...",
                        lines=4,
                        elem_classes=["compact-input"],
                    )

                    preset_voice = gr.Dropdown(
                        choices=VOICES,
                        value=VOICES[0],
                        label="Voice",
                    )

                    preset_model = gr.Dropdown(
                        choices=list(PRESET_MODELS.keys()),
                        value=list(PRESET_MODELS.keys())[0],
                        label="Model",
                    )

                    gr.Markdown("**Style presets**", elem_id="style-label")
                    with gr.Row():
                        preset_btns = []
                        for preset in INSTRUCT_PRESETS:
                            btn = gr.Button(preset, size="sm", elem_classes=["preset-btn"])
                            preset_btns.append(btn)

                    preset_instruct = gr.Textbox(
                        label="Style instruction",
                        placeholder="e.g., excited and happy...",
                        lines=1,
                        elem_classes=["compact-input"],
                    )

                    for btn, preset in zip(preset_btns, INSTRUCT_PRESETS):
                        btn.click(fn=lambda p=preset: p, outputs=preset_instruct)

                    preset_temp = gr.Textbox(
                        label="Temperature (0.0 - 2.0)",
                        value="1.0",
                        elem_classes=["compact-input"],
                    )

                    preset_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])

                # VOICE DESIGN TAB
                with gr.Tab("Voice Design"):
                    design_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to synthesize...",
                        lines=4,
                        elem_classes=["compact-input"],
                    )

                    gr.Markdown("**Voice description presets**")
                    with gr.Row():
                        design_preset_btns = []
                        for preset in VOICE_DESIGN_PRESETS:
                            btn = gr.Button(preset, size="sm", elem_classes=["preset-btn"])
                            design_preset_btns.append(btn)

                    design_instruct = gr.Textbox(
                        label="Voice Description",
                        placeholder="e.g., deep male voice with British accent, calm and authoritative...",
                        lines=3,
                        elem_classes=["compact-input"],
                    )

                    for btn, preset in zip(design_preset_btns, VOICE_DESIGN_PRESETS):
                        btn.click(fn=lambda p=preset: p, outputs=design_instruct)

                    design_language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="Auto",
                        label="Language",
                    )

                    design_temp = gr.Textbox(
                        label="Temperature (0.0 - 2.0)",
                        value="0.9",
                        elem_classes=["compact-input"],
                    )

                    design_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])

                    with gr.Accordion("Save as Voice", open=False):
                        design_save_name = gr.Textbox(
                            label="Voice name",
                            placeholder="Name for this voice...",
                            elem_classes=["compact-input"],
                        )
                        design_save_btn = gr.Button("Save Voice", size="sm")

                # VOICE CLONING TAB
                with gr.Tab("Voice Cloning"):
                    clone_text = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to synthesize...",
                        lines=4,
                        elem_classes=["compact-input"],
                    )

                    saved_voice_dropdown = gr.Dropdown(
                        choices=get_saved_voice_choices(),
                        value="-- Upload new --",
                        label="Saved Voice",
                        interactive=True,
                    )

                    with gr.Accordion("Create New Voice", open=False):
                        clone_ref_audio = gr.Audio(
                            label="Reference audio",
                            type="filepath",
                            sources=["upload"],
                        )
                        clone_ref_text = gr.Textbox(
                            label="Transcript",
                            placeholder="Words spoken in reference audio...",
                            lines=2,
                            elem_classes=["compact-input"],
                        )
                        with gr.Row():
                            save_voice_name = gr.Textbox(
                                label="Voice name",
                                placeholder="Name for this voice...",
                                scale=3,
                                elem_classes=["compact-input"],
                            )
                            save_voice_btn = gr.Button("Save", size="sm", scale=1)

                    clone_model = gr.Dropdown(
                        choices=list(CLONE_MODELS.keys()),
                        value=list(CLONE_MODELS.keys())[0],
                        label="Model",
                    )

                    clone_temp = gr.Textbox(
                        label="Temperature (0.0 - 2.0)",
                        value="1.0",
                        elem_classes=["compact-input"],
                    )

                    clone_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])

                # MODELS TAB
                with gr.Tab("Models"):
                    gr.Markdown("### Model Management")
                    gr.Markdown("Download models before use. Models are stored in `~/.cache/huggingface/hub/`")

                    model_status_display = gr.Dataframe(
                        headers=["Model", "Status", "Size"],
                        label="Available Models",
                        interactive=False,
                    )

                    with gr.Row():
                        model_selector = gr.Dropdown(
                            choices=list(ALL_MODELS.keys()),
                            label="Select Model",
                        )
                        download_btn = gr.Button("Download", variant="primary")
                        delete_btn_models = gr.Button("Delete", variant="stop")

                    model_output = gr.Textbox(label="Status", interactive=False)
                    refresh_btn = gr.Button("Refresh Status")

        # RIGHT COLUMN - Output
        with gr.Column(scale=1, elem_classes=["history-section"]):
            gr.Markdown("### Output")

            # Create 5 output slots (current + 4 history)
            output_slots = []

            for i in range(5):
                is_current = i == 0
                with gr.Group(visible=is_current) as container:
                    with gr.Row(elem_classes=["filename-row"]):
                        filename_box = gr.Textbox(
                            label="",
                            placeholder="Generated Audio" if is_current else f"Previous {i}",
                            scale=4,
                            container=False,
                        )
                        rewind_btn = gr.Button("⏮", size="sm", scale=0, elem_classes=["icon-btn", "icon-btn-divider"])
                        delete_btn = gr.Button("🗑️", size="sm", scale=0, elem_classes=["icon-btn"])
                    audio_player = gr.Audio(
                        type="filepath",
                        label="",
                    )
                output_slots.append({
                    "filename": filename_box,
                    "rewind": rewind_btn,
                    "delete": delete_btn,
                    "audio": audio_player,
                    "container": container,
                })


    # Build flat lists for outputs
    all_slot_outputs = []
    for slot in output_slots:
        all_slot_outputs.extend([slot["container"], slot["filename"], slot["audio"]])

    # Event handlers
    def shift_history(history):
        """Shift history before generating - instant update."""
        return refresh_slots_for_shift(history)

    def do_generate_preset(text, voice, instruct, temp, model, history):
        path, new_history = generate_preset(text, voice, instruct, temp, model, history)
        # Only update slot 0 (current)
        return [
            new_history,
            gr.update(visible=True),
            gr.update(value=new_history[0]["filename"]),
            gr.update(value=path, label=build_metadata_str(new_history[0])),
        ]

    def do_generate_clone(text, saved_voice, ref_audio, ref_text, temp, model, history):
        path, new_history = generate_clone(text, saved_voice, ref_audio, ref_text, temp, model, history)
        return [
            new_history,
            gr.update(visible=True),
            gr.update(value=new_history[0]["filename"]),
            gr.update(value=path, label=build_metadata_str(new_history[0])),
        ]

    def do_generate_design(text, instruct, language, temp, history):
        path, new_history = generate_design(text, instruct, language, temp, history)
        return [
            new_history,
            gr.update(visible=True),
            gr.update(value=new_history[0]["filename"]),
            gr.update(value=path, label=build_metadata_str(new_history[0])),
        ]

    def make_rename_handler(slot_index):
        def do_rename(new_name, history):
            if slot_index >= len(history):
                return [history] + refresh_all_slots(history)
            if not new_name.strip():
                return [history] + refresh_all_slots(history)
            new_history = rename_generation(history, slot_index, new_name.strip())
            return [new_history] + refresh_all_slots(new_history)
        return do_rename

    def make_delete_handler(slot_index):
        def do_delete(history):
            if slot_index >= len(history):
                return [history] + refresh_all_slots(history)
            # Always delete from disk
            new_history = delete_generation(history, slot_index, delete_file=True)
            return [new_history] + refresh_all_slots(new_history)
        return do_delete

    def do_save_voice(audio, transcript, name):
        safe_name = save_cloned_voice(audio, transcript, name)
        new_choices = get_saved_voice_choices()
        gr.Info(f"Voice '{safe_name}' saved")
        return gr.update(choices=new_choices, value=safe_name)

    def do_save_designed_voice(history, text, instruct, name):
        if not history:
            raise gr.Error("Please generate audio first")
        audio_path = history[0]["path"]
        safe_name = save_designed_voice(audio_path, text, instruct, name)
        new_choices = get_saved_voice_choices()
        gr.Info(f"Voice '{safe_name}' saved")
        return gr.update(choices=new_choices, value=safe_name)

    def do_refresh_voices():
        return gr.update(choices=get_saved_voice_choices())

    # Slot 0 outputs (current) for generate
    slot0_outputs = [output_slots[0]["container"], output_slots[0]["filename"], output_slots[0]["audio"]]

    # JavaScript to reset audio seek position
    reset_audio_js = "() => { document.querySelectorAll('audio').forEach(a => { a.currentTime = 0; a.pause(); }); }"

    # Chain: first shift history (instant), then generate (slow, only updates slot 0), then reset audio
    preset_btn.click(
        fn=shift_history,
        inputs=[history_state],
        outputs=all_slot_outputs,
    ).then(
        fn=do_generate_preset,
        inputs=[preset_text, preset_voice, preset_instruct, preset_temp, preset_model, history_state],
        outputs=[history_state] + slot0_outputs,
    ).then(
        fn=None,
        js=reset_audio_js,
    )

    clone_btn.click(
        fn=shift_history,
        inputs=[history_state],
        outputs=all_slot_outputs,
    ).then(
        fn=do_generate_clone,
        inputs=[clone_text, saved_voice_dropdown, clone_ref_audio, clone_ref_text, clone_temp, clone_model, history_state],
        outputs=[history_state] + slot0_outputs,
    ).then(
        fn=None,
        js=reset_audio_js,
    )

    design_btn.click(
        fn=shift_history,
        inputs=[history_state],
        outputs=all_slot_outputs,
    ).then(
        fn=do_generate_design,
        inputs=[design_text, design_instruct, design_language, design_temp, history_state],
        outputs=[history_state] + slot0_outputs,
    ).then(
        fn=None,
        js=reset_audio_js,
    )

    # Wire up rename (on blur/submit), rewind, and delete for each slot
    rewind_js = """(e) => {
        let el = e.target;
        while (el && !el.querySelector('audio')) el = el.parentElement;
        if (el) { const a = el.querySelector('audio'); a.currentTime = 0; a.pause(); }
    }"""
    for i, slot in enumerate(output_slots):
        slot["filename"].submit(
            fn=make_rename_handler(i),
            inputs=[slot["filename"], history_state],
            outputs=[history_state] + all_slot_outputs,
        )
        slot["filename"].blur(
            fn=make_rename_handler(i),
            inputs=[slot["filename"], history_state],
            outputs=[history_state] + all_slot_outputs,
        )
        slot["rewind"].click(fn=None, js=rewind_js)
        slot["delete"].click(
            fn=make_delete_handler(i),
            inputs=[history_state],
            outputs=[history_state] + all_slot_outputs,
            js="() => confirm('Delete this audio file from disk?')",
        )

    save_voice_btn.click(
        fn=do_save_voice,
        inputs=[clone_ref_audio, clone_ref_text, save_voice_name],
        outputs=[saved_voice_dropdown],
    )

    design_save_btn.click(
        fn=do_save_designed_voice,
        inputs=[history_state, design_text, design_instruct, design_save_name],
        outputs=[saved_voice_dropdown],
    )

    # Model management handlers
    def refresh_model_status():
        statuses = get_model_status()
        data = []
        for s in statuses:
            status = "✅ Downloaded" if s["downloaded"] else "❌ Not downloaded"
            size = f"{s['size'] / 1e9:.1f} GB" if s["downloaded"] else "—"
            data.append([s["name"], status, size])
        return data

    def do_download_model(model_name):
        if not model_name:
            return "Please select a model", refresh_model_status()
        repo_id = ALL_MODELS[model_name]
        result = download_model(repo_id)
        return result, refresh_model_status()

    def do_delete_model(model_name):
        if not model_name:
            return "Please select a model", refresh_model_status()
        repo_id = ALL_MODELS[model_name]
        result = delete_model(repo_id)
        return result, refresh_model_status()

    refresh_btn.click(fn=refresh_model_status, outputs=[model_status_display])
    download_btn.click(
        fn=do_download_model,
        inputs=[model_selector],
        outputs=[model_output, model_status_display],
    )
    delete_btn_models.click(
        fn=do_delete_model,
        inputs=[model_selector],
        outputs=[model_output, model_status_display],
        js="() => confirm('Delete this model from cache? You will need to re-download it to use again.')",
    )

if __name__ == "__main__":
    app.launch(css=CSS)
