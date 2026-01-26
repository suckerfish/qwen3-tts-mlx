"""Gradio web app for Qwen3-TTS on Apple Silicon."""

import json
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

VOICES = ["Ryan", "Vivian", "Serena", "Aiden", "Dylan", "Eric", "Uncle_Fu", "Ono_Anna", "Sohee"]
INSTRUCT_PRESETS = [
    "excited and happy",
    "calm and soothing",
    "serious and professional",
    "warm and friendly",
]

PRESET_MODELS = {
    "1.7B-CustomVoice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
}
CLONE_MODELS = {
    "0.6B-Base": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
}

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


def load_saved_voices() -> dict:
    """Scan saved_voices/ on startup and return dict of saved voices."""
    voices = {}
    for voice_dir in SAVED_VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            audio_path = voice_dir / "audio.wav"
            transcript_path = voice_dir / "transcript.txt"
            metadata_path = voice_dir / "metadata.json"
            if audio_path.exists() and transcript_path.exists():
                voices[voice_dir.name] = {
                    "audio": str(audio_path),
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
    shutil.copy(audio_path, voice_dir / "audio.wav")
    (voice_dir / "transcript.txt").write_text(transcript.strip())
    (voice_dir / "metadata.json").write_text(json.dumps({
        "name": name.strip(),
        "created": datetime.now().isoformat(),
    }))

    return f"Voice '{safe_name}' saved"


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

    if not new_name.endswith(".wav"):
        new_name = new_name + ".wav"

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

    temp = validate_temperature(temp_str)
    model_path = PRESET_MODELS[model_name]
    model = get_model(model_path)

    results = list(model.generate(
        text=text,
        voice=voice,
        instruct=instruct.strip() or None,
        temperature=temp,
    ))

    audio = np.array(results[0].audio)
    metadata = save_generation(audio, voice, temp, instruct.strip())

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
    model_path = CLONE_MODELS[model_name]
    model = get_model(model_path)

    results = list(model.generate(
        text=text,
        audio=actual_audio,
        ref_text=actual_text,
        temperature=temp,
    ))

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
                        value="Ryan",
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

                    clone_ref_audio = gr.Audio(
                        label="Reference audio",
                        type="filepath",
                        sources=["upload"],
                    )

                    clone_ref_text = gr.Textbox(
                        label="Reference transcript",
                        placeholder="Words spoken in reference audio...",
                        lines=2,
                        elem_classes=["compact-input"],
                    )

                    with gr.Row():
                        save_voice_name = gr.Textbox(
                            label="Save as",
                            placeholder="Voice name...",
                            scale=2,
                            elem_classes=["compact-input"],
                        )
                        save_voice_btn = gr.Button("Save", size="sm", scale=1)

                    save_voice_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        visible=False,
                    )

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
                        delete_btn = gr.Button("🗑️", size="sm", scale=0, min_width=40)
                    audio_player = gr.Audio(
                        type="filepath",
                        label="",
                    )
                output_slots.append({
                    "filename": filename_box,
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
        msg = save_cloned_voice(audio, transcript, name)
        new_choices = get_saved_voice_choices()
        return gr.update(value=msg, visible=True), gr.update(choices=new_choices)

    def do_refresh_voices():
        return gr.update(choices=get_saved_voice_choices())

    # Slot 0 outputs (current) for generate
    slot0_outputs = [output_slots[0]["container"], output_slots[0]["filename"], output_slots[0]["audio"]]

    # Chain: first shift history (instant), then generate (slow, only updates slot 0)
    preset_btn.click(
        fn=shift_history,
        inputs=[history_state],
        outputs=all_slot_outputs,
    ).then(
        fn=do_generate_preset,
        inputs=[preset_text, preset_voice, preset_instruct, preset_temp, preset_model, history_state],
        outputs=[history_state] + slot0_outputs,
    )

    clone_btn.click(
        fn=shift_history,
        inputs=[history_state],
        outputs=all_slot_outputs,
    ).then(
        fn=do_generate_clone,
        inputs=[clone_text, saved_voice_dropdown, clone_ref_audio, clone_ref_text, clone_temp, clone_model, history_state],
        outputs=[history_state] + slot0_outputs,
    )

    # Wire up rename (on blur/submit) and delete for each slot
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
        slot["delete"].click(
            fn=make_delete_handler(i),
            inputs=[history_state],
            outputs=[history_state] + all_slot_outputs,
            js="() => confirm('Delete this audio file from disk?')",
        )

    save_voice_btn.click(
        fn=do_save_voice,
        inputs=[clone_ref_audio, clone_ref_text, save_voice_name],
        outputs=[save_voice_status, saved_voice_dropdown],
    )

if __name__ == "__main__":
    app.launch(css=CSS)
