from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import gradio as gr

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boson_multimodal.webui import CHUNK_LABELS, DEVICE_LABELS, NONE_OPTION, TASK_LABELS, WebUIService, real_backend_ready
from boson_multimodal.webui.service import read_text_file
from boson_multimodal.webui.types import BackendConfig


SERVICE = WebUIService()
SERVER_NAME = os.environ.get("WEBUI_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("WEBUI_PORT", "7860"))
IN_BROWSER = os.environ.get("WEBUI_INBROWSER", "1") != "0"

UI_NONE_LABEL = "\uff08\u65e0\uff09"
TASK_CHOICE_LABELS = {
    "smart_voice": "\u667a\u80fd\u97f3\u8272",
    "voice_clone": "\u58f0\u97f3\u514b\u9686",
    "multi_speaker": "\u591a\u89d2\u8272\u5bf9\u8bdd",
}
CHUNK_CHOICE_LABELS = {
    "none": "\u4e0d\u5206\u6bb5",
    "speaker": "\u6309\u8bf4\u8bdd\u4eba\u5206\u6bb5",
    "word": "\u6309\u5b57\u8bcd\u5206\u6bb5",
}
DEVICE_CHOICE_LABELS = {
    "auto": "\u81ea\u52a8",
    "cuda": "CUDA",
    "mps": "MPS",
    "cpu": "CPU",
}


def get_transcript_choices() -> list[tuple[str, str]]:
    return [(UI_NONE_LABEL, NONE_OPTION), *[(name, name) for name in SERVICE.transcript_presets.keys()]]


def get_scene_choices() -> list[tuple[str, str]]:
    return [(UI_NONE_LABEL, NONE_OPTION), *[(name, name) for name in SERVICE.scene_presets.keys()]]


def get_task_choices() -> list[tuple[str, str]]:
    return [(TASK_CHOICE_LABELS[key], key) for key in TASK_LABELS]


def get_chunk_choices() -> list[tuple[str, str]]:
    return [(CHUNK_CHOICE_LABELS[key], key) for key in CHUNK_LABELS]


def get_device_choices() -> list[tuple[str, str]]:
    return [(DEVICE_CHOICE_LABELS[key], key) for key in DEVICE_LABELS]


def get_voice_choices() -> list[str]:
    return list(SERVICE.voice_presets.keys())


def _lan_host() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


def _listener_urls() -> list[str]:
    if SERVER_NAME in {"0.0.0.0", "::"}:
        urls = [f"http://127.0.0.1:{SERVER_PORT}"]
        lan_host = _lan_host()
        if lan_host:
            urls.append(f"http://{lan_host}:{SERVER_PORT}")
        return urls
    return [f"http://{SERVER_NAME}:{SERVER_PORT}"]


def _serialize_json(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _feedback_markdown(message: str) -> str:
    return message or ""


def _build_service_status(project_name: str, backend_mode: str, connection_message: str) -> str:
    urls = " / ".join(f"`{url}`" for url in _listener_urls())
    backend_label = "\u672c\u5730" if backend_mode == "local" else "\u8fdc\u7a0b vLLM"
    local_status = "\u5df2\u5c31\u7eea" if real_backend_ready() else "\u672a\u5c31\u7eea"
    remote_status = connection_message or "\u672a\u6d4b\u8bd5"
    lines = [
        "### \u670d\u52a1\u72b6\u6001",
        "",
        f"- \u8bbf\u95ee\u5730\u5740: {urls}",
        f"- \u5f53\u524d\u9879\u76ee: `{project_name or '-'}`",
        f"- \u5f53\u524d\u540e\u7aef: `{backend_label}`",
        f"- \u672c\u5730\u6a21\u578b: `{local_status}`",
    ]
    if backend_mode == "remote_vllm":
        lines.append(f"- \u8fdc\u7a0b\u8fde\u901a\u6027: `{remote_status}`")
    return "\n".join(lines)


def _history_page_markdown(page: int, total_pages: int, total_count: int) -> str:
    return f"\u7b2c `{page}` / `{total_pages}` \u9875\uff0c\u5171 `{total_count}` \u6761\u8bb0\u5f55"


def _history_detail_markdown(summary, request_payload: dict[str, Any], result_payload: dict[str, Any]) -> str:
    backend_mode = request_payload.get("backend", {}).get("mode", "local")
    backend_label = "\u672c\u5730" if backend_mode == "local" else "\u8fdc\u7a0b"
    elapsed = result_payload.get("elapsed_seconds")
    elapsed_label = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else "-"
    lines = [
        "### \u8bb0\u5f55\u8be6\u60c5",
        "",
        f"- \u65f6\u95f4: `{summary.created_at}`",
        f"- \u72b6\u6001: `{summary.status}`",
        f"- \u6a21\u5f0f: `{TASK_CHOICE_LABELS.get(summary.task_mode, summary.task_mode)}`",
        f"- \u540e\u7aef: `{backend_label}`",
        f"- \u6807\u9898: `{summary.title}`",
        f"- \u7528\u65f6: `{elapsed_label}`",
        f"- \u8f93\u51fa\u76ee\u5f55: `{summary.run_dir}`",
    ]
    return "\n".join(lines)


def _empty_history_detail() -> tuple[str, None, str]:
    return (
        "### \u5386\u53f2\u8be6\u60c5\n\n\u5f53\u524d\u9879\u76ee\u8fd8\u6ca1\u6709\u751f\u6210\u8bb0\u5f55\uff0c\u6216\u8bf7\u5728\u5de6\u4e0b\u89d2\u9009\u4e2d\u4e00\u6761\u5386\u53f2\u3002",
        None,
        "",
    )


def _default_connection_message(backend_mode: str) -> str:
    if backend_mode == "local":
        return "\u672c\u5730\u6a21\u578b\u5df2\u5c31\u7eea" if real_backend_ready() else "\u672c\u5730\u6a21\u578b\u672a\u5c31\u7eea"
    return "\u672a\u6d4b\u8bd5"


def _project_defaults() -> dict[str, Any]:
    return {
        "project_id": "",
        "project_name": "",
        "backend_mode": "local",
        "remote_base_url": BackendConfig().base_url,
        "remote_model_name": BackendConfig().model_name,
        "remote_api_key": "",
        "remote_timeout_seconds": 120,
        "device": "auto",
        "use_static_kv_cache": True,
        "task_mode": "smart_voice",
        "transcript_preset": NONE_OPTION,
        "transcript_text": "",
        "scene_preset": NONE_OPTION,
        "scene_text": "",
        "voice_presets": [],
        "saved_reference_audio_path": "",
        "reference_audio_transcript": "",
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "seed": 1234,
        "chunk_method": "none",
        "chunk_max_word_num": 200,
        "chunk_max_num_turns": 1,
    }


def _overlay_form_values(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base
    merged = dict(base)
    merged.update(override)
    return merged


def _coerce_seed(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)


def _collect_form_inputs(
    project_name: str,
    backend_mode: str,
    remote_base_url: str,
    remote_model_name: str,
    remote_api_key: str,
    remote_timeout_seconds: int | float,
    task_mode: str,
    transcript_preset: str,
    transcript_file_path: str | None,
    transcript_text: str,
    scene_preset: str,
    scene_text: str,
    voice_presets: list[str] | None,
    saved_reference_audio_path: str | None,
    reference_audio_upload_path: str | None,
    reference_audio_transcript: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    seed: int | float | None,
    chunk_method: str,
    chunk_max_word_num: int,
    chunk_max_num_turns: int,
    device: str,
    use_static_kv_cache: bool,
) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "backend_mode": backend_mode,
        "remote_base_url": remote_base_url,
        "remote_model_name": remote_model_name,
        "remote_api_key": remote_api_key,
        "remote_timeout_seconds": remote_timeout_seconds,
        "task_mode": task_mode,
        "transcript_preset": transcript_preset,
        "transcript_file_path": transcript_file_path,
        "transcript_text": transcript_text,
        "scene_preset": scene_preset,
        "scene_text": scene_text,
        "voice_presets": list(voice_presets or []),
        "saved_reference_audio_path": saved_reference_audio_path or "",
        "reference_audio_upload_path": reference_audio_upload_path,
        "reference_audio_transcript": reference_audio_transcript,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "chunk_method": chunk_method,
        "chunk_max_word_num": chunk_max_word_num,
        "chunk_max_num_turns": chunk_max_num_turns,
        "device": device,
        "use_static_kv_cache": use_static_kv_cache,
    }


def _form_override_from_inputs(form: dict[str, Any]) -> dict[str, Any]:
    transcript_text = form["transcript_text"] or ""
    transcript_file_path = form.get("transcript_file_path")
    if not transcript_text and transcript_file_path:
        try:
            transcript_text = read_text_file(Path(transcript_file_path))
        except OSError:
            transcript_text = ""

    reference_source = form.get("reference_audio_upload_path") or form.get("saved_reference_audio_path") or ""
    return {
        "project_name": form.get("project_name", ""),
        "backend_mode": form.get("backend_mode", "local"),
        "remote_base_url": form.get("remote_base_url", BackendConfig().base_url),
        "remote_model_name": form.get("remote_model_name", BackendConfig().model_name),
        "remote_api_key": form.get("remote_api_key", ""),
        "remote_timeout_seconds": int(form.get("remote_timeout_seconds") or 120),
        "device": form.get("device", "auto"),
        "use_static_kv_cache": bool(form.get("use_static_kv_cache", True)),
        "task_mode": form.get("task_mode", "smart_voice"),
        "transcript_preset": form.get("transcript_preset") or NONE_OPTION,
        "transcript_text": transcript_text,
        "scene_preset": form.get("scene_preset") or NONE_OPTION,
        "scene_text": form.get("scene_text") or "",
        "voice_presets": list(form.get("voice_presets") or []),
        "saved_reference_audio_path": reference_source,
        "reference_audio_transcript": form.get("reference_audio_transcript") or "",
        "temperature": float(form.get("temperature") or 0.8),
        "top_k": int(form.get("top_k") or 50),
        "top_p": float(form.get("top_p") or 0.95),
        "max_new_tokens": int(form.get("max_new_tokens") or 256),
        "seed": _coerce_seed(form.get("seed")),
        "chunk_method": form.get("chunk_method") or "none",
        "chunk_max_word_num": int(form.get("chunk_max_word_num") or 200),
        "chunk_max_num_turns": int(form.get("chunk_max_num_turns") or 1),
    }


def _backend_from_form(form: dict[str, Any]) -> BackendConfig:
    return BackendConfig(
        mode=form.get("backend_mode", "local"),
        base_url=(form.get("remote_base_url") or "").strip() or BackendConfig().base_url,
        model_name=(form.get("remote_model_name") or "").strip() or BackendConfig().model_name,
        api_key=(form.get("remote_api_key") or "").strip(),
        timeout_seconds=max(1, int(form.get("remote_timeout_seconds") or 120)),
        device=form.get("device", "auto"),
        use_static_kv_cache=bool(form.get("use_static_kv_cache", True)),
    )


def _render_page(
    selected_project_id: str,
    search_text: str,
    page: int,
    selected_run_id: str,
    connection_message: str,
    feedback_message: str,
    form_override: dict[str, Any] | None = None,
):
    project_choices = SERVICE.project_choices(search_text)
    project_ids = [value for _, value in project_choices]
    current_project_id = selected_project_id if selected_project_id in project_ids else (project_ids[0] if project_ids else "")

    if not current_project_id:
        defaults = _project_defaults()
        history_detail, history_audio, history_json = _empty_history_detail()
        connection = connection_message or _default_connection_message(defaults["backend_mode"])
        return (
            gr.update(choices=project_choices, value=None),
            "",
            defaults["project_name"],
            defaults["backend_mode"],
            defaults["remote_base_url"],
            defaults["remote_model_name"],
            defaults["remote_api_key"],
            defaults["remote_timeout_seconds"],
            defaults["device"],
            defaults["use_static_kv_cache"],
            defaults["task_mode"],
            defaults["transcript_preset"],
            None,
            defaults["transcript_text"],
            defaults["scene_preset"],
            defaults["scene_text"],
            defaults["voice_presets"],
            defaults["saved_reference_audio_path"],
            None,
            None,
            defaults["reference_audio_transcript"],
            defaults["temperature"],
            defaults["top_k"],
            defaults["top_p"],
            defaults["max_new_tokens"],
            defaults["seed"],
            defaults["chunk_method"],
            defaults["chunk_max_word_num"],
            defaults["chunk_max_num_turns"],
            1,
            gr.update(choices=[], value=None),
            _history_page_markdown(1, 1, 0),
            "",
            history_detail,
            history_audio,
            history_json,
            _feedback_markdown(feedback_message),
            connection,
            connection,
            _build_service_status("", defaults["backend_mode"], connection),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    project = SERVICE.get_project(current_project_id)
    form_values = _overlay_form_values(SERVICE.project_to_form_values(project), form_override)
    history_choices, safe_page, total_pages, total_count = SERVICE.history_choices(current_project_id, page)
    history_ids = [value for _, value in history_choices]
    active_run_id = selected_run_id if selected_run_id in history_ids else (history_ids[0] if history_ids else "")

    if active_run_id:
        summary, request_payload, result_payload = SERVICE.get_run(current_project_id, active_run_id)
        history_detail = _history_detail_markdown(summary, request_payload, result_payload)
        history_audio = result_payload.get("output_audio_path")
        history_json = _serialize_json(request_payload)
    else:
        history_detail, history_audio, history_json = _empty_history_detail()

    connection = connection_message or _default_connection_message(form_values["backend_mode"])
    remote_visible = form_values["backend_mode"] == "remote_vllm"
    local_visible = form_values["backend_mode"] == "local"
    return (
        gr.update(choices=project_choices, value=current_project_id),
        current_project_id,
        form_values["project_name"],
        form_values["backend_mode"],
        form_values["remote_base_url"],
        form_values["remote_model_name"],
        form_values["remote_api_key"],
        form_values["remote_timeout_seconds"],
        form_values["device"],
        form_values["use_static_kv_cache"],
        form_values["task_mode"],
        form_values["transcript_preset"],
        None,
        form_values["transcript_text"],
        form_values["scene_preset"],
        form_values["scene_text"],
        form_values["voice_presets"],
        form_values["saved_reference_audio_path"],
        form_values["saved_reference_audio_path"] or None,
        None,
        form_values["reference_audio_transcript"],
        form_values["temperature"],
        form_values["top_k"],
        form_values["top_p"],
        form_values["max_new_tokens"],
        form_values["seed"],
        form_values["chunk_method"],
        form_values["chunk_max_word_num"],
        form_values["chunk_max_num_turns"],
        safe_page,
        gr.update(choices=history_choices, value=active_run_id or None),
        _history_page_markdown(safe_page, total_pages, total_count),
        active_run_id,
        history_detail,
        history_audio,
        history_json,
        _feedback_markdown(feedback_message),
        connection,
        connection,
        _build_service_status(form_values["project_name"], form_values["backend_mode"], connection),
        gr.update(visible=remote_visible),
        gr.update(visible=local_visible),
    )


def _open_directory(path: str) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
        return
    subprocess.Popen(["xdg-open", path])


def _load_preset_text(preset_name: str, presets: dict[str, Path]) -> str:
    if not preset_name or preset_name == NONE_OPTION:
        return ""
    return read_text_file(presets[preset_name])


def _load_uploaded_text(file_path: str | None) -> str:
    if not file_path:
        return ""
    return read_text_file(Path(file_path))


def _toggle_backend_groups(backend_mode: str):
    return gr.update(visible=backend_mode == "remote_vllm"), gr.update(visible=backend_mode == "local")


def build_app() -> gr.Blocks:
    css = """
    .app-shell {max-width: 1520px; margin: 0 auto;}
    .status-card {border: 1px solid #d6d2c4; border-radius: 18px; padding: 16px; background: linear-gradient(180deg, #f8f2e7 0%, #f3ece1 100%);}
    .panel-card {border: 1px solid #e0d8c9; border-radius: 16px; padding: 14px;}
    """

    transcript_choices = get_transcript_choices()
    scene_choices = get_scene_choices()
    task_choices = get_task_choices()
    chunk_choices = get_chunk_choices()
    device_choices = get_device_choices()
    voice_choices = get_voice_choices()

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="amber")) as demo:
        selected_project_id = gr.State("")
        history_page_state = gr.State(1)
        selected_run_id = gr.State("")
        connection_message_state = gr.State("")

        with gr.Column(elem_classes=["app-shell"]):
            gr.Markdown(
                """
                # Higgs Audio \u9879\u76ee\u5236 Web UI

                \u5de6\u4fa7\u7ba1\u9879\u76ee\uff0c\u53f3\u4fa7\u7ba1\u53c2\u6570\u548c\u5386\u53f2\u3002
                \u9879\u76ee\u53c2\u6570\u9700\u8981\u70b9\u201c\u4fdd\u5b58\u9879\u76ee\u53c2\u6570\u201d\u624d\u4f1a\u6301\u4e45\u5316\uff0c
                \u201c\u5f00\u59cb\u751f\u6210\u201d\u53ea\u4f1a\u4fdd\u5b58\u4e3a\u4e00\u6761\u5386\u53f2\u8bb0\u5f55\u3002
                """
            )
            service_status = gr.Markdown(elem_classes=["status-card"])
            project_feedback = gr.Markdown()

            with gr.Row():
                with gr.Column(scale=3, min_width=300):
                    project_search = gr.Textbox(label="\u641c\u7d22\u9879\u76ee", placeholder="\u8f93\u5165\u9879\u76ee\u540d")
                    project_list = gr.Radio(label="\u9879\u76ee\u5217\u8868", choices=[])
                    project_name = gr.Textbox(label="\u9879\u76ee\u540d\u79f0", placeholder="\u4f8b\u5982\uff1a\u5e7f\u64ad\u7247\u82b1")
                    with gr.Row():
                        create_project_button = gr.Button("\u65b0\u5efa\u9879\u76ee", variant="primary")
                        delete_project_button = gr.Button("\u5220\u9664\u9879\u76ee", variant="stop")

                with gr.Column(scale=7):
                    with gr.Row():
                        with gr.Column(scale=6, elem_classes=["panel-card"]):
                            task_mode = gr.Radio(choices=task_choices, value="smart_voice", label="\u751f\u6210\u6a21\u5f0f")
                            transcript_preset = gr.Dropdown(
                                choices=transcript_choices,
                                value=NONE_OPTION,
                                label="\u6587\u672c\u9884\u8bbe",
                            )
                            transcript_file = gr.File(
                                label="\u4e0a\u4f20\u6587\u672c\u6587\u4ef6",
                                file_count="single",
                                type="filepath",
                            )
                            transcript_text = gr.Textbox(
                                label="\u6587\u672c\u5185\u5bb9",
                                lines=10,
                                placeholder="\u76f4\u63a5\u8f93\u5165\u6587\u672c\uff0c\u6216\u5148\u9009\u62e9/\u4e0a\u4f20\u6587\u672c\u3002",
                            )
                            with gr.Row():
                                scene_preset = gr.Dropdown(
                                    choices=scene_choices,
                                    value=NONE_OPTION,
                                    label="\u573a\u666f\u9884\u8bbe",
                                )
                                voice_presets = gr.Dropdown(
                                    choices=voice_choices,
                                    value=[],
                                    multiselect=True,
                                    label="\u9884\u8bbe\u97f3\u8272",
                                )
                            scene_text = gr.Textbox(
                                label="\u81ea\u5b9a\u4e49\u573a\u666f\u63cf\u8ff0",
                                lines=4,
                                placeholder="\u53ef\u7559\u7a7a\uff0c\u6216\u8f93\u5165/\u9009\u62e9\u573a\u666f\u63cf\u8ff0\u3002",
                            )
                            saved_reference_audio_path = gr.Textbox(visible=False)
                            saved_reference_audio = gr.Audio(
                                label="\u5f53\u524d\u53c2\u8003\u97f3\u9891",
                                type="filepath",
                                interactive=False,
                            )
                            ref_audio_upload = gr.File(
                                label="\u4e0a\u4f20\u65b0\u53c2\u8003\u97f3\u9891\uff08voice clone \u7528\uff09",
                                file_count="single",
                                type="filepath",
                            )
                            clear_reference_audio_button = gr.Button("\u6e05\u7a7a\u5f53\u524d\u53c2\u8003\u97f3\u9891")
                            reference_audio_transcript = gr.Textbox(
                                label="\u53c2\u8003\u97f3\u9891\u6587\u672c",
                                lines=3,
                                placeholder="\u4e0a\u4f20\u81ea\u5b9a\u4e49\u53c2\u8003\u97f3\u9891\u65f6\u5fc5\u586b\u3002",
                            )
                            with gr.Accordion("\u9ad8\u7ea7\u53c2\u6570", open=False):
                                with gr.Row():
                                    temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="\u6e29\u5ea6")
                                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top P")
                                with gr.Row():
                                    top_k = gr.Slider(1, 200, value=50, step=1, label="Top K")
                                    max_new_tokens = gr.Slider(
                                        128, 4096, value=256, step=64, label="\u6700\u5927\u65b0 token \u6570"
                                    )
                                with gr.Row():
                                    seed = gr.Number(value=1234, precision=0, label="\u968f\u673a\u79cd\u5b50")
                                    chunk_method = gr.Dropdown(
                                        choices=chunk_choices,
                                        value="none",
                                        label="\u5206\u6bb5\u65b9\u5f0f",
                                    )
                                with gr.Row():
                                    chunk_max_word_num = gr.Slider(
                                        20, 400, value=200, step=10, label="\u6bcf\u6bb5\u6700\u5927\u5b57\u8bcd\u6570"
                                    )
                                    chunk_max_num_turns = gr.Slider(
                                        1, 8, value=1, step=1, label="\u6bcf\u6bb5\u6700\u5927\u8f6e\u6b21\u6570"
                                    )

                        with gr.Column(scale=4, elem_classes=["panel-card"]):
                            backend_mode = gr.Radio(
                                choices=[("\u672c\u5730", "local"), ("\u8fdc\u7a0b vLLM", "remote_vllm")],
                                value="local",
                                label="\u540e\u7aef\u6a21\u5f0f",
                            )
                            with gr.Group(visible=False) as remote_group:
                                remote_base_url = gr.Textbox(label="\u8fdc\u7a0b\u5730\u5740", value=BackendConfig().base_url)
                                remote_model_name = gr.Textbox(
                                    label="\u8fdc\u7a0b\u6a21\u578b\u540d",
                                    value=BackendConfig().model_name,
                                )
                                remote_api_key = gr.Textbox(
                                    label="API Key",
                                    type="password",
                                    placeholder="\u5185\u7f51\u73af\u5883\u53ef\u4e3a\u7a7a",
                                )
                                remote_timeout_seconds = gr.Number(
                                    label="\u8bf7\u6c42\u8d85\u65f6\uff08\u79d2\uff09",
                                    value=120,
                                    precision=0,
                                )
                                test_connection_button = gr.Button("\u6d4b\u8bd5\u8fde\u63a5")
                            with gr.Group(visible=True) as local_group:
                                device = gr.Dropdown(
                                    choices=device_choices,
                                    value="auto",
                                    label="\u8bbe\u5907",
                                )
                                use_static_kv_cache = gr.Checkbox(
                                    value=True,
                                    label="\u542f\u7528\u9759\u6001 KV Cache",
                                )
                                local_hint = (
                                    "\u672c\u5730\u6a21\u578b\u5df2\u5c31\u7eea\u3002"
                                    if real_backend_ready()
                                    else "\u672c\u5730\u6a21\u578b\u672a\u5c31\u7eea\uff0c\u53ef\u6539\u7528\u8fdc\u7a0b vLLM\u3002"
                                )
                                gr.Markdown(local_hint)
                            connection_markdown = gr.Markdown("\u672a\u6d4b\u8bd5")
                            save_project_button = gr.Button("\u4fdd\u5b58\u9879\u76ee\u53c2\u6570")
                            generate_button = gr.Button("\u5f00\u59cb\u751f\u6210", variant="primary")

                    gr.Markdown("## \u5386\u53f2\u8bb0\u5f55")
                    with gr.Row():
                        previous_page_button = gr.Button("\u4e0a\u4e00\u9875")
                        next_page_button = gr.Button("\u4e0b\u4e00\u9875")
                        history_page_info = gr.Markdown()
                    with gr.Row():
                        with gr.Column(scale=4, elem_classes=["panel-card"]):
                            history_list = gr.Radio(label="\u8be5\u9879\u76ee\u7684\u751f\u6210\u8bb0\u5f55", choices=[])
                            with gr.Row():
                                load_run_button = gr.Button("\u8f7d\u5165\u5230\u8868\u5355")
                                open_run_dir_button = gr.Button("\u6253\u5f00\u8f93\u51fa\u76ee\u5f55")
                                delete_run_button = gr.Button("\u5220\u9664\u8bb0\u5f55", variant="stop")
                        with gr.Column(scale=6, elem_classes=["panel-card"]):
                            history_detail = gr.Markdown()
                            history_audio = gr.Audio(label="\u5386\u53f2\u97f3\u9891", type="filepath")
                            history_request_json = gr.Code(label="\u672c\u6b21\u8bf7\u6c42", language="json")

        action_form_inputs = [
            project_name,
            backend_mode,
            remote_base_url,
            remote_model_name,
            remote_api_key,
            remote_timeout_seconds,
            task_mode,
            transcript_preset,
            transcript_file,
            transcript_text,
            scene_preset,
            scene_text,
            voice_presets,
            saved_reference_audio_path,
            ref_audio_upload,
            reference_audio_transcript,
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            seed,
            chunk_method,
            chunk_max_word_num,
            chunk_max_num_turns,
            device,
            use_static_kv_cache,
        ]

        page_outputs = [
            project_list,
            selected_project_id,
            project_name,
            backend_mode,
            remote_base_url,
            remote_model_name,
            remote_api_key,
            remote_timeout_seconds,
            device,
            use_static_kv_cache,
            task_mode,
            transcript_preset,
            transcript_file,
            transcript_text,
            scene_preset,
            scene_text,
            voice_presets,
            saved_reference_audio_path,
            saved_reference_audio,
            ref_audio_upload,
            reference_audio_transcript,
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            seed,
            chunk_method,
            chunk_max_word_num,
            chunk_max_num_turns,
            history_page_state,
            history_list,
            history_page_info,
            selected_run_id,
            history_detail,
            history_audio,
            history_request_json,
            project_feedback,
            connection_message_state,
            connection_markdown,
            service_status,
            remote_group,
            local_group,
        ]

        def wrapped(callback):
            def runner(*args):
                try:
                    return callback(*args)
                except Exception as exc:  # noqa: BLE001
                    raise gr.Error(str(exc)) from exc

            return runner

        def on_load():
            return _render_page("", "", 1, "", "", "")

        def on_search_change(search_text, project_id, connection_message, *form_args):
            form = _collect_form_inputs(*form_args)
            return _render_page(project_id, search_text, 1, "", connection_message, "", _form_override_from_inputs(form))

        def on_project_change(project_id, search_text):
            return _render_page(project_id or "", search_text, 1, "", "", "")

        def on_create(project_name_value, search_text):
            project = SERVICE.create_project(project_name_value)
            return _render_page(project.id, search_text, 1, "", "", "\u65b0\u9879\u76ee\u5df2\u521b\u5efa")

        def on_save(project_id, search_text, history_page, *form_args):
            form = _collect_form_inputs(*form_args)
            SERVICE.save_project_form(project_id, form)
            return _render_page(
                project_id,
                search_text,
                history_page,
                "",
                _default_connection_message(form["backend_mode"]),
                "\u9879\u76ee\u53c2\u6570\u5df2\u4fdd\u5b58",
            )

        def on_delete_project(project_id, search_text):
            if not project_id:
                raise ValueError("\u8bf7\u5148\u9009\u62e9\u8981\u5220\u9664\u7684\u9879\u76ee\u3002")
            SERVICE.delete_project(project_id)
            return _render_page("", search_text, 1, "", "", "\u9879\u76ee\u5df2\u5220\u9664")

        def on_test_connection(project_id, search_text, history_page, *form_args):
            if not project_id:
                raise ValueError("\u8bf7\u5148\u9009\u62e9\u4e00\u4e2a\u9879\u76ee\u3002")
            form = _collect_form_inputs(*form_args)
            validation = SERVICE.validate_backend(_backend_from_form(form))
            message = validation.message
            if validation.available_models:
                models_preview = ", ".join(validation.available_models[:5])
                message = f"{message}\n\n\u53ef\u7528\u6a21\u578b: `{models_preview}`"
            return _render_page(
                project_id,
                search_text,
                history_page,
                "",
                message,
                "\u8fde\u63a5\u6d4b\u8bd5\u5b8c\u6210",
                _form_override_from_inputs(form),
            )

        def on_generate(project_id, search_text, connection_message, *form_args):
            form = _collect_form_inputs(*form_args)
            result, _ = SERVICE.generate_for_project(project_id, form)
            feedback = f"\u751f\u6210\u5b8c\u6210\uff0c\u7528\u65f6 {result.elapsed_seconds:.2f} \u79d2"
            return _render_page(
                project_id,
                search_text,
                1,
                "",
                connection_message,
                feedback,
                _form_override_from_inputs(form),
            )

        def on_previous_page(project_id, search_text, history_page, connection_message, *form_args):
            form = _collect_form_inputs(*form_args)
            target_page = max(1, int(history_page) - 1)
            return _render_page(
                project_id,
                search_text,
                target_page,
                "",
                connection_message,
                "",
                _form_override_from_inputs(form),
            )

        def on_next_page(project_id, search_text, history_page, connection_message, *form_args):
            form = _collect_form_inputs(*form_args)
            target_page = int(history_page) + 1
            return _render_page(
                project_id,
                search_text,
                target_page,
                "",
                connection_message,
                "",
                _form_override_from_inputs(form),
            )

        def on_history_change(run_id, project_id, search_text, history_page, connection_message, *form_args):
            form = _collect_form_inputs(*form_args)
            return _render_page(
                project_id,
                search_text,
                history_page,
                run_id or "",
                connection_message,
                "",
                _form_override_from_inputs(form),
            )

        def on_load_run(project_id, run_id, search_text, history_page):
            if not project_id or not run_id:
                raise ValueError("\u8bf7\u5148\u9009\u4e2d\u4e00\u6761\u5386\u53f2\u8bb0\u5f55\u3002")
            form_override = SERVICE.run_to_form_values(project_id, run_id)
            return _render_page(
                project_id,
                search_text,
                history_page,
                run_id,
                _default_connection_message(form_override["backend_mode"]),
                "\u5df2\u5c06\u8be5\u6b21\u53c2\u6570\u8f7d\u5165\u8868\u5355",
                form_override,
            )

        def on_delete_run(project_id, run_id, search_text, history_page, connection_message, *form_args):
            if not project_id or not run_id:
                raise ValueError("\u8bf7\u5148\u9009\u4e2d\u4e00\u6761\u5386\u53f2\u8bb0\u5f55\u3002")
            form = _collect_form_inputs(*form_args)
            SERVICE.delete_run(project_id, run_id)
            return _render_page(
                project_id,
                search_text,
                history_page,
                "",
                connection_message,
                "\u5386\u53f2\u8bb0\u5f55\u5df2\u5220\u9664",
                _form_override_from_inputs(form),
            )

        def on_open_run_dir(project_id, run_id):
            if not project_id or not run_id:
                raise ValueError("\u8bf7\u5148\u9009\u4e2d\u4e00\u6761\u5386\u53f2\u8bb0\u5f55\u3002")
            summary, _, _ = SERVICE.get_run(project_id, run_id)
            _open_directory(summary.run_dir)
            return "\u5df2\u6253\u5f00\u8f93\u51fa\u76ee\u5f55\u3002"

        demo.load(wrapped(on_load), outputs=page_outputs)
        project_search.change(
            wrapped(on_search_change),
            inputs=[project_search, selected_project_id, connection_message_state, *action_form_inputs],
            outputs=page_outputs,
        )
        project_list.change(
            wrapped(on_project_change),
            inputs=[project_list, project_search],
            outputs=page_outputs,
        )
        create_project_button.click(
            wrapped(on_create),
            inputs=[project_name, project_search],
            outputs=page_outputs,
        )
        save_project_button.click(
            wrapped(on_save),
            inputs=[selected_project_id, project_search, history_page_state, *action_form_inputs],
            outputs=page_outputs,
        )
        delete_project_button.click(
            wrapped(on_delete_project),
            inputs=[selected_project_id, project_search],
            outputs=page_outputs,
        )
        test_connection_button.click(
            wrapped(on_test_connection),
            inputs=[selected_project_id, project_search, history_page_state, *action_form_inputs],
            outputs=page_outputs,
        )
        generate_button.click(
            wrapped(on_generate),
            inputs=[selected_project_id, project_search, connection_message_state, *action_form_inputs],
            outputs=page_outputs,
        )
        previous_page_button.click(
            wrapped(on_previous_page),
            inputs=[selected_project_id, project_search, history_page_state, connection_message_state, *action_form_inputs],
            outputs=page_outputs,
        )
        next_page_button.click(
            wrapped(on_next_page),
            inputs=[selected_project_id, project_search, history_page_state, connection_message_state, *action_form_inputs],
            outputs=page_outputs,
        )
        history_list.change(
            wrapped(on_history_change),
            inputs=[
                history_list,
                selected_project_id,
                project_search,
                history_page_state,
                connection_message_state,
                *action_form_inputs,
            ],
            outputs=page_outputs,
        )
        load_run_button.click(
            wrapped(on_load_run),
            inputs=[selected_project_id, selected_run_id, project_search, history_page_state],
            outputs=page_outputs,
        )
        delete_run_button.click(
            wrapped(on_delete_run),
            inputs=[
                selected_project_id,
                selected_run_id,
                project_search,
                history_page_state,
                connection_message_state,
                *action_form_inputs,
            ],
            outputs=page_outputs,
        )
        open_run_dir_button.click(
            wrapped(on_open_run_dir),
            inputs=[selected_project_id, selected_run_id],
            outputs=[project_feedback],
        )
        transcript_preset.change(
            fn=lambda preset_name: _load_preset_text(preset_name, SERVICE.transcript_presets),
            inputs=[transcript_preset],
            outputs=[transcript_text],
        )
        scene_preset.change(
            fn=lambda preset_name: _load_preset_text(preset_name, SERVICE.scene_presets),
            inputs=[scene_preset],
            outputs=[scene_text],
        )
        transcript_file.change(
            fn=_load_uploaded_text,
            inputs=[transcript_file],
            outputs=[transcript_text],
        )
        backend_mode.change(
            fn=_toggle_backend_groups,
            inputs=[backend_mode],
            outputs=[remote_group, local_group],
        )
        clear_reference_audio_button.click(
            fn=lambda: ("", None, None, "\u5df2\u6e05\u7a7a\u5f53\u524d\u53c2\u8003\u97f3\u9891\uff0c\u70b9\u201c\u4fdd\u5b58\u9879\u76ee\u53c2\u6570\u201d\u540e\u624d\u4f1a\u6301\u4e45\u5316\u3002"),
            outputs=[saved_reference_audio_path, saved_reference_audio, ref_audio_upload, project_feedback],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, inbrowser=IN_BROWSER)
