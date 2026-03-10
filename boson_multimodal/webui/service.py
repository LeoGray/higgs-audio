from __future__ import annotations

import base64
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import certifi_win32  # noqa: F401
except ImportError:
    certifi_win32 = None

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.data_types import AudioContent, Message
from examples.generation import (
    HiggsAudioModelClient,
    MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE,
    normalize_chinese_punctuation,
    prepare_chunk_text,
)

from .storage import ProjectStore
from .types import (
    BackendConfig,
    GenerationDefaults,
    GenerationRequest,
    GenerationResult,
    ProjectConfig,
    ProjectRecord,
    RunRecordSummary,
    ValidationResult,
)


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

VOICE_DIR = ROOT / "examples" / "voice_prompts"
TRANSCRIPT_DIR = ROOT / "examples" / "transcript"
SCENE_DIR = ROOT / "examples" / "scene_prompts"
OUTPUT_DIR = ROOT / "outputs" / "webui"
LOCAL_MODEL_DIR = ROOT / "local_models" / "higgs-audio-v2-generation-3B-base"
LOCAL_TOKENIZER_DIR = ROOT / "local_models" / "higgs-audio-v2-tokenizer"

MODEL_REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
TOKENIZER_REQUIRED_FILES = [
    "config.json",
    "model.pth",
]

NONE_OPTION = "__NONE__"
PAGE_SIZE = 10
TASK_LABELS = {
    "smart_voice": "Smart Voice",
    "voice_clone": "Voice Clone",
    "multi_speaker": "Multi Speaker",
}
CHUNK_LABELS = {
    "none": "No Chunking",
    "speaker": "By Speaker",
    "word": "By Word",
}
DEVICE_LABELS = {
    "auto": "Auto",
    "cuda": "CUDA",
    "mps": "MPS",
    "cpu": "CPU",
}

_MODEL_CLIENTS: dict[tuple[str, bool], HiggsAudioModelClient] = {}
_MODEL_LOCK = threading.Lock()


@dataclass
class ReferenceVoice:
    label: str
    transcript: str
    audio_path: str


def list_text_presets(root: Path) -> dict[str, Path]:
    presets: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        if path.name == "profile.yaml":
            continue
        presets[path.relative_to(root).as_posix()] = path
    return presets


def list_voice_presets() -> dict[str, dict[str, str]]:
    presets: dict[str, dict[str, str]] = {}
    for wav_path in sorted(VOICE_DIR.glob("*.wav")):
        text_path = wav_path.with_suffix(".txt")
        transcript = read_text_file(text_path) if text_path.exists() else ""
        presets[wav_path.stem] = {
            "audio_path": str(wav_path),
            "transcript": transcript,
        }
    return presets


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore").strip()


def list_missing_files(root: Path, required_files: list[str]) -> list[str]:
    return [name for name in required_files if not (root / name).exists()]


def real_backend_ready() -> bool:
    return not list_missing_files(LOCAL_MODEL_DIR, MODEL_REQUIRED_FILES) and not list_missing_files(
        LOCAL_TOKENIZER_DIR, TOKENIZER_REQUIRED_FILES
    )


def resolve_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    if device == "mps":
        return "mps"
    if device == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model_client(resolved_device: str, use_static_kv_cache: bool) -> HiggsAudioModelClient:
    cache_key = (resolved_device, use_static_kv_cache)
    with _MODEL_LOCK:
        client = _MODEL_CLIENTS.get(cache_key)
        if client is not None:
            return client

        audio_tokenizer_device = "cpu" if resolved_device == "mps" else resolved_device
        audio_tokenizer = load_higgs_audio_tokenizer(str(LOCAL_TOKENIZER_DIR), device=audio_tokenizer_device)
        static_cache_enabled = use_static_kv_cache and resolved_device.startswith("cuda")
        client = HiggsAudioModelClient(
            model_path=str(LOCAL_MODEL_DIR),
            audio_tokenizer=audio_tokenizer,
            device=resolved_device,
            device_id=0 if resolved_device.startswith("cuda") else None,
            max_new_tokens=2048,
            use_static_kv_cache=static_cache_enabled,
        )
        _MODEL_CLIENTS[cache_key] = client
        return client


def normalize_text_for_generation(transcript: str) -> tuple[str, list[str]]:
    pattern = re.compile(r"\[(SPEAKER\d+)\]")
    transcript = normalize_chinese_punctuation(transcript)
    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("\u00b0F", " degrees Fahrenheit")
    transcript = transcript.replace("\u00b0C", " degrees Celsius")

    replacements = [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]
    for source, target in replacements:
        transcript = transcript.replace(source, target)

    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()]).strip()
    if transcript and not any(transcript.endswith(char) for char in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]):
        transcript += "."

    return transcript, sorted(set(pattern.findall(transcript)))


class WebUIService:
    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or OUTPUT_DIR
        self.store = ProjectStore(self.output_dir)
        self.voice_presets = list_voice_presets()
        self.transcript_presets = list_text_presets(TRANSCRIPT_DIR)
        self.scene_presets = list_text_presets(SCENE_DIR)
        self.page_size = PAGE_SIZE
        self._generation_lock = threading.Lock()

    def list_projects(self, search: str = "") -> list[ProjectRecord]:
        return self.store.list_projects(search)

    def create_project(self, name: str | None) -> ProjectConfig:
        return self.store.create_project(name)

    def delete_project(self, project_id: str) -> None:
        self.store.delete_project(project_id)

    def get_project(self, project_id: str) -> ProjectConfig:
        return self.store.get_project(project_id)

    def list_runs(self, project_id: str, page: int) -> tuple[list[RunRecordSummary], int, int]:
        return self.store.list_runs(project_id, page, self.page_size)

    def get_run(self, project_id: str, run_id: str) -> tuple[RunRecordSummary, dict[str, Any], dict[str, Any]]:
        return self.store.get_run(project_id, run_id)

    def delete_run(self, project_id: str, run_id: str) -> None:
        self.store.delete_run(project_id, run_id)

    def validate_backend(self, backend: BackendConfig) -> ValidationResult:
        if backend.mode == "local":
            if real_backend_ready():
                return ValidationResult(ok=True, message="Local model is ready.")
            missing_model = list_missing_files(LOCAL_MODEL_DIR, MODEL_REQUIRED_FILES)
            missing_tokenizer = list_missing_files(LOCAL_TOKENIZER_DIR, TOKENIZER_REQUIRED_FILES)
            return ValidationResult(
                ok=False,
                message=(
                    f"Local model is not ready: missing {len(missing_model)} model files and "
                    f"{len(missing_tokenizer)} tokenizer files."
                ),
            )

        if not backend.base_url.strip():
            return ValidationResult(ok=False, message="Remote base URL is required.")
        if not backend.model_name.strip():
            return ValidationResult(ok=False, message="Remote model name is required.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for remote_vllm mode.") from exc

        client = OpenAI(
            api_key=backend.api_key or "EMPTY",
            base_url=backend.base_url.rstrip("/"),
            timeout=float(backend.timeout_seconds),
        )
        try:
            models = client.models.list()
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(ok=False, message=f"Remote validation failed: {exc}")

        model_ids = [item.id for item in models.data]
        if backend.model_name not in model_ids:
            return ValidationResult(
                ok=False,
                message=f"Connected, but `{backend.model_name}` is not in the remote model list.",
                available_models=model_ids,
            )
        return ValidationResult(ok=True, message="Remote connection succeeded.", available_models=model_ids)

    def project_choices(self, search: str = "") -> list[tuple[str, str]]:
        records = self.list_projects(search)
        choices = []
        for record in records:
            backend_label = "Local" if record.backend_mode == "local" else "Remote"
            choices.append((f"{record.name} | {backend_label}", record.id))
        return choices

    def history_choices(self, project_id: str, page: int) -> tuple[list[tuple[str, str]], int, int, int]:
        runs, total_pages, total_count = self.list_runs(project_id, page)
        choices = []
        for run in runs:
            label = f"{run.created_at.replace('T', ' ')} | {TASK_LABELS.get(run.task_mode, run.task_mode)} | {run.title}"
            choices.append((label, run.run_id))
        safe_page = min(max(page, 1), total_pages)
        return choices, safe_page, total_pages, total_count

    def project_to_form_values(self, project: ProjectConfig) -> dict[str, Any]:
        defaults = project.generation_defaults
        reference_audio_path = self._absolute_reference_audio_path(project.id, defaults.reference_audio_asset)
        return {
            "project_id": project.id,
            "project_name": project.name,
            "backend_mode": project.backend.mode,
            "remote_base_url": project.backend.base_url,
            "remote_model_name": project.backend.model_name,
            "remote_api_key": project.backend.api_key,
            "remote_timeout_seconds": project.backend.timeout_seconds,
            "device": project.backend.device,
            "use_static_kv_cache": project.backend.use_static_kv_cache,
            "task_mode": defaults.task_mode,
            "transcript_preset": defaults.transcript_preset or NONE_OPTION,
            "transcript_text": defaults.transcript_text,
            "scene_preset": defaults.scene_preset or NONE_OPTION,
            "scene_text": defaults.scene_text,
            "voice_presets": [value for value in defaults.voice_presets if value in self.voice_presets],
            "saved_reference_audio_path": reference_audio_path,
            "reference_audio_transcript": defaults.reference_audio_transcript,
            "temperature": defaults.temperature,
            "top_k": defaults.top_k,
            "top_p": defaults.top_p,
            "max_new_tokens": defaults.max_new_tokens,
            "seed": defaults.seed,
            "chunk_method": defaults.chunk_method,
            "chunk_max_word_num": defaults.chunk_max_word_num,
            "chunk_max_num_turns": defaults.chunk_max_num_turns,
        }

    def run_to_form_values(self, project_id: str, run_id: str) -> dict[str, Any]:
        _, request_payload, _ = self.get_run(project_id, run_id)
        project = self.get_project(project_id)
        backend = BackendConfig.from_dict(request_payload.get("backend"))
        if backend.api_key in ("", "***"):
            backend.api_key = project.backend.api_key
        return {
            "project_id": project_id,
            "project_name": request_payload.get("project_name", ""),
            "backend_mode": backend.mode,
            "remote_base_url": backend.base_url,
            "remote_model_name": backend.model_name,
            "remote_api_key": backend.api_key,
            "remote_timeout_seconds": backend.timeout_seconds,
            "device": backend.device,
            "use_static_kv_cache": backend.use_static_kv_cache,
            "task_mode": request_payload.get("task_mode", "smart_voice"),
            "transcript_preset": request_payload.get("transcript_preset") or NONE_OPTION,
            "transcript_text": request_payload.get("transcript_text", ""),
            "scene_preset": request_payload.get("scene_preset") or NONE_OPTION,
            "scene_text": request_payload.get("scene_text", ""),
            "voice_presets": [
                value for value in list(request_payload.get("voice_presets") or []) if value in self.voice_presets
            ],
            "saved_reference_audio_path": request_payload.get("reference_audio_path"),
            "reference_audio_transcript": request_payload.get("reference_audio_transcript", ""),
            "temperature": float(request_payload.get("temperature", 0.8)),
            "top_k": int(request_payload.get("top_k", 50)),
            "top_p": float(request_payload.get("top_p", 0.95)),
            "max_new_tokens": int(request_payload.get("max_new_tokens", 256)),
            "seed": self._coerce_seed(request_payload.get("seed")),
            "chunk_method": request_payload.get("chunk_method") or "none",
            "chunk_max_word_num": int(request_payload.get("chunk_max_word_num", 200)),
            "chunk_max_num_turns": int(request_payload.get("chunk_max_num_turns", 1)),
        }

    def save_project_form(self, project_id: str, form: dict[str, Any]) -> ProjectConfig:
        if not project_id:
            raise ValueError("Create or select a project first.")
        project = self.get_project(project_id)
        project.name = (form.get("project_name") or project.name).strip() or project.name
        project.backend = self._build_backend_config(form)
        project.generation_defaults = self._build_generation_defaults(project, form)
        return self.store.save_project(project)

    def generate_for_project(self, project_id: str, form: dict[str, Any]) -> tuple[GenerationResult, dict[str, Any]]:
        if not project_id:
            raise ValueError("Create or select a project first.")
        project = self.get_project(project_id)
        project_name = (form.get("project_name") or project.name).strip() or project.name
        backend = self._build_backend_config(form)

        if backend.mode == "local" and not real_backend_ready():
            raise ValueError("Local model is not ready. Use remote_vllm or install the local model files first.")

        run_id, run_dir = self.store.create_run_dir(project_id)
        request = self._build_generation_request(
            project_id=project_id,
            project_name=project_name,
            run_dir=run_dir,
            backend=backend,
            form=form,
        )

        if not self._generation_lock.acquire(blocking=False):
            raise ValueError("Another generation job is already running.")
        try:
            started_at = time.perf_counter()
            if backend.mode == "local":
                result = self._run_local_generation(request, run_dir)
            else:
                result = self._run_remote_generation(request, run_dir)
            result.elapsed_seconds = time.perf_counter() - started_at
        except Exception:  # noqa: BLE001
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
        finally:
            self._generation_lock.release()

        request_payload = request.to_dict()
        if request_payload["backend"].get("api_key"):
            request_payload["backend"]["api_key"] = "***"
        request_payload.update(
            {
                "project_name": project_name,
                "normalized_transcript": result.normalized_transcript,
                "scene_text": request.scene_text,
                "resolved_device": resolve_device(backend.device) if backend.mode == "local" else None,
            }
        )
        result_payload = result.to_dict()
        result_payload["run_dir"] = str(run_dir)
        result_payload["output_audio_path"] = result.output_audio_path

        summary = RunRecordSummary(
            run_id=run_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            task_mode=request.task_mode,
            status=result.status,
            title=self._build_run_title(request.transcript_text, request.task_mode),
            audio_path=result.output_audio_path,
            elapsed_seconds=result.elapsed_seconds,
            run_dir=str(run_dir),
        )
        self.store.save_run(project_id, run_id, request_payload, result_payload, summary)
        return result, request_payload

    def _build_backend_config(self, form: dict[str, Any]) -> BackendConfig:
        return BackendConfig(
            mode=(form.get("backend_mode") or "local").strip(),
            base_url=(form.get("remote_base_url") or "").strip() or BackendConfig().base_url,
            model_name=(form.get("remote_model_name") or "").strip() or BackendConfig().model_name,
            api_key=(form.get("remote_api_key") or "").strip(),
            timeout_seconds=max(1, int(form.get("remote_timeout_seconds") or 120)),
            device=(form.get("device") or "auto").strip(),
            use_static_kv_cache=bool(form.get("use_static_kv_cache", True)),
        )

    def _build_generation_defaults(self, project: ProjectConfig, form: dict[str, Any]) -> GenerationDefaults:
        reference_audio_asset = self._persist_project_reference_audio(project.id, form.get("saved_reference_audio_path"))
        if form.get("reference_audio_upload_path"):
            reference_audio_asset = self.store.save_project_asset(
                project.id,
                form["reference_audio_upload_path"],
                "reference_audio",
            )
        if not reference_audio_asset:
            project.project_assets.pop("reference_audio", None)
        else:
            project.project_assets["reference_audio"] = reference_audio_asset
        return GenerationDefaults(
            task_mode=form.get("task_mode", "smart_voice"),
            transcript_text=(form.get("transcript_text") or "").strip(),
            transcript_preset=self._clean_choice(form.get("transcript_preset")),
            scene_text=(form.get("scene_text") or "").strip(),
            scene_preset=self._clean_choice(form.get("scene_preset")),
            voice_presets=list(form.get("voice_presets") or []),
            reference_audio_asset=reference_audio_asset,
            reference_audio_transcript=(form.get("reference_audio_transcript") or "").strip(),
            temperature=float(form.get("temperature") or 0.8),
            top_k=int(form.get("top_k") or 50),
            top_p=float(form.get("top_p") or 0.95),
            max_new_tokens=int(form.get("max_new_tokens") or 256),
            seed=self._coerce_seed(form.get("seed")),
            chunk_method=form.get("chunk_method") or "none",
            chunk_max_word_num=int(form.get("chunk_max_word_num") or 200),
            chunk_max_num_turns=int(form.get("chunk_max_num_turns") or 1),
        )

    def _build_generation_request(
        self,
        project_id: str,
        project_name: str,
        run_dir: Path,
        backend: BackendConfig,
        form: dict[str, Any],
    ) -> GenerationRequest:
        transcript_file_path = None
        if form.get("transcript_file_path"):
            transcript_file_path = str(self._copy_input_file(run_dir, form["transcript_file_path"], "transcript"))

        reference_audio_path = None
        source_reference_audio = form.get("reference_audio_upload_path") or form.get("saved_reference_audio_path")
        if source_reference_audio:
            reference_audio_path = str(self._copy_input_file(run_dir, source_reference_audio, "reference_audio"))

        transcript_preset = self._clean_choice(form.get("transcript_preset"))
        scene_preset = self._clean_choice(form.get("scene_preset"))
        transcript_text = self._resolve_text_input(
            direct_text=form.get("transcript_text"),
            upload_path=transcript_file_path,
            preset_name=transcript_preset,
            presets=self.transcript_presets,
        )
        scene_text = self._resolve_text_input(
            direct_text=form.get("scene_text"),
            upload_path=None,
            preset_name=scene_preset,
            presets=self.scene_presets,
        )

        request = GenerationRequest(
            project_id=project_id,
            project_name=project_name,
            backend=backend,
            task_mode=form.get("task_mode", "smart_voice"),
            transcript_text=transcript_text,
            transcript_preset=transcript_preset,
            transcript_file_path=transcript_file_path,
            scene_text=scene_text,
            scene_preset=scene_preset,
            voice_presets=list(form.get("voice_presets") or []),
            reference_audio_path=reference_audio_path,
            reference_audio_transcript=(form.get("reference_audio_transcript") or "").strip(),
            temperature=float(form.get("temperature") or 0.8),
            top_k=int(form.get("top_k") or 50),
            top_p=float(form.get("top_p") or 0.95),
            max_new_tokens=int(form.get("max_new_tokens") or 256),
            seed=self._coerce_seed(form.get("seed")),
            chunk_method=form.get("chunk_method") or "none",
            chunk_max_word_num=int(form.get("chunk_max_word_num") or 200),
            chunk_max_num_turns=int(form.get("chunk_max_num_turns") or 1),
        )
        self._validate_request(request)
        return request

    def _validate_request(self, request: GenerationRequest) -> None:
        if not request.transcript_text.strip():
            raise ValueError("Provide transcript text, upload a transcript file, or choose a preset.")
        if request.task_mode in {"smart_voice", "voice_clone"} and len(request.voice_presets) > 1:
            raise ValueError("Single-speaker modes only support one preset voice.")
        if request.reference_audio_path and request.task_mode != "voice_clone":
            raise ValueError("Uploaded reference audio is only supported in voice_clone mode.")
        if request.reference_audio_path and not request.reference_audio_transcript.strip():
            raise ValueError("Reference transcript is required when uploading custom reference audio.")
        if request.task_mode == "voice_clone" and not request.voice_presets and not request.reference_audio_path:
            raise ValueError("voice_clone mode needs a preset voice or uploaded reference audio.")

        normalized_transcript, speaker_tags = normalize_text_for_generation(request.transcript_text)
        if request.task_mode == "multi_speaker":
            reference_count = len(request.voice_presets)
            if reference_count == 1:
                raise ValueError("multi_speaker mode needs at least 2 preset voices, or none for auto assignment.")
            if reference_count and speaker_tags and reference_count < len(speaker_tags):
                raise ValueError("Preset voice count is smaller than the speaker tag count.")

        if request.backend.mode == "remote_vllm":
            if not request.backend.base_url.strip():
                raise ValueError("Remote base URL is required.")
            if not request.backend.model_name.strip():
                raise ValueError("Remote model name is required.")
        elif request.backend.mode != "local":
            raise ValueError(f"Unsupported backend mode: {request.backend.mode}")

    def _run_local_generation(self, request: GenerationRequest, run_dir: Path) -> GenerationResult:
        normalized_transcript, speaker_tags = normalize_text_for_generation(request.transcript_text)
        resolved_device = resolve_device(request.backend.device)
        model_client = get_model_client(resolved_device, request.backend.use_static_kv_cache)
        model_client._max_new_tokens = int(request.max_new_tokens)
        messages, audio_ids = self._build_local_messages(request, model_client._audio_tokenizer, speaker_tags)
        chunked_text = prepare_chunk_text(
            normalized_transcript,
            chunk_method=None if request.chunk_method == "none" else request.chunk_method,
            chunk_max_word_num=int(request.chunk_max_word_num),
            chunk_max_num_turns=int(request.chunk_max_num_turns),
        )
        waveform, sampling_rate, generated_text = model_client.generate(
            messages=messages,
            audio_ids=audio_ids,
            chunked_text=chunked_text,
            generation_chunk_buffer_size=None,
            temperature=float(request.temperature),
            top_k=int(request.top_k),
            top_p=float(request.top_p),
            seed=request.seed,
        )
        output_audio_path = run_dir / "output.wav"
        sf.write(output_audio_path, waveform, sampling_rate)
        return GenerationResult(
            status="completed",
            run_dir=str(run_dir),
            output_audio_path=str(output_audio_path),
            sampling_rate=sampling_rate,
            generated_text=generated_text,
            normalized_transcript=normalized_transcript,
            elapsed_seconds=0.0,
        )

    def _run_remote_generation(self, request: GenerationRequest, run_dir: Path) -> GenerationResult:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for remote_vllm mode.") from exc

        normalized_transcript, speaker_tags = normalize_text_for_generation(request.transcript_text)
        messages = self._build_remote_messages(request, normalized_transcript, speaker_tags)
        client = OpenAI(
            api_key=request.backend.api_key or "EMPTY",
            base_url=request.backend.base_url.rstrip("/"),
            timeout=float(request.backend.timeout_seconds),
        )
        request_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": request.backend.model_name,
            "modalities": ["text", "audio"],
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "max_completion_tokens": int(request.max_new_tokens),
            "extra_body": {"top_k": int(request.top_k)},
            "stop": ["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
        }
        if request.seed is not None:
            request_kwargs["seed"] = int(request.seed)
        response = client.chat.completions.create(**request_kwargs)
        choice = response.choices[0].message
        audio_bytes = base64.b64decode(choice.audio.data)
        output_audio_path = run_dir / "output.wav"
        output_audio_path.write_bytes(audio_bytes)
        return GenerationResult(
            status="completed",
            run_dir=str(run_dir),
            output_audio_path=str(output_audio_path),
            sampling_rate=24000,
            generated_text=choice.content or "",
            normalized_transcript=normalized_transcript,
            elapsed_seconds=0.0,
        )

    def _build_local_messages(
        self,
        request: GenerationRequest,
        audio_tokenizer,
        speaker_tags: list[str],
    ) -> tuple[list[Message], list[Any]]:
        references = self._build_reference_voices(request)
        messages: list[Message] = []
        audio_ids: list[Any] = []
        system_prompt = self._build_system_prompt(request.task_mode, request.scene_text, speaker_tags, bool(references))
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        for index, reference in enumerate(references):
            prefix = f"[SPEAKER{index}] " if request.task_mode == "multi_speaker" else ""
            messages.append(Message(role="user", content=f"{prefix}{reference.transcript}".strip()))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=reference.audio_path)))
            audio_ids.append(audio_tokenizer.encode(reference.audio_path))
        return messages, audio_ids

    def _build_remote_messages(
        self,
        request: GenerationRequest,
        normalized_transcript: str,
        speaker_tags: list[str],
    ) -> list[dict[str, Any]]:
        references = self._build_reference_voices(request)
        messages: list[dict[str, Any]] = []
        system_prompt = self._build_system_prompt(request.task_mode, request.scene_text, speaker_tags, bool(references))
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for index, reference in enumerate(references):
            prefix = f"[SPEAKER{index}] " if request.task_mode == "multi_speaker" else ""
            messages.append({"role": "user", "content": f"{prefix}{reference.transcript}".strip()})
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": self._encode_base64_content(reference.audio_path),
                                "format": Path(reference.audio_path).suffix.lower().lstrip(".") or "wav",
                            },
                        }
                    ],
                }
            )
        messages.append({"role": "user", "content": normalized_transcript})
        return messages

    def _build_reference_voices(self, request: GenerationRequest) -> list[ReferenceVoice]:
        references: list[ReferenceVoice] = []
        if request.reference_audio_path:
            references.append(
                ReferenceVoice(
                    label="custom",
                    transcript=request.reference_audio_transcript.strip(),
                    audio_path=request.reference_audio_path,
                )
            )
        elif request.voice_presets:
            for preset_name in request.voice_presets:
                preset = self.voice_presets.get(preset_name)
                if preset is None:
                    continue
                references.append(
                    ReferenceVoice(
                        label=preset_name,
                        transcript=preset["transcript"],
                        audio_path=preset["audio_path"],
                    )
                )
        return references

    def _build_system_prompt(
        self,
        task_mode: str,
        scene_text: str,
        speaker_tags: list[str],
        has_references: bool,
    ) -> str:
        if task_mode == "multi_speaker":
            scene_sections = []
            if scene_text:
                scene_sections.append(scene_text)
            if not has_references and speaker_tags:
                speaker_lines = []
                for index, tag in enumerate(speaker_tags):
                    speaker_lines.append(f"{tag}: {'feminine' if index % 2 == 0 else 'masculine'}")
                scene_sections.append("\n".join(speaker_lines))
            system_sections = [MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE]
            if scene_sections:
                scene_body = "\n\n".join(scene_sections)
                system_sections.append(f"<|scene_desc_start|>\n{scene_body}\n<|scene_desc_end|>")
            return "\n\n".join(system_sections)

        system_sections = ["Generate audio following instruction."]
        if scene_text:
            system_sections.append(f"<|scene_desc_start|>\n{scene_text}\n<|scene_desc_end|>")
        return "\n\n".join(system_sections)

    def _resolve_text_input(
        self,
        direct_text: str | None,
        upload_path: str | None,
        preset_name: str | None,
        presets: dict[str, Path],
    ) -> str:
        if upload_path:
            return read_text_file(Path(upload_path))
        if preset_name:
            return read_text_file(presets[preset_name])
        return (direct_text or "").strip()

    def _copy_input_file(self, run_dir: Path, source_path: str | Path, prefix: str) -> Path:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Input file does not exist: {source}")
        suffix = source.suffix.lower() or ".bin"
        destination = run_dir / "inputs" / f"{prefix}{suffix}"
        shutil.copy2(source, destination)
        return destination

    def _persist_project_reference_audio(self, project_id: str, source_path: str | None) -> str | None:
        if not source_path:
            return None
        source = Path(source_path)
        if not source.exists():
            return None
        project_dir = self.store.projects_root / project_id
        try:
            return source.relative_to(project_dir).as_posix()
        except ValueError:
            return self.store.save_project_asset(project_id, source, "reference_audio")

    def _absolute_reference_audio_path(self, project_id: str, relative_path: str | None) -> str | None:
        path = self.store.project_asset_path(project_id, relative_path)
        if path is None or not path.exists():
            return None
        return str(path)

    def _clean_choice(self, value: str | None) -> str | None:
        if value in (None, "", NONE_OPTION):
            return None
        return value

    def _coerce_seed(self, value: Any) -> int | None:
        if value in (None, "", "None"):
            return None
        return int(value)

    def _build_run_title(self, transcript_text: str, task_mode: str) -> str:
        first_line = transcript_text.strip().splitlines()[0] if transcript_text.strip() else "Empty Transcript"
        compact = re.sub(r"\s+", " ", first_line)
        return compact[:48] or TASK_LABELS.get(task_mode, task_mode)

    def _encode_base64_content(self, file_path: str) -> str:
        with open(file_path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")
