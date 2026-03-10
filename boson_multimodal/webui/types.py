from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_REMOTE_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_REMOTE_MODEL_NAME = "higgs-audio-v2-generation-3B-base"


def _copy_dict(data: dict[str, Any] | None) -> dict[str, Any]:
    return dict(data or {})


@dataclass
class BackendConfig:
    mode: str = "local"
    base_url: str = DEFAULT_REMOTE_BASE_URL
    model_name: str = DEFAULT_REMOTE_MODEL_NAME
    api_key: str = ""
    timeout_seconds: int = 120
    device: str = "auto"
    use_static_kv_cache: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BackendConfig":
        values = _copy_dict(data)
        return cls(
            mode=values.get("mode", "local"),
            base_url=values.get("base_url", DEFAULT_REMOTE_BASE_URL),
            model_name=values.get("model_name", DEFAULT_REMOTE_MODEL_NAME),
            api_key=values.get("api_key", ""),
            timeout_seconds=int(values.get("timeout_seconds", 120)),
            device=values.get("device", "auto"),
            use_static_kv_cache=bool(values.get("use_static_kv_cache", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationDefaults:
    task_mode: str = "smart_voice"
    transcript_text: str = ""
    transcript_preset: str | None = None
    scene_text: str = ""
    scene_preset: str | None = None
    voice_presets: list[str] = field(default_factory=list)
    reference_audio_asset: str | None = None
    reference_audio_transcript: str = ""
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 256
    seed: int | None = 1234
    chunk_method: str = "none"
    chunk_max_word_num: int = 200
    chunk_max_num_turns: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationDefaults":
        values = _copy_dict(data)
        return cls(
            task_mode=values.get("task_mode", "smart_voice"),
            transcript_text=values.get("transcript_text", ""),
            transcript_preset=values.get("transcript_preset"),
            scene_text=values.get("scene_text", ""),
            scene_preset=values.get("scene_preset"),
            voice_presets=list(values.get("voice_presets") or []),
            reference_audio_asset=values.get("reference_audio_asset"),
            reference_audio_transcript=values.get("reference_audio_transcript", ""),
            temperature=float(values.get("temperature", 0.8)),
            top_k=int(values.get("top_k", 50)),
            top_p=float(values.get("top_p", 0.95)),
            max_new_tokens=int(values.get("max_new_tokens", 256)),
            seed=int(values["seed"]) if values.get("seed") not in (None, "") else None,
            chunk_method=values.get("chunk_method", "none") or "none",
            chunk_max_word_num=int(values.get("chunk_max_word_num", 200)),
            chunk_max_num_turns=int(values.get("chunk_max_num_turns", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectConfig:
    id: str
    name: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    last_run_at: str | None = None
    backend: BackendConfig = field(default_factory=BackendConfig)
    generation_defaults: GenerationDefaults = field(default_factory=GenerationDefaults)
    project_assets: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        values = _copy_dict(data)
        return cls(
            id=values["id"],
            name=values.get("name", "未命名项目"),
            description=values.get("description", ""),
            created_at=values.get("created_at", ""),
            updated_at=values.get("updated_at", ""),
            last_run_at=values.get("last_run_at"),
            backend=BackendConfig.from_dict(values.get("backend")),
            generation_defaults=GenerationDefaults.from_dict(values.get("generation_defaults")),
            project_assets=_copy_dict(values.get("project_assets")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectRecord:
    id: str
    name: str
    created_at: str
    updated_at: str
    last_run_at: str | None
    backend_mode: str

    @classmethod
    def from_project(cls, project: ProjectConfig) -> "ProjectRecord":
        return cls(
            id=project.id,
            name=project.name,
            created_at=project.created_at,
            updated_at=project.updated_at,
            last_run_at=project.last_run_at,
            backend_mode=project.backend.mode,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectRecord":
        values = _copy_dict(data)
        return cls(
            id=values["id"],
            name=values.get("name", "未命名项目"),
            created_at=values.get("created_at", ""),
            updated_at=values.get("updated_at", ""),
            last_run_at=values.get("last_run_at"),
            backend_mode=values.get("backend_mode", "local"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunRecordSummary:
    run_id: str
    created_at: str
    task_mode: str
    status: str
    title: str
    audio_path: str | None
    elapsed_seconds: float | None
    run_dir: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecordSummary":
        values = _copy_dict(data)
        elapsed = values.get("elapsed_seconds")
        return cls(
            run_id=values["run_id"],
            created_at=values.get("created_at", ""),
            task_mode=values.get("task_mode", "smart_voice"),
            status=values.get("status", "unknown"),
            title=values.get("title", ""),
            audio_path=values.get("audio_path"),
            elapsed_seconds=float(elapsed) if elapsed is not None else None,
            run_dir=values.get("run_dir", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRequest:
    project_id: str
    project_name: str
    backend: BackendConfig
    task_mode: str
    transcript_text: str
    transcript_preset: str | None
    transcript_file_path: str | None
    scene_text: str
    scene_preset: str | None
    voice_presets: list[str]
    reference_audio_path: str | None
    reference_audio_transcript: str
    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int
    seed: int | None
    chunk_method: str
    chunk_max_word_num: int
    chunk_max_num_turns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    status: str
    run_dir: str
    output_audio_path: str | None
    sampling_rate: int | None
    generated_text: str
    normalized_transcript: str
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    ok: bool
    message: str
    available_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
