from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .types import ProjectConfig, ProjectRecord, RunRecordSummary


def _read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ProjectStore:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.projects_root = output_root / "projects"
        self.index_path = self.projects_root / "index.json"
        self._lock = threading.Lock()
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.projects_root.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            _write_json(self.index_path, [])

    def _project_dir(self, project_id: str) -> Path:
        return self.projects_root / project_id

    def _project_file(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "project.json"

    def _runs_root(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "runs"

    def _runs_index_path(self, project_id: str) -> Path:
        return self._runs_root(project_id) / "index.json"

    def _assets_root(self, project_id: str) -> Path:
        return self._project_dir(project_id) / "assets"

    def _read_project_index(self) -> list[ProjectRecord]:
        items = _read_json(self.index_path, [])
        return [ProjectRecord.from_dict(item) for item in items]

    def _write_project_index(self, items: list[ProjectRecord]) -> None:
        _write_json(self.index_path, [item.to_dict() for item in items])

    def _read_runs_index(self, project_id: str) -> list[RunRecordSummary]:
        items = _read_json(self._runs_index_path(project_id), [])
        return [RunRecordSummary.from_dict(item) for item in items]

    def _write_runs_index(self, project_id: str, items: list[RunRecordSummary]) -> None:
        _write_json(self._runs_index_path(project_id), [item.to_dict() for item in items])

    def list_projects(self, search: str = "") -> list[ProjectRecord]:
        with self._lock:
            records = self._read_project_index()
        query = (search or "").strip().lower()
        if query:
            records = [record for record in records if query in record.name.lower()]
        return sorted(
            records,
            key=lambda record: record.last_run_at or record.updated_at or record.created_at,
            reverse=True,
        )

    def create_project(self, name: str | None) -> ProjectConfig:
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            project_id = f"{timestamp}-{uuid4().hex[:8]}"
            display_name = (name or "").strip() or f"新项目 {datetime.now().strftime('%m-%d %H:%M')}"
            now = _now_iso()
            project = ProjectConfig(
                id=project_id,
                name=display_name,
                created_at=now,
                updated_at=now,
            )
            project_dir = self._project_dir(project_id)
            self._assets_root(project_id).mkdir(parents=True, exist_ok=True)
            self._runs_root(project_id).mkdir(parents=True, exist_ok=True)
            _write_json(self._project_file(project_id), project.to_dict())
            self._write_runs_index(project_id, [])

            records = self._read_project_index()
            records.insert(0, ProjectRecord.from_project(project))
            self._write_project_index(records)
            return project

    def get_project(self, project_id: str) -> ProjectConfig:
        project_path = self._project_file(project_id)
        if not project_path.exists():
            raise FileNotFoundError(f"Project {project_id} does not exist.")
        return ProjectConfig.from_dict(_read_json(project_path, {}))

    def save_project(self, project: ProjectConfig) -> ProjectConfig:
        with self._lock:
            project.updated_at = _now_iso()
            _write_json(self._project_file(project.id), project.to_dict())
            records = self._read_project_index()
            updated = False
            for index, record in enumerate(records):
                if record.id == project.id:
                    records[index] = ProjectRecord.from_project(project)
                    updated = True
                    break
            if not updated:
                records.insert(0, ProjectRecord.from_project(project))
            self._write_project_index(records)
        return project

    def delete_project(self, project_id: str) -> None:
        with self._lock:
            project_dir = self._project_dir(project_id)
            if project_dir.exists():
                shutil.rmtree(project_dir)
            records = [record for record in self._read_project_index() if record.id != project_id]
            self._write_project_index(records)

    def project_asset_path(self, project_id: str, relative_path: str | None) -> Path | None:
        if not relative_path:
            return None
        return self._project_dir(project_id) / relative_path

    def save_project_asset(self, project_id: str, source_path: str | Path, asset_name: str) -> str:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Asset source {source} does not exist.")
        suffix = source.suffix.lower() or ".bin"
        asset_dir = self._assets_root(project_id)
        asset_dir.mkdir(parents=True, exist_ok=True)
        destination = asset_dir / f"{asset_name}{suffix}"
        shutil.copy2(source, destination)
        return destination.relative_to(self._project_dir(project_id)).as_posix()

    def create_run_dir(self, project_id: str) -> tuple[str, Path]:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}-{uuid4().hex[:8]}"
        run_dir = self._runs_root(project_id) / run_id
        (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    def save_run(
        self,
        project_id: str,
        run_id: str,
        request_payload: dict,
        result_payload: dict,
        summary: RunRecordSummary,
    ) -> None:
        with self._lock:
            run_dir = self._runs_root(project_id) / run_id
            _write_json(run_dir / "request.json", request_payload)
            _write_json(run_dir / "result.json", result_payload)

            runs = [item for item in self._read_runs_index(project_id) if item.run_id != run_id]
            runs.insert(0, summary)
            self._write_runs_index(project_id, runs)

            project = self.get_project(project_id)
            project.last_run_at = summary.created_at
            project.updated_at = _now_iso()
            _write_json(self._project_file(project_id), project.to_dict())

            records = self._read_project_index()
            for index, record in enumerate(records):
                if record.id == project.id:
                    records[index] = ProjectRecord.from_project(project)
                    break
            self._write_project_index(records)

    def list_runs(self, project_id: str, page: int, page_size: int) -> tuple[list[RunRecordSummary], int, int]:
        with self._lock:
            runs = self._read_runs_index(project_id)
        total_count = len(runs)
        total_pages = max(1, (total_count + page_size - 1) // page_size)
        safe_page = min(max(page, 1), total_pages)
        start = (safe_page - 1) * page_size
        end = start + page_size
        return runs[start:end], total_pages, total_count

    def get_run(self, project_id: str, run_id: str) -> tuple[RunRecordSummary, dict, dict]:
        runs = self._read_runs_index(project_id)
        summary = next((item for item in runs if item.run_id == run_id), None)
        if summary is None:
            raise FileNotFoundError(f"Run {run_id} does not exist.")
        run_dir = self._runs_root(project_id) / run_id
        request_payload = _read_json(run_dir / "request.json", {})
        result_payload = _read_json(run_dir / "result.json", {})
        return summary, request_payload, result_payload

    def delete_run(self, project_id: str, run_id: str) -> None:
        with self._lock:
            run_dir = self._runs_root(project_id) / run_id
            if run_dir.exists():
                shutil.rmtree(run_dir)
            runs = [item for item in self._read_runs_index(project_id) if item.run_id != run_id]
            self._write_runs_index(project_id, runs)
