from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from boson_multimodal.webui.service import WebUIService, normalize_text_for_generation
from boson_multimodal.webui.types import BackendConfig, RunRecordSummary


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        frames = bytearray()
        for index in range(240):
            sample = int(1200 * math.sin(index / 12))
            frames.extend(int(sample).to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(bytes(frames))


class WebUIServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "webui"
        self.service = WebUIService(output_dir=self.output_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_project_save_roundtrip_and_legacy_outputs_are_ignored(self) -> None:
        legacy_dir = self.output_dir / "20260310-legacy-run"
        legacy_dir.mkdir(parents=True, exist_ok=True)

        project = self.service.create_project("Demo Project")
        ref_audio = Path(self.temp_dir.name) / "inputs" / "voice.wav"
        write_wav(ref_audio)

        form = {
            "project_name": "Persisted Project",
            "backend_mode": "local",
            "remote_base_url": BackendConfig().base_url,
            "remote_model_name": BackendConfig().model_name,
            "remote_api_key": "",
            "remote_timeout_seconds": 120,
            "device": "auto",
            "use_static_kv_cache": True,
            "task_mode": "smart_voice",
            "transcript_preset": None,
            "transcript_text": "hello project",
            "scene_preset": None,
            "scene_text": "quiet room",
            "voice_presets": [],
            "saved_reference_audio_path": str(ref_audio),
            "reference_audio_upload_path": None,
            "reference_audio_transcript": "hello reference",
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "seed": 7,
            "chunk_method": "none",
            "chunk_max_word_num": 200,
            "chunk_max_num_turns": 1,
        }

        self.service.save_project_form(project.id, form)
        reloaded = self.service.get_project(project.id)
        form_values = self.service.project_to_form_values(reloaded)

        self.assertEqual(reloaded.name, "Persisted Project")
        self.assertEqual(reloaded.generation_defaults.transcript_text, "hello project")
        self.assertTrue(reloaded.generation_defaults.reference_audio_asset)
        self.assertTrue(Path(form_values["saved_reference_audio_path"]).exists())
        self.assertEqual(len(self.service.list_projects()), 1)

    def test_generation_request_copies_inputs_and_run_snapshot_loads_back(self) -> None:
        project = self.service.create_project("Voice Clone")
        project.backend.mode = "remote_vllm"
        project.backend.api_key = "top-secret"
        self.service.store.save_project(project)

        transcript_file = Path(self.temp_dir.name) / "inputs" / "script.txt"
        transcript_file.parent.mkdir(parents=True, exist_ok=True)
        transcript_file.write_text("hello from file", encoding="utf-8")
        ref_audio = Path(self.temp_dir.name) / "inputs" / "voice.wav"
        write_wav(ref_audio)

        form = {
            "project_name": project.name,
            "backend_mode": "remote_vllm",
            "remote_base_url": "http://127.0.0.1:8000/v1",
            "remote_model_name": "demo-model",
            "remote_api_key": "top-secret",
            "remote_timeout_seconds": 120,
            "device": "auto",
            "use_static_kv_cache": True,
            "task_mode": "voice_clone",
            "transcript_preset": None,
            "transcript_file_path": str(transcript_file),
            "transcript_text": "",
            "scene_preset": None,
            "scene_text": "",
            "voice_presets": [],
            "saved_reference_audio_path": "",
            "reference_audio_upload_path": str(ref_audio),
            "reference_audio_transcript": "This is my reference.",
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "max_new_tokens": 256,
            "seed": 1234,
            "chunk_method": "none",
            "chunk_max_word_num": 200,
            "chunk_max_num_turns": 1,
        }

        run_id, run_dir = self.service.store.create_run_dir(project.id)
        request = self.service._build_generation_request(
            project_id=project.id,
            project_name=project.name,
            run_dir=run_dir,
            backend=self.service._build_backend_config(form),
            form=form,
        )

        self.assertTrue((run_dir / "inputs" / "transcript.txt").exists())
        self.assertTrue((run_dir / "inputs" / "reference_audio.wav").exists())

        request_payload = request.to_dict()
        request_payload["backend"]["api_key"] = "***"
        result_payload = {
            "status": "completed",
            "run_dir": str(run_dir),
            "output_audio_path": str(run_dir / "output.wav"),
            "sampling_rate": 24000,
            "generated_text": "",
            "normalized_transcript": "hello from file.",
            "elapsed_seconds": 1.5,
        }
        summary = RunRecordSummary(
            run_id=run_id,
            created_at="2026-03-10T18:00:00",
            task_mode="voice_clone",
            status="completed",
            title="hello from file",
            audio_path=result_payload["output_audio_path"],
            elapsed_seconds=1.5,
            run_dir=str(run_dir),
        )
        self.service.store.save_run(project.id, run_id, request_payload, result_payload, summary)

        form_values = self.service.run_to_form_values(project.id, run_id)
        self.assertEqual(form_values["transcript_text"], "hello from file")
        self.assertTrue(form_values["saved_reference_audio_path"].endswith("reference_audio.wav"))
        self.assertEqual(form_values["remote_api_key"], "top-secret")

    def test_remote_message_builder_and_history_pagination(self) -> None:
        project = self.service.create_project("Remote Project")
        backend = BackendConfig(mode="remote_vllm", base_url="http://127.0.0.1:8000/v1", model_name="demo-model")

        transcript = "[SPEAKER0] Hello.\n[SPEAKER1] Hi there."
        normalized, speaker_tags = normalize_text_for_generation(transcript)
        request = self.service._build_generation_request(
            project_id=project.id,
            project_name=project.name,
            run_dir=self.service.store.create_run_dir(project.id)[1],
            backend=backend,
            form={
                "task_mode": "multi_speaker",
                "transcript_preset": None,
                "transcript_text": transcript,
                "scene_preset": None,
                "scene_text": "Podcast discussion",
                "voice_presets": [],
                "saved_reference_audio_path": "",
                "reference_audio_upload_path": None,
                "reference_audio_transcript": "",
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "max_new_tokens": 256,
                "seed": 1,
                "chunk_method": "speaker",
                "chunk_max_word_num": 200,
                "chunk_max_num_turns": 1,
            },
        )

        messages = self.service._build_remote_messages(request, normalized, speaker_tags)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("SPEAKER0: feminine", messages[0]["content"])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Hello.", messages[-1]["content"])

        for index in range(12):
            run_id, run_dir = self.service.store.create_run_dir(project.id)
            summary = RunRecordSummary(
                run_id=run_id,
                created_at=f"2026-03-10T18:{index:02d}:00",
                task_mode="smart_voice",
                status="completed",
                title=f"run-{index}",
                audio_path=None,
                elapsed_seconds=0.5,
                run_dir=str(run_dir),
            )
            self.service.store.save_run(
                project.id,
                run_id,
                {"task_mode": "smart_voice"},
                {"status": "completed", "run_dir": str(run_dir), "elapsed_seconds": 0.5},
                summary,
            )

        runs_page_1, total_pages, total_count = self.service.list_runs(project.id, 1)
        runs_page_2, _, _ = self.service.list_runs(project.id, 2)
        self.assertEqual(total_pages, 2)
        self.assertEqual(total_count, 12)
        self.assertEqual(len(runs_page_1), 10)
        self.assertEqual(len(runs_page_2), 2)


if __name__ == "__main__":
    unittest.main()
