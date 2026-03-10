from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

import certifi_win32  # noqa: F401
import requests
from huggingface_hub import hf_hub_url
from tqdm import tqdm


MODEL_REPO = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_REPO = "bosonai/higgs-audio-v2-tokenizer"

MODEL_FILES = [
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

TOKENIZER_FILES = [
    "config.json",
    "model.pth",
]

LOG_FILE: Path | None = None


def log(message: str) -> None:
    timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped)
    if LOG_FILE is not None:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def head_file_size(session: requests.Session, url: str) -> int:
    response = session.head(url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    return int(response.headers.get("content-length", "0"))


def download_file(session: requests.Session, repo_id: str, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo_id, filename=filename)
    destination = output_dir / filename
    total_size = head_file_size(session, url)
    existing_size = destination.stat().st_size if destination.exists() else 0

    if total_size and existing_size == total_size:
        log(f"SKIP {destination} ({existing_size} bytes)")
        return

    headers = {}
    mode = "wb"
    progress_initial = 0

    if existing_size and total_size and existing_size < total_size:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"
        progress_initial = existing_size
        log(f"RESUME {destination} from {existing_size}/{total_size}")
    else:
        log(f"DOWNLOAD {destination}")

    with session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(30, 120)) as response:
        if response.status_code == 200 and mode == "ab":
            mode = "wb"
            progress_initial = 0
        response.raise_for_status()
        with open(destination, mode) as handle, tqdm(
            total=total_size or None,
            initial=progress_initial,
            unit="B",
            unit_scale=True,
            desc=filename,
        ) as progress:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))

    final_size = destination.stat().st_size
    if total_size and final_size != total_size:
        raise RuntimeError(f"Size mismatch for {destination}: expected {total_size}, got {final_size}")
    log(f"DONE {destination} ({final_size} bytes)")


def download_repo(session: requests.Session, repo_id: str, filenames: list[str], root_dir: Path) -> None:
    target_dir = root_dir / repo_id.split("/")[-1]
    for filename in filenames:
        download_file(session, repo_id, filename, target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Higgs model files with resume support.")
    parser.add_argument(
        "--root-dir",
        default="local_models",
        help="Root directory used to store downloaded model files.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "model", "tokenizer"],
        default="all",
        help="Limit download scope.",
    )
    parser.add_argument(
        "--log-file",
        default="local_models/download.log",
        help="Path of the log file.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    global LOG_FILE
    LOG_FILE = Path(args.log_file).resolve()

    root_dir = Path(args.root_dir).resolve()
    session = requests.Session()

    if args.only in {"all", "tokenizer"}:
        download_repo(session, TOKENIZER_REPO, TOKENIZER_FILES, root_dir)
    if args.only in {"all", "model"}:
        download_repo(session, MODEL_REPO, MODEL_FILES, root_dir)


if __name__ == "__main__":
    main()
