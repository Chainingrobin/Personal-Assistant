"""Download the Kokoro ONNX model assets into the configured model directory."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

# Allow this file to be run directly as `python scripts/download_tts_models.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TTS_CONFIG


BASE_URL = "https://huggingface.co/hexgrad/Kokoro-82M-v1.0-ONNX/resolve/main"


def download_file(filename: str, destination: Path) -> None:
    if destination.is_file():
        print(f"[tts] Already present: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tts] Downloading {filename}...")
    urlretrieve(f"{BASE_URL}/{filename}", destination)
    print(f"[tts] Saved: {destination}")


def main() -> int:
    try:
        model_path = Path(TTS_CONFIG.model_path)
        voices_path = Path(TTS_CONFIG.voices_path)
        download_file(model_path.name, model_path)
        download_file(voices_path.name, voices_path)
    except Exception as exc:
        print(f"[tts] Download failed: {exc}", file=sys.stderr)
        print("[tts] Download the two files manually from:", file=sys.stderr)
        print(f"      {BASE_URL}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())