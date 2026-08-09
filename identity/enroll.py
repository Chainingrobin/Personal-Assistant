"""Enroll a new speaker profile for ECAPA verification.

Usage:
    python identity/enroll.py robin path\to\robin_clips

The folder should contain 5-10 short WAV recordings of one speaker using
different phrases. The script loads each clip, extracts an ECAPA embedding,
averages the embeddings, and saves the normalized profile vector to
identity/profiles/<user_id>.npy. SpeechBrain downloads the ECAPA weights on the
first run and caches them locally under the Torch/Hugging Face cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("soundfile is required for enrollment") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identity.voice_id import SpeakerVerifier  # noqa: E402


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load one WAV clip as mono float32 audio for embedding extraction."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sample_rate)


def enroll_user(user_id: str, recordings_dir: Path, profile_dir: Path = Path("identity/profiles")) -> Path:
    """Build and save an averaged embedding for one user_id.

    Input: user_id and a directory of WAV files. Output: path to the saved
    .npy profile vector. This is an enrollment-only path, so the one-time cost
    of encoding multiple clips is acceptable.
    """
    verifier = SpeakerVerifier(profile_dir=profile_dir)
    wav_paths = sorted([path for path in recordings_dir.iterdir() if path.suffix.lower() == ".wav"])
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {recordings_dir}")

    embeddings: list[np.ndarray] = []
    for wav_path in wav_paths:
        audio, sample_rate = _load_wav(wav_path)
        embeddings.append(verifier.embed_audio(audio, sample_rate=sample_rate))
        print(f"[enroll] encoded {wav_path.name}")

    profile = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(profile))
    if norm > 0.0:
        profile = profile / norm

    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"{user_id}.npy"
    np.save(profile_path, profile)
    print(f"[enroll] saved profile to {profile_path}")
    return profile_path


def main() -> None:
    """CLI entrypoint for one-time user enrollment."""
    parser = argparse.ArgumentParser(description="Enroll a user by averaging ECAPA embeddings from WAV clips.")
    parser.add_argument("user_id", help="User/profile name to save, for example robin")
    parser.add_argument("recordings_dir", type=Path, help="Folder containing 5-10 WAV recordings")
    parser.add_argument("--profile-dir", type=Path, default=Path("identity/profiles"), help="Directory for saved .npy profiles")
    args = parser.parse_args()

    enroll_user(args.user_id, args.recordings_dir, profile_dir=args.profile_dir)


if __name__ == "__main__":
    main()
