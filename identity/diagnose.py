"""Enroll a user and run diagnostics on their clips, in one pass.

Usage:
    python identity/diagnose.py youssef userclips/youssef

What it does:
  1. Enrolls normally (averages ALL clips -> saves identity/profiles/youssef.npy)
     This overwrites any previous profile for this user_id -- no "unenroll" needed.
  2. Runs leave-one-out diagnostics: for each clip, builds a profile from the
     OTHER clips and checks how well the held-out clip matches. This tells you
     how internally consistent your recordings are, without needing a second
     person's voice yet.
  3. Prints per-clip scores, min/max/avg, and flags any clip that looks like
     an outlier (bad mic moment, mumbled take, etc).

Re-run this any time you add/replace clips in the folder -- it just overwrites
the saved profile, safe to run repeatedly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identity.voice_id import SpeakerVerifier


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sample_rate)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0


def diagnose(user_id: str, clips_dir: Path, profile_dir: Path, threshold: float) -> None:
    verifier = SpeakerVerifier(profile_dir=profile_dir)
    wav_paths = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() == ".wav")

    if not wav_paths:
        print(f"No WAV files found in {clips_dir}")
        return
    if len(wav_paths) < 3:
        print(f"Only {len(wav_paths)} clips found -- need at least 3 for leave-one-out diagnostics")
        return

    print(f"Found {len(wav_paths)} clips for '{user_id}'\n")

    # Step 1: embed every clip once
    embeddings = {}
    for path in wav_paths:
        audio, sample_rate = _load_wav(path)
        duration = len(audio) / float(sample_rate)
        emb = verifier.embed_audio(audio, sample_rate=sample_rate)
        embeddings[path.name] = emb
        print(f"[embed] {path.name} ({duration:.1f}s)")

    # Step 2: real enrollment -- average ALL clips, save profile (overwrites old one)
    all_embs = np.stack(list(embeddings.values()), axis=0)
    full_profile = _normalize(np.mean(all_embs, axis=0).astype(np.float32))
    profile_dir.mkdir(parents=True, exist_ok=True)
    out_path = profile_dir / f"{user_id}.npy"
    np.save(out_path, full_profile)
    print(f"\n[enroll] saved profile -> {out_path} ({len(embeddings)} clips averaged)\n")

    # Step 3: leave-one-out diagnostics
    print("--- Leave-one-out consistency check ---")
    names = list(embeddings.keys())
    scores = []
    for held_out_name in names:
        others = [emb for name, emb in embeddings.items() if name != held_out_name]
        loo_profile = _normalize(np.mean(np.stack(others, axis=0), axis=0).astype(np.float32))
        score = _cosine(loo_profile, embeddings[held_out_name])
        scores.append(score)
        flag = ""
        if score < threshold:
            flag = "  <-- BELOW THRESHOLD (outlier clip? re-record this one)"
        print(f"  {held_out_name:30s} score={score:.4f}{flag}")

    scores_arr = np.array(scores)
    print(f"\nSummary: min={scores_arr.min():.4f}  max={scores_arr.max():.4f}  "
          f"avg={scores_arr.mean():.4f}  threshold={threshold:.2f}")
    if scores_arr.min() < threshold:
        print("-> At least one clip is dragging profile quality down. Consider re-recording it "
              "or dropping it from the folder and re-running this script.")
    else:
        print("-> All clips are internally consistent. Profile looks solid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a user and diagnose clip consistency.")
    parser.add_argument("user_id", help="User to enroll, e.g. youssef")
    parser.add_argument("clips_dir", type=Path, help="Folder of WAV clips, e.g. userclips/youssef")
    parser.add_argument("--profile-dir", type=Path, default=Path("identity/profiles"))
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    diagnose(args.user_id, args.clips_dir, args.profile_dir, args.threshold)


if __name__ == "__main__":
    main()