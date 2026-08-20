"""Record enrollment clips through the REAL VAD capture path and save them
as WAV files into the userclips folder, ready for diagnose.py.

Why this exists: enrollment clips recorded in a separate app (Audacity, the
Windows Voice Recorder, etc.) go through a different mic/DSP/resampling chain
than the live wake-word pipeline captures at runtime. That mismatch is a
likely cause of enrollment-vs-live score gaps. This script uses the same
VADRecorder class main.py / live_test_verify.py use, so every enrollment
clip is captured exactly the way a live utterance will be.

Before running:
  - Disable Windows "Audio Enhancements" / Nahimic / RealTek DSP on your
    input device (Sound settings -> Input -> Device properties -> Additional
    device properties -> Enhancements tab -> Disable all).
  - Close any virtual audio software (Voicemeeter, etc.) that might sit
    between the mic and this script.

How duration targeting works:
  Your target phrase ("switch user youssef") is naturally ~2-2.5s, but you
  found scores are far more consistent once an utterance crosses ~4s. VAD
  only ends a recording on sustained silence -- it does NOT care how many
  words you said. So instead of speaking once and stopping, say the trigger
  phrase, then IMMEDIATELY (no pause) repeat it once or twice more before
  going quiet. That gives the VAD one continuous utterance comfortably past
  4 seconds while still being 100% real trigger-phrase audio, not filler
  speech. This script tells you the duration right after each take and lets
  you discard/retry before it's saved, so you're not stuck culling bad
  clips afterward.

Usage:
    python identity/enroll_record.py youssef
    python identity/enroll_record.py youssef --out-dir "D:\\GUC\\Personal_Assistant_Project\\Personal Assistant\\userclips\\youssef"
    python identity/enroll_record.py youssef --target-seconds 4.0 --count 15
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.vad_recorder import VADRecorder

DEFAULT_OUT_DIR = Path(r"D:\GUC\Personal_Assistant_Project\Personal Assistant\userclips\youssef")
DEFAULT_TARGET_SECONDS = 4.0
CLIP_NAME_RE = re.compile(r"^clip_(\d+)\.wav$", re.IGNORECASE)


def _next_clip_index(out_dir: Path) -> int:
    """Continue numbering from whatever's already in the folder instead of overwriting."""
    existing = 0
    if out_dir.exists():
        for path in out_dir.glob("*.wav"):
            match = CLIP_NAME_RE.match(path.name)
            if match:
                existing = max(existing, int(match.group(1)))
            else:
                existing += 0  # non-matching filenames don't affect numbering, just ignored
    return existing + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Record enrollment clips via the real VAD pipeline.")
    parser.add_argument("user_id", help="User these clips belong to, e.g. youssef")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                         help="Folder to save clips into (default: the youssef userclips folder)")
    parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS,
                         help="Clips shorter than this trigger a warning + re-record prompt")
    parser.add_argument("--count", type=int, default=None,
                         help="Stop automatically after saving this many new clips (default: run until Ctrl+C)")
    parser.add_argument("--vad-debug", action="store_true", help="Show verbose VAD frame-level logging")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    recorder = VADRecorder(debug=args.vad_debug)
    next_index = _next_clip_index(out_dir)
    saved_count = 0

    print(f"Saving to: {out_dir}")
    print(f"Next clip will be numbered: clip_{next_index}.wav")
    print(f"Target duration: {args.target_seconds:.1f}s+  "
          f"(say your trigger phrase, then immediately repeat it 1-2x without pausing)")
    print("Ctrl+C to stop at any time.\n")

    while args.count is None or saved_count < args.count:
        try:
            input(f"[{saved_count} saved this session] Press Enter, then speak >> ")
        except KeyboardInterrupt:
            print("\nStopping.")
            break

        print("  [listening]")
        result = recorder.record_utterance()
        if result is None:
            print("  [no speech detected] -- try again\n")
            continue

        audio = np.asarray(result.samples)
        duration = result.duration_seconds
        sample_rate = result.sample_rate
        peak = float(np.abs(audio).max()) if audio.size else 0.0

        print(f"  [captured] duration={duration:.2f}s  sr={sample_rate}  peak={peak:.3f}")

        if duration < args.target_seconds:
            print(f"  [warn] under the {args.target_seconds:.1f}s target -- "
                  f"try repeating the phrase 1-2 more times before going quiet.")
            choice = input("  Save anyway (s), retry without saving (r), or skip (Enter=retry) >> ").strip().lower()
            if choice != "s":
                print()
                continue
        else:
            choice = input("  Save this clip? (Y/n) >> ").strip().lower()
            if choice == "n":
                print()
                continue

        clip_path = out_dir / f"clip_{next_index}.wav"
        sf.write(clip_path, audio, sample_rate)
        print(f"  [saved] {clip_path}\n")

        next_index += 1
        saved_count += 1

    print(f"\nDone. Saved {saved_count} new clip(s) to {out_dir}")
    print("Next: re-run diagnose.py against this folder to rebuild the profile and check consistency:")
    print(f'    python identity/diagnose.py {args.user_id} "{out_dir}"')


if __name__ == "__main__":
    main()
