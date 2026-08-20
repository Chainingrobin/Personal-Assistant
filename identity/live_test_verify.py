"""Live-microphone speaker verification test using the repo's real VAD pipeline.

Uses VADRecorder (the same silence-based end-of-utterance detection the
wake-word pipeline uses in production) instead of a manual Enter-to-stop
toggle -- so this test exercises the exact same audio capture path main.py
does, not a diverging stand-in.

Speak whenever you see "[listening]" -- it starts recording immediately and
stops automatically once you go silent for the configured duration. Runs in
a loop so you can test several utterances back to back.

This calls the real switch_user() from voice_id.py, so it also exercises
the "already active user" short-circuit -- if you claim the user that's
already active, you'll see the no-op message instead of a fresh score.

CHANGE LOG:
  - Reports utterance duration alongside the score and flags short (<3s)
    utterances, since short "switch user X" phrases are the primary source
    of score instability -- this makes that trade-off visible per-attempt
    instead of only showing up as unexplained variance.
  - No change to the verification logic itself; this file's correctness
    depended on voice_id.py's resampling, which has been fixed there
    (polyphase instead of naive linear interpolation).

Usage:
    python identity/live_test_verify.py robin
    python identity/live_test_verify.py robin --threshold 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identity.voice_id import SpeakerVerifier, get_current_user, switch_user
from audio.vad_recorder import VADRecorder

SHORT_UTTERANCE_SECONDS = 3.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Live mic test for speaker verification using real VAD capture.")
    parser.add_argument("user_id", help="Claimed user_id, e.g. robin")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold")
    parser.add_argument("--profile-dir", type=Path, default=Path("identity/profiles"))
    parser.add_argument("--vad-debug", action="store_true", help="Show verbose VAD frame-level logging")
    parser.add_argument("--force", action="store_true",
                         help="Always run real ECAPA comparison, even if claimed user is already active "
                              "(use this to test whether a disguised voice still matches your own profile)")
    args = parser.parse_args()

    print("Loading ECAPA model (first run may take a moment)...")
    verifier = SpeakerVerifier(profile_dir=args.profile_dir)
    print("Model loaded.\n")

    recorder = VADRecorder(debug=args.vad_debug)
    print(f"Ready. Active user is currently: '{get_current_user()}'")
    print(f"Claiming identity: '{args.user_id}'")
    print("Ctrl+C to quit.\n")

    active_threshold = args.threshold if args.threshold is not None else verifier.threshold

    while True:
        try:
            input("Press Enter, then speak >> ")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        print("  [listening]")
        result = recorder.record_utterance()
        print("  [utterance captured]" if result else "  [no speech detected]")
        if result is None:
            print()
            continue

        audio = result.samples
        duration = result.duration_seconds
        print(f"  [debug] dtype={audio.dtype} shape={audio.shape} sr={result.sample_rate} "
              f"peak={np.abs(audio).max()} rms={np.sqrt(np.mean(audio.astype(np.float64)**2)):.1f}")

        verified, score = switch_user(
            args.user_id, audio, sample_rate=result.sample_rate,
            threshold=args.threshold, verifier=verifier, force_verify=args.force,
        )

        duration_flag = ""
        if duration < SHORT_UTTERANCE_SECONDS:
            duration_flag = f"  <-- under {SHORT_UTTERANCE_SECONDS:.0f}s, expect more score variance"

        print(f"  Duration:  {duration:.1f}s{duration_flag}")
        print(f"  Score:     {score:.4f}")
        print(f"  Threshold: {active_threshold:.2f}")
        print(f"  Result:    {'MATCH -- switched to ' + args.user_id if verified else 'NO MATCH'}")
        print(f"  Active user is now: '{get_current_user()}'\n")


if __name__ == "__main__":
    main()