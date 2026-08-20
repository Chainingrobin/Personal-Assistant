"""Compare the frequency content of two WAV clips (e.g. one from
enroll_record.py vs one from Windows Sound Recorder) to see whether a
background tone/buzz is present, and if so, roughly what frequency it's at.

This does NOT fix anything -- it's purely diagnostic, to tell you whether
what you're hearing is:
  (a) a narrow, sustained tone at a specific frequency (classic sign of
      electrical/ground-loop noise or a driver-level sample-rate-conversion
      whine), or
  (b) broadband noise (fan, room hum, mic self-noise), or
  (c) not actually there in the digital signal at all (i.e. it's monitoring
      feedback / headphones / something outside the recording chain).

Usage:
    python identity/inspect_audio.py path/to/enroll_record_clip.wav
    python identity/inspect_audio.py path/to/enroll_record_clip.wav --compare path/to/sound_recorder_clip.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _load(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


def _top_tones(audio: np.ndarray, sr: int, n: int = 8, min_hz: float = 3000.0) -> list[tuple[float, float]]:
    """Return the top N spectral peaks above min_hz with their relative magnitude in dB.

    Restricting to >3kHz on purpose: speech fundamentals + most formants sit
    below this, so peaks up here are the most likely candidates for an
    audible 'high-pitched' artifact rather than normal voiced speech.
    """
    windowed = audio * np.hanning(len(audio))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    spectrum_db = 20 * np.log10(spectrum + 1e-9)

    mask = freqs >= min_hz
    freqs_masked = freqs[mask]
    spectrum_masked = spectrum_db[mask]

    if len(spectrum_masked) == 0:
        return []

    peak_db = spectrum_masked.max()
    order = np.argsort(spectrum_masked)[::-1]

    results = []
    seen_bands = set()
    for idx in order:
        freq = freqs_masked[idx]
        band = round(freq / 50) * 50  # dedupe peaks within the same ~50Hz neighborhood
        if band in seen_bands:
            continue
        seen_bands.add(band)
        rel_db = spectrum_masked[idx] - peak_db
        results.append((float(freq), float(rel_db)))
        if len(results) >= n:
            break
    return results


def _report(label: str, path: Path) -> None:
    audio, sr = _load(path)
    duration = len(audio) / float(sr)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.abs(audio).max())

    print(f"\n=== {label}: {path.name} ===")
    print(f"  sample_rate={sr}  duration={duration:.2f}s  rms={rms:.4f}  peak={peak:.4f}")

    # Look at a silent-ish region if possible (first 0.3s) to isolate background noise
    # from speech -- speech will dominate the full-clip spectrum otherwise.
    lead_in = audio[: int(0.3 * sr)]
    if len(lead_in) > 512:
        print("  Top tones in first 0.3s (likely background-only, before speech starts):")
        for freq, rel_db in _top_tones(lead_in, sr):
            marker = "  <-- dominant" if rel_db == 0.0 else ""
            print(f"    {freq:8.1f} Hz   {rel_db:6.1f} dB rel{marker}")
    else:
        print("  [clip too short to isolate a lead-in silence window]")

    print("  Top tones across full clip:")
    for freq, rel_db in _top_tones(audio, sr):
        marker = "  <-- dominant" if rel_db == 0.0 else ""
        print(f"    {freq:8.1f} Hz   {rel_db:6.1f} dB rel{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a WAV clip's frequency content for background tones.")
    parser.add_argument("clip", type=Path, help="Path to a WAV clip, e.g. from enroll_record.py")
    parser.add_argument("--compare", type=Path, default=None,
                         help="Optional second clip to compare against, e.g. an old Sound Recorder clip")
    args = parser.parse_args()

    _report("Clip A", args.clip)
    if args.compare is not None:
        _report("Clip B", args.compare)
        print(
            "\nRead this as: if Clip A (enroll_record.py) shows a strong, narrow peak "
            "(small |dB| relative to dominant, i.e. close to 0) at some frequency that "
            "Clip B does NOT show at a similar relative level, that's likely an artifact "
            "specific to the enroll_record.py capture path (driver resampling, USB power "
            "noise, etc.) rather than something inherent to your voice or environment."
        )


if __name__ == "__main__":
    main()
