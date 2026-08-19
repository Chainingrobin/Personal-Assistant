"""Local, offline text-to-speech synthesis for Aegis using Piper."""

from __future__ import annotations

import io
import random
import re
import wave
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    sd = None
    _SOUNDDEVICE_ERROR = exc
else:
    _SOUNDDEVICE_ERROR = None

try:
    from piper import PiperVoice, SynthesisConfig
except ImportError as exc:
    PiperVoice = None
    SynthesisConfig = None
    _PIPER_ERROR = exc
else:
    _PIPER_ERROR = None

try:
    from scipy.signal import resample
except ImportError as exc:
    resample = None
    _SCIPY_ERROR = exc
else:
    _SCIPY_ERROR = None


# Change this later to your chosen deep male voice.
DEFAULT_MODEL_PATH = "models/tts/en_US-ryan-medium.onnx"


# ---------------------------------------------------------------------------
# Markdown cleanup — the LLM sometimes returns **bold**, [links](url), bullet
# dashes, etc. Piper has no idea these are formatting and will try to
# pronounce the literal symbols, which is what caused the robotic reading.
# ---------------------------------------------------------------------------

_MARKDOWN_PATTERNS = [
    (re.compile(r"\*\*(.*?)\*\*"), r"\1"),              # **bold**
    (re.compile(r"__(.*?)__"), r"\1"),                  # __bold__
    (re.compile(r"\*(.*?)\*"), r"\1"),                  # *italic*
    (re.compile(r"_(.*?)_"), r"\1"),                    # _italic_
    (re.compile(r"`{1,3}(.*?)`{1,3}", re.DOTALL), r"\1"),  # `code` / ```code```
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE), ""),  # # headers
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),      # [text](url) -> text
    (re.compile(r"https?://\S+"), "the link"),          # bare URLs -> spoken placeholder
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),    # bullet markers
]


def _strip_markdown(text: str) -> str:
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    # Collapse any doubled-up whitespace left behind by the substitutions.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pitch_shift_resample(samples: np.ndarray, n_steps: float) -> np.ndarray:
    """Shift pitch by resampling, not by librosa's STFT phase vocoder.

    ~4 microseconds vs ~1.7s for a 3s clip in local testing — the STFT
    approach was too slow to run on every spoken response, especially on
    a Pi 5. The trade-off: pitch and speed move together here (higher
    pitch = shorter/faster, lower pitch = longer/slower), same as a
    tape/record played at the wrong speed.

    n_steps=0 is a no-op and returns samples unchanged.
    """
    if n_steps == 0 or resample is None:
        return samples

    rate = 2.0 ** (n_steps / 12.0)
    new_length = max(1, int(len(samples) / rate))

    float_samples = samples.astype(np.float32)
    shifted = resample(float_samples, new_length)

    shifted = np.clip(shifted, -32768, 32767)
    return shifted.astype(np.int16)


class PiperSpeaker:
    """Load one Piper voice and change its delivery using moods.

    For moods with a REACTIONS entry, speak() synthesizes a short punchy
    reaction phrase first (e.g. "Congrats!" at a stronger pitch/speed),
    then the rest of the reply at a calmer, more human version of the same
    mood, and plays them back as one stitched clip. This is what makes
    "add a celebration" sound like a quick genuine reaction followed by
    normal happy speech, instead of the whole sentence shouted uniformly.
    """

    # Short reaction phrases per mood, spoken with REACTION_SETTINGS before
    # the main body. Moods not listed here just speak normally, unchanged.
    REACTIONS = {
        "excited": ["Congrats!", "Yay!", "That's exciting!", "Love that!"],
        "warning": ["Heads up.", "Okay."],
    }

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        device=None,
        debug=True,
    ):
        if sd is None:
            raise ImportError(
                "sounddevice is required for audio playback"
            ) from _SOUNDDEVICE_ERROR

        if PiperVoice is None or SynthesisConfig is None:
            raise ImportError(
                "piper-tts is required for speech synthesis"
            ) from _PIPER_ERROR

        if resample is None:
            raise ImportError(
                "scipy is required for pitch shifting"
            ) from _SCIPY_ERROR

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice model not found at {model_path}"
            )

        self.model_path = model_path
        self.device = device
        self.debug = debug

        self.voice = PiperVoice.load(str(model_path))

    def _get_mood_settings(self, mood):
        """Body-of-response settings — toned down from the old all-in
        values for moods that now also get a punchy reaction phrase, so
        the sustained speech sounds like natural happy/serious talking
        rather than shouting or monotone-warning for the whole reply.
        """

        moods = {
            "normal": {
                "length_scale": 1.0,
                "noise_scale": 0.667,
                "noise_w_scale": 0.8,
                "pitch_semitones": 0,
            },

            "greeting": {
                "length_scale": 1.05,
                "noise_scale": 0.70,
                "noise_w_scale": 0.82,
                "pitch_semitones": 1,
            },

            "excited": {
                # Softer than the reaction phrase — happy/warm, not shouted.
                "length_scale": 0.94,
                "noise_scale": 0.70,
                "noise_w_scale": 0.82,
                "pitch_semitones": 1,
            },

            "warning": {
                "length_scale": 1.12,
                "noise_scale": 0.58,
                "noise_w_scale": 0.68,
                "pitch_semitones": -1,
            },

            "slow": {
                "length_scale": 1.30,
                "noise_scale": 0.60,
                "noise_w_scale": 0.70,
                "pitch_semitones": -1,
            },
        }

        return moods.get(mood, moods["normal"])

    def _get_reaction_settings(self, mood):
        """Punchier settings for the short lead-in reaction phrase only."""

        reaction_moods = {
            "excited": {
                "length_scale": 0.76,
                "noise_scale": 0.80,
                "noise_w_scale": 0.92,
                "pitch_semitones": 3,
            },
            "warning": {
                "length_scale": 1.05,
                "noise_scale": 0.55,
                "noise_w_scale": 0.62,
                "pitch_semitones": -3,
            },
        }

        return reaction_moods.get(mood, self._get_mood_settings(mood))

    def _pick_reaction(self, mood):
        phrases = self.REACTIONS.get(mood)
        if not phrases:
            return None
        return random.choice(phrases)

    def _prepare_text(self, text, mood):
        """Adjust punctuation to make the delivery more natural."""

        text = text.strip()

        if mood == "excited":
            if text and not text.endswith(("!", "?")):
                text += "!"

        elif mood == "warning":
            if text and not text.endswith(
                (".", "!", "?")
            ):
                text += "."

        elif mood == "slow":
            text = text.replace(
                ". ",
                "... ",
            )

        elif mood == "greeting":
            if text and not text.endswith(
                (".", "!", "?")
            ):
                text += "."

        return text

    def _synthesize_to_wav_bytes(
        self,
        text,
        length_scale=1.0,
        noise_scale=0.667,
        noise_w_scale=0.8,
    ):
        """Synthesize text into WAV bytes."""

        buffer = io.BytesIO()

        syn_config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )

        with wave.open(buffer, "wb") as wav_file:
            self.voice.synthesize_wav(
                text,
                wav_file,
                syn_config,
            )

        return buffer.getvalue()

    def synthesize(
        self,
        text,
        length_scale=1.0,
        noise_scale=0.667,
        noise_w_scale=0.8,
        pitch_semitones=0,
    ):
        """Synthesize text and return samples plus sample rate."""

        wav_bytes = self._synthesize_to_wav_bytes(
            text,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
        )

        with wave.open(
            io.BytesIO(wav_bytes),
            "rb",
        ) as wav_file:

            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            raw_audio = wav_file.readframes(n_frames)

        samples = np.frombuffer(
            raw_audio,
            dtype=np.int16,
        )

        if pitch_semitones:
            samples = _pitch_shift_resample(samples, pitch_semitones)

        return samples, sample_rate

    def speak(self, text, mood="normal"):
        """Speak using one voice with a mood-dependent delivery.

        If the mood has a reaction phrase configured, this synthesizes a
        short punchy lead-in plus a calmer body and plays them as one
        stitched clip. Otherwise it behaves exactly as before: one
        synthesis pass for the whole text.
        """

        text = _strip_markdown(text)
        text = self._prepare_text(text, mood)

        if not text:
            if self.debug:
                print(
                    "[tts-debug] empty text, "
                    "skipping playback"
                )
            return

        reaction_phrase = self._pick_reaction(mood)

        if reaction_phrase:
            reaction_settings = self._get_reaction_settings(mood)
            body_settings = self._get_mood_settings(mood)

            if self.debug:
                print(
                    f"[tts-debug] reaction='{reaction_phrase}' "
                    f"(mood={mood}) "
                    f"speed={reaction_settings['length_scale']} "
                    f"pitch={reaction_settings['pitch_semitones']}"
                )
                print(
                    f"[tts-debug] body {len(text)} chars "
                    f"speed={body_settings['length_scale']} "
                    f"pitch={body_settings['pitch_semitones']}"
                )

            reaction_samples, sample_rate = self.synthesize(
                reaction_phrase,
                length_scale=reaction_settings["length_scale"],
                noise_scale=reaction_settings["noise_scale"],
                noise_w_scale=reaction_settings["noise_w_scale"],
                pitch_semitones=reaction_settings["pitch_semitones"],
            )
            body_samples, body_sample_rate = self.synthesize(
                text,
                length_scale=body_settings["length_scale"],
                noise_scale=body_settings["noise_scale"],
                noise_w_scale=body_settings["noise_w_scale"],
                pitch_semitones=body_settings["pitch_semitones"],
            )

            # Small breathing gap between the reaction and the body so it
            # reads as two distinct beats of speech, not one run-on clip.
            gap = np.zeros(int(sample_rate * 0.25), dtype=np.int16)
            samples = np.concatenate([reaction_samples, gap, body_samples])

        else:
            settings = self._get_mood_settings(mood)

            if self.debug:
                print(
                    f"[tts-debug] synthesizing "
                    f"{len(text)} chars "
                    f"(mood={mood})"
                )
                print(
                    f"[tts-debug] "
                    f"speed={settings['length_scale']} "
                    f"noise={settings['noise_scale']} "
                    f"variation={settings['noise_w_scale']} "
                    f"pitch={settings['pitch_semitones']}"
                )

            samples, sample_rate = self.synthesize(
                text,
                length_scale=settings["length_scale"],
                noise_scale=settings["noise_scale"],
                noise_w_scale=settings["noise_w_scale"],
                pitch_semitones=settings["pitch_semitones"],
            )

        if self.debug:
            duration = (
                samples.shape[0]
                / float(sample_rate)
            )

            print(
                "[tts-debug] playing "
                + str(round(duration, 2))
                + "s audio"
            )

        sd.play(
            samples,
            samplerate=sample_rate,
            device=self.device,
        )

        sd.wait()