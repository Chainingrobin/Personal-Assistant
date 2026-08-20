"""Speaker verification and active-user management for Aegis.

This module wraps SpeechBrain's ECAPA-TDNN encoder (`spkrec-ecapa-voxceleb`)
for 1:1 speaker verification against enrolled profile embeddings. Install with
`pip install speechbrain soundfile scipy`. The first run downloads model weights
into the local Torch/Hugging Face cache, typically under `~/.cache/torch` and
`~/.cache/huggingface`. The encoder is loaded once when :class:`SpeakerVerifier`
is created; verification then runs sequentially after STT, never concurrently
with Whisper or the LLM.

CHANGE LOG (consistency fixes):
- Replaced naive `np.interp` linear resampling with `scipy.signal.resample_poly`
  (polyphase filter, includes anti-aliasing). Linear interpolation does not
  band-limit before downsampling and was smearing high-frequency content on
  the 44.1kHz -> 16kHz conversion, shifting embeddings versus audio captured
  natively at 16kHz. Prefer capturing at 16kHz directly in VADRecorder so this
  path is a no-op at runtime; this fallback now only matters for legacy clips.
"""
from __future__ import annotations

import os
import re
from math import gcd
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - import guard
    sf = None  # type: ignore[assignment]

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard
    torch = None  # type: ignore[assignment]
    _TORCH_ERROR = exc
else:
    _TORCH_ERROR = None

try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError as exc:  # pragma: no cover - import guard
    EncoderClassifier = None  # type: ignore[assignment]
    _SPEECHBRAIN_ERROR = exc
else:
    _SPEECHBRAIN_ERROR = None

try:
    from scipy.signal import resample_poly
except ImportError as exc:  # pragma: no cover - import guard
    resample_poly = None  # type: ignore[assignment]
    _SCIPY_ERROR = exc
else:
    _SCIPY_ERROR = None

DEFAULT_KNOWN_USERS = ("youssef", "robin", "mariam")
PROFILE_DIR = Path("identity/profiles")
DEFAULT_THRESHOLD = float(os.getenv("AEGIS_VOICE_ID_THRESHOLD", "0.65"))
_CURRENT_USER = os.getenv("AEGIS_DEFAULT_USER", DEFAULT_KNOWN_USERS[0])


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _ensure_mono_float32(audio_buffer: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio_buffer)
    if audio.ndim == 2:
        audio = audio[:, 0]
    if audio.dtype.kind in {"i", "u"}:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32, copy=False)
    return np.ascontiguousarray(audio.reshape(-1))


def _resample_poly_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Anti-aliased polyphase resample. Replaces the old naive linear interp.

    Falls back to linear interpolation only if scipy is unavailable, with a
    loud warning, since that path is known to degrade embedding quality.
    """
    if source_rate == target_rate or audio.size == 0:
        return audio
    if resample_poly is None:
        print(
            "[voice_id] WARNING: scipy not installed, falling back to naive linear "
            "resampling. This will hurt embedding consistency -- `pip install scipy`."
        )
        duration = audio.shape[0] / float(source_rate)
        target_count = max(1, int(duration * target_rate))
        source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, duration, num=target_count, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)
    g = gcd(source_rate, target_rate)
    up, down = target_rate // g, source_rate // g
    return resample_poly(audio, up, down).astype(np.float32)


def get_known_user_ids() -> tuple[str, ...]:
    """Return the set of switchable user IDs known to the deployment.

    Output includes the default demo users plus any saved profile filenames.
    """
    profile_names = []
    if PROFILE_DIR.exists():
        profile_names = [path.stem for path in PROFILE_DIR.glob("*.npy")]
    merged = list(dict.fromkeys([*DEFAULT_KNOWN_USERS, *profile_names]))
    return tuple(merged)


def set_current_user(user_id: str) -> None:
    """Set the currently active user_id used for normal command turns."""
    global _CURRENT_USER
    if user_id not in get_known_user_ids():
        print(f"[identity] Warning: '{user_id}' is not in the known user list {get_known_user_ids()}")
    _CURRENT_USER = user_id


def get_current_user() -> str:
    """Return the active user_id for the next command turn."""
    return _CURRENT_USER


class SpeakerVerifier:
    """Load ECAPA once and compute embeddings or verification scores.

    The public methods take mono audio buffers and sample rates. Embedding
    extraction is the expensive step and takes roughly 0.3-0.8 s for a short
    clip on a Pi 5 CPU, depending on clip length and system load.
    """

    def __init__(
        self,
        profile_dir: Path = PROFILE_DIR,
        threshold: float = DEFAULT_THRESHOLD,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
    ):
        if EncoderClassifier is None:
            raise ImportError("speechbrain is required for speaker verification") from _SPEECHBRAIN_ERROR
        if torch is None:
            raise ImportError("torch is required for speaker verification") from _TORCH_ERROR
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.model_source = model_source
        self.encoder = EncoderClassifier.from_hparams(source=model_source)

    def _prepare_audio(self, audio_buffer: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = _ensure_mono_float32(audio_buffer)
        audio = _resample_poly_audio(audio, sample_rate, 16000)
        return audio

    def embed_audio(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Convert one utterance into a normalized ECAPA embedding vector.

        Input: mono audio buffer plus sample_rate. Output: 1D float32 embedding.
        """
        audio = self._prepare_audio(audio_buffer, sample_rate)
        if audio.size == 0:
            raise ValueError("Cannot embed an empty audio buffer")
        tensor = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            embedding = self.encoder.encode_batch(tensor).squeeze().detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding = embedding / norm
        return embedding

    def load_profile_embedding(self, user_id: str) -> np.ndarray:
        """Load the saved averaged profile vector for one user_id."""
        profile_path = self.profile_dir / f"{user_id}.npy"
        if not profile_path.exists():
            raise FileNotFoundError(f"No saved profile found for user '{user_id}' at {profile_path}")
        embedding = np.load(profile_path).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding = embedding / norm
        return embedding

    def similarity(self, claimed_user_id: str, audio_buffer: np.ndarray, sample_rate: int = 16000) -> float:
        """Return cosine similarity between a live utterance and one user profile."""
        profile = self.load_profile_embedding(claimed_user_id)
        live_embedding = self.embed_audio(audio_buffer, sample_rate=sample_rate)
        denominator = float(np.linalg.norm(profile) * np.linalg.norm(live_embedding))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(profile, live_embedding) / denominator)

    def verify_speaker(
        self,
        claimed_user_id: str,
        audio_buffer: np.ndarray,
        sample_rate: int = 16000,
        threshold: float | None = None,
    ) -> tuple[bool, float]:
        """Verify whether the audio matches the claimed user profile.

        Output: (is_match, cosine_similarity). The threshold defaults to the
        deployment-wide AEGIS_VOICE_ID_THRESHOLD or 0.70 if unset.
        """
        active_threshold = self.threshold if threshold is None else threshold
        score = self.similarity(claimed_user_id, audio_buffer, sample_rate=sample_rate)
        return score >= active_threshold, score


def switch_user(
    claimed_user_id: str,
    audio_buffer: np.ndarray,
    sample_rate: int = 16000,
    threshold: float | None = None,
    verifier: SpeakerVerifier | None = None,
    force_verify: bool = False,
) -> tuple[bool, float]:
    """Verify the claim and update the active user on success.

    Input: claimed user_id plus the same utterance buffer that produced the
    switch request. Output: (verified, similarity_score). No heavy models are
    loaded here if a verifier instance is provided by the caller.

    If claimed_user_id is already the active user, this short-circuits before
    running ECAPA at all -- no point re-verifying someone who's already
    selected, and it avoids the (small but nonzero) cost of an embedding pass
    on the Pi for a no-op switch. Score is reported as 1.0 in this case since
    no real comparison was made.

    Pass force_verify=True to always run the real ECAPA comparison even if
    claimed_user_id matches the current user -- useful for testing whether a
    disguised/altered voice would still pass against your own profile,
    without needing a second enrolled user.
    """
    if claimed_user_id == get_current_user() and not force_verify:
        print(f"[identity] '{claimed_user_id}' is already the active user — no switch needed")
        return True, 1.0
    verifier = verifier or SpeakerVerifier()
    verified, score = verifier.verify_speaker(
        claimed_user_id,
        audio_buffer,
        sample_rate=sample_rate,
        threshold=threshold,
    )
    if verified:
        set_current_user(claimed_user_id)
    return verified, score