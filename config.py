from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _split_scopes(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default

    scopes = tuple(scope.strip() for scope in value.split(",") if scope.strip())
    return scopes or default


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AgentConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60.0"))
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
    api_path: str = os.getenv("OLLAMA_API_PATH", "/api/chat")
    provider: str = os.getenv("OLLAMA_PROVIDER", "ollama")
    hardware_profile: str = os.getenv("HARDWARE_PROFILE", "desktop")
    # Local/dev/headless toggling: use text when a working microphone is not available.
    input_mode: str = os.getenv("INPUT_MODE", "voice").strip().lower()

    def __post_init__(self) -> None:
        if self.input_mode not in {"voice", "text"}:
            raise ValueError("INPUT_MODE must be 'voice' or 'text'")


@dataclass(frozen=True)
class GoogleConfig:
    credentials_path: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials/credentials.json")
    token_dir: str = os.getenv("GOOGLE_TOKEN_DIR", "credentials/tokens")
    scopes: tuple[str, ...] = _split_scopes(
        os.getenv("GOOGLE_SCOPES"),
        (
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/calendar",
        ),
    )


@dataclass(frozen=True)
class VisionConfig:
    camera_backend: str = os.getenv("AEGIS_CAMERA_BACKEND", "auto")
    camera_fps: int = int(os.getenv("AEGIS_CAMERA_FPS", "10"))
    calibration_duration_sec: float = float(os.getenv("AEGIS_CALIBRATION_DURATION_SEC", "5.0"))
    yaw_tolerance_deg: float = float(os.getenv("AEGIS_YAW_TOLERANCE_DEG", "12.0"))
    pitch_tolerance_deg: float = float(os.getenv("AEGIS_PITCH_TOLERANCE_DEG", "10.0"))
    distraction_duration_threshold_sec: float = float(
        os.getenv("AEGIS_DISTRACTION_DURATION_THRESHOLD_SEC", "3.0")
    )
    escalation_event_count: int = int(os.getenv("AEGIS_ESCALATION_EVENT_COUNT", "3"))
    escalation_window_sec: float = float(os.getenv("AEGIS_ESCALATION_WINDOW_SEC", "300.0"))
    show_debug_window: bool = _as_bool(os.getenv("AEGIS_SHOW_DEBUG_WINDOW"), True)


@dataclass(frozen=True)
class DisplayConfig:
    backend: str = os.environ.get("AEGIS_DISPLAY_BACKEND", "emulator")
    i2c_port: int = int(os.environ.get("AEGIS_DISPLAY_I2C_PORT", "1"))
    i2c_address: int = int(os.environ.get("AEGIS_DISPLAY_I2C_ADDR", "0x3C"), 16)
    width: int = 128
    height: int = 64


@dataclass(frozen=True)
class TTSConfig:
    backend: str = os.getenv("TTS_BACKEND", "kokoro").strip().lower()
    model_path: str = os.getenv("TTS_MODEL_PATH", "models/tts/kokoro-v1.0.int8.onnx")
    voices_path: str = os.getenv("TTS_VOICES_PATH", "models/tts/voices-v1.0.bin")
    voice: str = os.getenv("TTS_VOICE", "af_sky")
    speed: float = float(os.getenv("TTS_SPEED", "1.2"))

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError("TTS_SPEED must be greater than zero")


AGENT_CONFIG = AgentConfig()
GOOGLE_CONFIG = GoogleConfig()
VISION_CONFIG = VisionConfig()
DISPLAY_CONFIG = DisplayConfig()
TTS_CONFIG = TTSConfig()
