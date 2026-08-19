from __future__ import annotations

import logging

from config import DisplayConfig

logger = logging.getLogger(__name__)


def create_device(config: DisplayConfig | None = None):
    cfg = config or DisplayConfig()
    backend = (cfg.backend or "emulator").lower()

    if backend == "ssd1306":
        try:
            from luma.core.interface.serial import i2c
            from luma.oled.device import ssd1306

            serial = i2c(port=cfg.i2c_port, address=cfg.i2c_address)
            return ssd1306(serial, width=cfg.width, height=cfg.height)
        except Exception:
            logger.exception("SSD1306 init failed; falling back to emulator backend.")
            backend = "emulator"

    if backend == "emulator":
        try:
            from luma.emulator.device import pygame

            return pygame(width=cfg.width, height=cfg.height)
        except Exception:
            logger.exception("Emulator init failed; no display backend available.")
            raise RuntimeError("No display backend could be initialized.") from None

    raise ValueError(f"Unsupported display backend: {backend!r}")
