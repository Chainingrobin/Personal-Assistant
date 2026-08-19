from __future__ import annotations

from config import DisplayConfig
from .factory import create_device
from .renderer import DisplayRenderer
from .states import DisplayState

_renderer: DisplayRenderer | None = None


def init_display(config: DisplayConfig | None = None) -> DisplayRenderer:
    global _renderer

    from config import DisplayConfig as Config

    final_config = config or Config()
    _renderer = DisplayRenderer(create_device(final_config), final_config)
    _renderer.set_state(DisplayState.IDLE)
    return _renderer


def set_state(state: DisplayState) -> None:
    if _renderer is None:
        init_display()
    _renderer.set_state(state)


def advance() -> None:
    if _renderer is None:
        init_display()
    _renderer.advance()


def shutdown() -> None:
    global _renderer
    if _renderer is not None:
        _renderer.shutdown()
        _renderer = None


__all__ = [
    "DisplayState",
    "init_display",
    "set_state",
    "advance",
    "shutdown",
]
