from __future__ import annotations

import logging
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from . import icons
from .states import DisplayState, STATE_CONFIG

logger = logging.getLogger(__name__)

# Icons are authored at 40x40 in icons.py. Displayed at native size (no
# nearest-neighbor scaling needed -- they were drawn at this resolution on
# purpose), positioned in the upper portion of the 128x64 canvas with the
# subtitle text below.
ICON_Y_OFFSET = 2

# A handful of common install locations across dev machines and Raspberry Pi
# OS, tried in order. Falls back to PIL's built-in bitmap font (small, but
# never crashes) if none of these exist on the current system.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]
_FONT_SIZE = 13
_FONT_SIZE_SMALL = 10  # used when the full-size subtitle would overflow the panel width


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    logger.warning(
        "No TTF font found at any known path; falling back to PIL's tiny "
        "default bitmap font. Subtitle text will look low-fidelity -- "
        "install one of %s on this system to fix.",
        _FONT_CANDIDATES,
    )
    return ImageFont.load_default()


def _validate_icon_grid(name: str, rows: Iterable[str]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Icon {name!r} is empty.")
    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"Icon {name!r} row {index} has length {len(row)}; expected {width}."
            )


for _name, _value in vars(icons).items():
    if isinstance(_value, list) and _value and all(isinstance(row, str) for row in _value):
        _validate_icon_grid(_name, _value)


class DisplayRenderer:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.current_state: DisplayState | None = None
        self.current_step = 0
        self.last_render_key: tuple[DisplayState | None, int, str] | None = None
        self._font = _load_font(_FONT_SIZE)
        self._font_small = _load_font(_FONT_SIZE_SMALL)

    def _safe_render(self, state: DisplayState | None, step_index: int) -> None:
        if self.device is None:
            return
        try:
            self._render_state(state, step_index)
        except Exception:
            logger.exception("Display render failed for state %s", state)
            try:
                self.current_state = DisplayState.ERROR
                self.current_step = 0
                self.last_render_key = None
                self._render_state(DisplayState.ERROR, 0)
            except Exception:
                logger.exception("Fallback render to ERROR state also failed.")

    def _subtitle_for(self, state: DisplayState | None) -> str:
        if state is None:
            return ""
        config = STATE_CONFIG.get(state)
        if config is None:
            return ""
        steps = config[1]
        if not steps:
            return ""
        return steps[min(self.current_step, len(steps) - 1)]

    def _icon_for(self, state: DisplayState | None):
        if state is None:
            return []
        config = STATE_CONFIG.get(state)
        if config is None:
            return []
        return config[0]

    def _draw_icon(self, canvas: Image.Image, icon_grid) -> int:
        """Draws the icon centered horizontally near the top of the canvas.
        Returns the y-coordinate immediately below the icon, so the caller
        knows where to start drawing the subtitle text."""
        if not icon_grid:
            return ICON_Y_OFFSET
        icon_w = len(icon_grid[0])
        icon_h = len(icon_grid)
        x0 = (self.config.width - icon_w) // 2
        y0 = ICON_Y_OFFSET
        for row_index, row in enumerate(icon_grid):
            for col_index, char in enumerate(row):
                if char != "#":
                    continue
                canvas.putpixel((x0 + col_index, y0 + row_index), 255)
        return y0 + icon_h

    def _draw_text(self, canvas: Image.Image, text: str, top: int) -> None:
        if not text:
            return
        draw = ImageDraw.Draw(canvas)
        font = self._font
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        # Drop to the smaller font if the full-size subtitle would overflow
        # the panel width, rather than silently clipping the last character(s).
        if text_width > self.config.width - 4:
            font = self._font_small
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = max(0, (self.config.width - text_width) // 2)
        # Center the remaining space between the icon's bottom edge and the
        # panel's bottom edge, so the label doesn't hug the icon too tightly
        # or run off the bottom on the smaller 64px-tall panel.
        remaining = self.config.height - top
        y = top + max(0, (remaining - text_height) // 2)
        draw.text((x, y), text, font=font, fill=255)

    def _render_state(self, state: DisplayState | None, step_index: int) -> None:
        if state is None:
            return
        icon_grid = self._icon_for(state)
        steps = STATE_CONFIG.get(state, ([], []))[1]
        subtitle = steps[step_index] if steps else ""

        canvas = Image.new("1", (self.config.width, self.config.height), 0)
        icon_bottom = self._draw_icon(canvas, icon_grid)
        self._draw_text(canvas, subtitle, icon_bottom)

        if self.device is not None:
            self.device.display(canvas)

    def set_state(self, state: DisplayState) -> None:
        if state is None:
            return
        self.current_state = state
        self.current_step = 0
        subtitle = self._subtitle_for(state)
        render_key = (state, self.current_step, subtitle)
        if self.last_render_key == render_key:
            return
        self.last_render_key = render_key
        self._safe_render(state, 0)

    def advance(self) -> None:
        if self.current_state is None:
            return
        steps = STATE_CONFIG.get(self.current_state, (None, []))[1]
        if not steps:
            return
        next_step = min(self.current_step + 1, len(steps) - 1)
        if next_step == self.current_step:
            return
        self.current_step = next_step
        subtitle = self._subtitle_for(self.current_state)
        render_key = (self.current_state, self.current_step, subtitle)
        if self.last_render_key == render_key:
            return
        self.last_render_key = render_key
        self._safe_render(self.current_state, self.current_step)

    def shutdown(self) -> None:
        try:
            if self.device is not None:
                self.device.clear()
        except Exception:
            logger.exception("Display shutdown failed while clearing device.")
        finally:
            self.current_state = None
            self.current_step = 0

    def render_forced(self, state: DisplayState, step_index: int) -> None:
        self._safe_render(state, step_index)