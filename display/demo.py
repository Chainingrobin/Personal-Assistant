from __future__ import annotations

import time

from config import DisplayConfig
from display import advance, init_display, set_state, shutdown
from display.states import DisplayState, STATE_CONFIG


def main() -> None:
    init_display(DisplayConfig(backend="emulator"))
    try:
        for state in DisplayState:
            set_state(state)
            subtitle_steps = STATE_CONFIG.get(state, (None, []))[1]
            for _ in range(max(0, len(subtitle_steps) - 1)):
                time.sleep(1.0)
                advance()
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
