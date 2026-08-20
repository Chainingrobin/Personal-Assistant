"""
DisplayState enum + per-state (icon, subtitle steps) mapping.
This is the single source of truth for what each state looks like.
Nothing outside this file should reference an icon name directly.
"""
from enum import Enum, auto
from . import icons


class DisplayState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    TOOL_GMAIL = auto()
    TOOL_CALENDAR = auto()
    TOOL_RAG = auto()
    STUDY_MODE = auto()
    CALIBRATING = auto()
    SWITCHING_USER = auto()
    SWITCH_SUCCESS = auto()
    SWITCH_FAILED = auto()
    DISTRACTION_ALERT = auto()
    ERROR = auto()


# state -> (icon bitmap, [subtitle steps in order])
STATE_CONFIG = {
    DisplayState.IDLE: (icons.IDLE, ["Aegis"]),
    DisplayState.LISTENING: (icons.LISTENING, ["Listening..."]),
    DisplayState.THINKING: (icons.THINKING, ["Reasoning...", "Generating..."]),
    DisplayState.TOOL_GMAIL: (icons.TOOL_GMAIL, ["Checking mail...", "Mail read"]),
    DisplayState.TOOL_CALENDAR: (icons.TOOL_CALENDAR, ["Reading events...", "Events read"]),
    DisplayState.TOOL_RAG: (icons.TOOL_RAG, ["Recalling...", "Recalled"]),
    DisplayState.STUDY_MODE: (icons.STUDY_MODE, ["Study Mode"]),
    DisplayState.CALIBRATING: (icons.CALIBRATING, [
        "Calibrating... 5", "Calibrating... 4", "Calibrating... 3",
        "Calibrating... 2", "Calibrating... 1", "Calibrated!",
    ]),
    DisplayState.SWITCHING_USER: (icons.SWITCHING_USER, ["Verifying voice..."]),
    DisplayState.SWITCH_SUCCESS: (icons.SWITCH_SUCCESS, ["Switched!"]),
    DisplayState.SWITCH_FAILED: (icons.ERROR, ["Switch failed"]),
    DisplayState.DISTRACTION_ALERT: (icons.DISTRACTION_ALERT, ["Stay focused!"]),
    DisplayState.ERROR: (icons.ERROR, ["Error"]),
}