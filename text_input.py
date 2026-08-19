"""Text input front end for local, development, and headless runs."""

from __future__ import annotations


class TextInputSource:
    """Read one transcript-shaped string at a time from stdin."""

    def read_transcript(self) -> str | None:
        """Return a stripped transcript, or None when stdin reaches EOF."""
        try:
            return input("> ").strip()
        except EOFError:
            return None