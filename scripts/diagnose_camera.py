#!/usr/bin/env python3
"""One-command camera diagnostics for the Raspberry Pi and webcam paths.

Run this from the activated repo venv. It prints the Python interpreter details,
OpenCV import state, Picamera2 import behavior, rpicam-vid discovery, camera
enumeration, /dev/video nodes, and the current user's group membership.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


RESULTS: list[tuple[str, bool, str]] = []


def record(label: str, passed: bool, details: str) -> None:
    RESULTS.append((label, passed, details))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {details}")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def main() -> int:
    print("== Python ==")
    python_version_result = run_command(["python3", "--version"])
    python_version = (python_version_result.stdout or python_version_result.stderr).strip()
    executable = str(Path(sys.executable).resolve())
    in_venv = sys.prefix != sys.base_prefix or "VIRTUAL_ENV" in os.environ
    print(f"python3 --version: {python_version}")
    print(f"sys.executable: {executable}")
    print(f"sys.prefix: {sys.prefix}")
    print(f"sys.base_prefix: {sys.base_prefix}")
    print(f"sys.version: {sys.version.split()[0]}")
    record(
        "venv Python is 3.11.9",
        sys.version.split()[0] == "3.11.9" and in_venv,
        f"python3={python_version or '<no output>'}, sys.executable={executable}, sys.prefix={sys.prefix}",
    )

    print("\n== Imports ==")
    try:
        import cv2  # type: ignore

        record("cv2 import", True, f"cv2 {cv2.__version__}")
    except Exception as exc:  # pragma: no cover - environment-dependent
        record("cv2 import", False, repr(exc))

    try:
        import picamera2  # type: ignore

        record("picamera2 import fails in this venv", False, f"unexpected success: {picamera2!r}")
    except Exception as exc:  # pragma: no cover - expected on this venv
        record("picamera2 import fails in this venv", True, f"expected failure: {type(exc).__name__}: {exc}")

    print("\n== rpicam-vid ==")
    rpicam_path = shutil.which("rpicam-vid")
    record("rpicam-vid on PATH", rpicam_path is not None, rpicam_path or "not found")

    if rpicam_path is not None:
        list_cameras = run_command([rpicam_path, "--list-cameras"])
        combined_output = (list_cameras.stdout + list_cameras.stderr).strip()
        print("rpicam-vid --list-cameras output:")
        print(combined_output or "<no output>")
        detected_camera = any(line.lstrip()[:1].isdigit() and ":" in line for line in combined_output.splitlines())
        record(
            "rpicam-vid detects at least one camera",
            list_cameras.returncode == 0 and detected_camera,
            f"returncode={list_cameras.returncode}",
        )
    else:
        record("rpicam-vid detects at least one camera", False, "skipped because rpicam-vid was not found")

    print("\n== Device nodes ==")
    video_nodes = sorted(glob.glob("/dev/video*"))
    if video_nodes:
        listing = run_command(["ls", "-l", *video_nodes])
        print(listing.stdout.rstrip() or listing.stderr.rstrip() or "<no output>")
        record("/dev/video* listing", listing.returncode == 0, f"{len(video_nodes)} node(s) found")
    else:
        print("No /dev/video* nodes found")
        record("/dev/video* listing", True, "no /dev/video* nodes found")

    groups_result = run_command(["groups"])
    groups_output = (groups_result.stdout or groups_result.stderr).strip()
    print(f"groups: {groups_output or '<no output>'}")
    in_video_group = " video " in f" {groups_output} " if groups_output else False
    record(
        "current user is in video group",
        in_video_group,
        groups_output or "no groups output",
    )

    print("\n== Summary ==")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    for label, ok, details in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}: {label} -> {details}")
    print(f"Overall: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
