#!/usr/bin/env python3
"""Stable entry point for the camera-ready CGPS evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from camera_ready_runner import main  # noqa: E402, I001


if __name__ == "__main__":
    main()
