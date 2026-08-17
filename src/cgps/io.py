"""Deterministic JSON output helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_result(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write a UTF-8 JSON result and return its resolved path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, destination)
    return destination
