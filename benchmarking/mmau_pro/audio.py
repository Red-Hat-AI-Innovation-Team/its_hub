"""Turn local audio files into OpenAI-style chat content parts.

Two modes (no local decode/resample needed — vLLM handles resampling server-side):
- "local-path": reference the file by a file:// URL (vLLM must be started with
  --allowed-local-media-path covering the data dir). Avoids huge base64 bodies; best
  for long clips. Uses the {"type":"audio_url", ...} content part.
- "base64": embed the raw file bytes as base64 via {"type":"input_audio", ...}.
"""

import base64
import os

_B64_CACHE: dict[str, str] = {}


def _fmt(path: str) -> str:
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return ext or "wav"


def _b64(path: str) -> str:
    if path not in _B64_CACHE:
        with open(path, "rb") as f:
            _B64_CACHE[path] = base64.b64encode(f.read()).decode("ascii")
    return _B64_CACHE[path]


def audio_content_parts(audio_paths: list[str], mode: str = "local-path") -> list[dict]:
    """Build one content part per audio clip, preserving order."""
    parts: list[dict] = []
    for p in audio_paths:
        if mode == "base64":
            parts.append(
                {"type": "input_audio", "input_audio": {"data": _b64(p), "format": _fmt(p)}}
            )
        elif mode == "local-path":
            parts.append(
                {"type": "audio_url", "audio_url": {"url": "file://" + os.path.abspath(p)}}
            )
        else:
            raise ValueError(f"audio mode must be 'local-path' or 'base64', got {mode!r}")
    return parts
